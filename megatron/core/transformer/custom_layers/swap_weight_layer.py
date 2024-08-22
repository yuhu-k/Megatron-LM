from typing import Any, Callable, List, Mapping, Optional, OrderedDict
from swap_manager import get_weight_swapper

import torch
import torch.distributed
import transformer_engine as te
from torch import Tensor

from megatron.core import ModelParallelConfig, tensor_parallel
from megatron.core.tensor_parallel.layers import VocabParallelEmbedding
from megatron.core.transformer.custom_layers.swap_weight_function import SwapTELinear
from megatron.core.transformer.custom_layers.transformer_engine import _get_extra_te_kwargs
from megatron.core.transformer.transformer_config import TransformerConfig
from torchao.dtypes.nf4tensor import linear_nf4, to_nf4


def log_gpu_memory_usage():
    with open("/tmp2/yuhu/result.txt", "a") as f:
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        f.write(f"Allocated GPU memory: {allocated:.2f} GB\n")
        f.write(f"Reserved GPU memory: {reserved:.2f} GB\n")
    

def swap_weight_to_gpu(weight_id, profile):
    if profile:
        torch.cuda.nvtx.range_push(f"swap weight to cuda")
    # if "cuda" not in str(weight.device):
    #     weight = weight.to(torch.cuda.current_device()).detach()
    # else:
    #     weight = weight.detach()
    weight_swapper = get_weight_swapper()
    weight = weight_swapper.get_weight(weight_id)
    if profile:
        torch.cuda.nvtx.range_pop()
    torch.cuda.empty_cache()
    if torch.cuda.is_available() and torch.distributed.get_rank() == 0:
        log_gpu_memory_usage()
    return weight

class SwapWeightTemplate:
    def init(self, weight_size, config, swap_weight=True, params_dtype=None):
        self.swap_weight = swap_weight and config.swap_weight
        self.config = config
        
        if self.swap_weight:
            self.__delete_all_weight()
            device = "cpu"
            require_grad = False
            if params_dtype == None:
                params_dtype = self.params_dtype
            self.register_parameter("weight", torch.nn.Parameter(torch.empty(weight_size, device=device, requires_grad=require_grad, dtype=params_dtype)))
            self.profile = config.profile
            
            weight_swapper = get_weight_swapper()
            self.swap_weight_id = weight_swapper.register(device=torch.cuda.current_device())
            
    def __delete_all_weight(self):
        self._parameters.pop("weight")
        self.weight = None
        del self.weight
        
    def _set_weight(self, weight):
        self.register_parameter("weight", torch.nn.Parameter(weight))
        
    def _load_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs, _load_from_state_dict_func):
        result = _load_from_state_dict_func(state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs)
        if self.swap_weight:
            weight_swapper = get_weight_swapper()
            weight_swapper.update_weight(self.weight, self.swap_weight_id)
            self.__delete_all_weight()
            self._set_weight(weight_swapper.get_weight_cpu(self.swap_weight_id))
            self.requires_grad_(False)
        return result
    
    def _forward(self, inp, forward_function, weight=None, weight_id=None):
        if weight_id != None:
            output = forward_function(inp, None, weight_id)
        else:
            w_id = weight
            weight = self.get_weight(w_id) if self.swap_weight else weight
            if weight != None:
                output = forward_function(inp, weight)
            else:
                output = forward_function(inp)
        if weight_id is not None or self.swap_weight:
            swapper = get_weight_swapper()
            swapper.offload_weight(weight_id if weight_id is not None else w_id if w_id is not None else self.swap_weight_id)
            
        return output
        
    def get_weight(self, weight_id=None):
        if weight_id == None:
            weight_id = self.swap_weight_id
        #log_gpu_memory_usage()
        weight = swap_weight_to_gpu(weight_id, self.profile)
        return weight

    

class SwapWeightLinear(SwapTELinear):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        parallel_mode: str,
        config: ModelParallelConfig,
        init_method: Callable,
        bias: bool,
        skip_bias_add: bool,
        skip_weight_param_allocation: bool,
        tp_comm_buffer_name: str = None,
        swap_weight: bool = True,
        quantize: bool = False
    ):
        super().__init__(
            input_size=input_size,
            output_size=output_size,
            parallel_mode=parallel_mode,
            config=config,
            init_method=init_method,
            bias=bias,
            skip_bias_add=skip_bias_add,
            skip_weight_param_allocation=skip_weight_param_allocation,
            tp_comm_buffer_name=tp_comm_buffer_name,
        )
        self.swap_weight = swap_weight and config.swap_weight
        self.quantize = quantize
        self.config = config
        
        
        if self.quantize or self.swap_weight:
            weight_size = (output_size, input_size)
            self.__delete_all_weight()
            params_dtype = self.params_dtype
            if self.swap_weight:
                weight_swapper = get_weight_swapper()
                self.swap_weight_id = weight_swapper.register(device=torch.cuda.current_device())
                device = "cpu"
            else:
                device = torch.cuda.current_device()

            self.register_parameter("weight", torch.nn.Parameter(self._create_weight_and_bias(weight_size, device, False, params_dtype)))
            self.profile = config.profile
                
        self.weight_tensor = None
            
    def _create_weight_and_bias(self, weight_size, device, require_grad, params_dtype):
        """
        Creates a linear weight and bias tensor, using NF4 dtype if we're quantizing
        (indicated via quantize_base=True).
        """
        tmp = torch.empty(weight_size, device=device, requires_grad=require_grad, dtype=params_dtype)
        weight = tmp if not self.quantize else to_nf4(tmp)

        return weight

    def __delete_all_weight(self):
        self._parameters.pop("weight")
        self.weight = None
        del self.weight
        self.weight_tensor = None
        
    def forward(self, inp: torch.Tensor):
        if self.swap_weight:
            return super().forward(inp, weight_id=self.swap_weight)
        else:
            return super().forward(inp)

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False):
        return SwapWeightTemplate._load_state_dict(self, state_dict, strict, assign)
    
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        return SwapWeightTemplate._load_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs, super()._load_from_state_dict)
            
class SwapWeightLayerNorm(te.pytorch.LayerNorm, SwapWeightTemplate):
    def __init__(
        self,
        hidden_size: int,
        config: ModelParallelConfig,
        eps: float = 1e-5,
        sequence_parallel: bool = False,
        params_dtype: Optional[torch.dtype] = None,
        zero_centered_gamma: bool = False,
    ):
        super().__init__(
            hidden_size = hidden_size,
            eps = eps,
            sequence_parallel = sequence_parallel,
            params_dtype=params_dtype,
            zero_centered_gamma=zero_centered_gamma,
        )
        SwapWeightTemplate.init(self, hidden_size, config, params_dtype=params_dtype)
    
    def forward(self, inp: torch.Tensor):
        return SwapWeightTemplate._forward(self, inp, super().forward)

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False):
        return SwapWeightTemplate._load_state_dict(state_dict, strict, assign)
    
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        return SwapWeightTemplate._load_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs, super()._load_from_state_dict)
            
class SwapWeightRMSNorm(te.pytorch.RMSNorm, SwapWeightTemplate):
    def __init__(
        self,
        hidden_size: int,
        config: ModelParallelConfig,
        eps: float = 1e-5,
        sequence_parallel: bool = False,
        params_dtype: Optional[torch.dtype] = None,
        zero_centered_gamma: bool = False,
    ):
        super().__init__(
            hidden_size = hidden_size,
            eps = eps,
            sequence_parallel = sequence_parallel,
            params_dtype=params_dtype,
            zero_centered_gamma=zero_centered_gamma,
        )
        SwapWeightTemplate.init(self, hidden_size, config, params_dtype=params_dtype)
        
    def forward(self, inp: torch.Tensor):
        return SwapWeightTemplate._forward(self, inp, super().forward)
    
    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False):
        return SwapWeightTemplate._load_state_dict(state_dict, strict, assign)
    
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        return SwapWeightTemplate._load_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs, super()._load_from_state_dict)
    
            
class SwapWeightNorm:
    """
    A conditional wrapper to initialize an instance of Transformer-Engine's
    `LayerNorm` or `RMSNorm` based on input
    """

    # TODO should we ditch normalization config and just use spec to choose LayerNorm vs RMSNorm?
    def __new__(
        cls, config: TransformerConfig, hidden_size: int, eps: float = 1e-5,
    ):
        if config.normalization == "LayerNorm":
            instance = SwapWeightLayerNorm(
                hidden_size=hidden_size,
                config=config,
                eps=eps,
                sequence_parallel=config.sequence_parallel,
                zero_centered_gamma=config.layernorm_zero_centered_gamma,
                **_get_extra_te_kwargs(config),
            )
        elif config.normalization == "RMSNorm":
            assert hasattr(
                te.pytorch, "RMSNorm"
            ), "Transformer-Engine >= v0.11 required to use this feature"
            instance = SwapWeightRMSNorm(
                hidden_size=hidden_size,
                config=config,
                eps=eps,
                sequence_parallel=config.sequence_parallel,
                zero_centered_gamma=config.layernorm_zero_centered_gamma,
                **_get_extra_te_kwargs(config),
            )
        else:
            raise Exception('Only LayerNorm and RMSNorm are curently supported')

        return instance
    
class SwapWeightVocabParallelEmbedding(VocabParallelEmbedding, SwapWeightTemplate):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        init_method: Callable,
        reduce_scatter_embeddings: bool = False,
        config: ModelParallelConfig,
    ):
        super().__init__(num_embeddings=num_embeddings,
                         embedding_dim=embedding_dim,
                         init_method=init_method,
                         reduce_scatter_embeddings=reduce_scatter_embeddings,
                         config=config)
        SwapWeightTemplate.init(self, (self.num_embeddings_per_partition, self.embedding_dim), config, params_dtype=config.params_dtype)
        
    def forward(self, inp: torch.Tensor):
        return SwapWeightTemplate._forward(self, inp, super().forward)
    
    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False):
        return SwapWeightTemplate._load_state_dict(self, state_dict, strict, assign)
    
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        return SwapWeightTemplate._load_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs, super()._load_from_state_dict)
        
class SwapTPColumnParallelLinear(tensor_parallel.ColumnParallelLinear, SwapWeightTemplate):
    def __init__(
        self,
        input_size,
        output_size,
        *,
        config: ModelParallelConfig,
        init_method: Callable,
        bias=True,
        gather_output=False,
        stride=1,
        keep_master_weight_for_test=False,
        skip_bias_add=False,
        skip_weight_param_allocation: bool = False,
        embedding_activation_buffer: Optional[List[torch.Tensor]] = None,
        grad_output_buffer: Optional[List[torch.Tensor]] = None,
        is_expert: bool = False,
        tp_comm_buffer_name: str = None,  # Not used
        disable_grad_reduce: bool = False,
    ):
        super().__init__(
            input_size=input_size,
            output_size=output_size,
            config=config,
            init_method=init_method,
            bias=bias,
            gather_output=gather_output,
            stride=stride,
            keep_master_weight_for_test=keep_master_weight_for_test,
            skip_bias_add=skip_bias_add,
            skip_weight_param_allocation=skip_weight_param_allocation,
            embedding_activation_buffer=embedding_activation_buffer,
            grad_output_buffer=grad_output_buffer,
            is_expert=is_expert,
            tp_comm_buffer_name=tp_comm_buffer_name,
            disable_grad_reduce=disable_grad_reduce,
        )
        SwapWeightTemplate.init(self, (self.output_size_per_partition, self.input_size), config, params_dtype=config.params_dtype)
        
    def forward(self, inp: torch.Tensor, weight=None):
        
        if self.config.swap_weight:
            if weight is None:
                if self.swap_weight_id is None:
                    raise RuntimeError(
                        "weight was not supplied to ColumnParallelLinear forward pass "
                        "and skip_weight_param_allocation is True."
                    )

        return SwapWeightTemplate._forward(self, inp, super().forward, weight if not self.swap_weight else None, weight if self.swap_weight else None)
    
    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False):
        return SwapWeightTemplate._load_state_dict(self, state_dict, strict, assign)
    
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        return SwapWeightTemplate._load_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs, super()._load_from_state_dict)