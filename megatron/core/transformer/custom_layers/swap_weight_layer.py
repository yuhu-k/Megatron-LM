import torch.distributed
from megatron.core.transformer.custom_layers.transformer_engine import TELinear, _get_extra_te_kwargs
from megatron.core import ModelParallelConfig
from typing import Callable, Optional, List, OrderedDict

import torch
import transformer_engine as te
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.tensor_parallel.layers import VocabParallelEmbedding
from megatron.core import tensor_parallel
from torch import Tensor

def log_gpu_memory_usage():
    with open("/tmp2/yuhu/result.txt", "a") as f:
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        f.write(f"Allocated GPU memory: {allocated:.2f} GB\n")
        f.write(f"Reserved GPU memory: {reserved:.2f} GB\n")
    

def swap_weight_to_gpu(weight: Tensor, profile):
    if profile:
        torch.cuda.nvtx.range_push(f"swap weight to cuda")
    if "cuda" not in str(weight.device):
        weight = weight.to(torch.cuda.current_device()).detach()
    else:
        weight = weight.detach()
    if profile:
        torch.cuda.nvtx.range_pop()
    torch.cuda.empty_cache()
    if torch.cuda.is_available() and torch.distributed.get_rank() == 0:
        log_gpu_memory_usage()
    return weight

class SwapWeightLinear(TELinear):
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
        self._parameters = OrderedDict()
        self.weight = None
        del self.weight
        self.weight_tensor = None
        self.swap_weight = swap_weight
        self.config = config
        if swap_weight and config.swap_weight:
            device = "cpu"
            require_grad = False
        else:
            device = torch.cuda.current_device()
            require_grad = True
        self.register_parameter("weight", torch.nn.Parameter(torch.empty((self.out_features, self.in_features), device=device, requires_grad=require_grad, dtype=self.params_dtype)))
    
    def forward(self, inp: torch.Tensor):
        
        log_gpu_memory_usage()
        if self.swap_weight and self.config.swap_weight:
            weight = swap_weight_to_gpu(self.weight.data, self.config.profile)
            output = super().forward(inp, weight)
            del weight
        else:
            output = super().forward(inp)
        return output
            
class SwapWeightLayerNorm(te.pytorch.LayerNorm):
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
        self._parameters = OrderedDict()
        self.weight = None
        self.config = config
        del self.weight
        if config.swap_weight:
            device = "cpu"
        else:
            device = torch.cuda.current_device()
        self.weight = torch.nn.Parameter(torch.empty(hidden_size, device=device, requires_grad=False, dtype=params_dtype))
    
    def forward(self, inp: torch.Tensor):
        
        log_gpu_memory_usage()
        if self.config.swap_weight:
            weight = swap_weight_to_gpu(self.weight.data, self.config.profile)
            output = super().forward(inp, weight)
            del weight
        else:
            output = super().forward(inp)

        return output
            
class SwapWeightRMSNorm(te.pytorch.RMSNorm):
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
        self._parameters = OrderedDict()
        self.weight = None
        self.config = config
        del self.weight
        if config.swap_weight:
            device = "cpu"
        else:
            device = torch.cuda.current_device()
        self.weight = torch.nn.Parameter(torch.empty(hidden_size, device=device, requires_grad=False, dtype=params_dtype))
    
    def forward(self, inp: torch.Tensor):
        
        log_gpu_memory_usage()
        if self.config.swap_weight:
            weight = swap_weight_to_gpu(self.weight.data, self.config.profile)
            output = super().forward(inp, weight)
            del weight
        else:
            output = super().forward(inp)

        return output
    
            
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
    
class SwapWeightVocabParallelEmbedding(VocabParallelEmbedding):
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
        self._parameters = OrderedDict()
        self.weight = None
        self.config = config
        del self.weight
        if config.swap_weight:
            device = "cpu"
        else:
            device = torch.cuda.current_device()
        self.weight = torch.nn.Parameter(torch.empty(self.num_embeddings_per_partition, self.embedding_dim, device=device, requires_grad=False, dtype=config.params_dtype))
    
    def forward(self, inp: torch.Tensor):
        
        log_gpu_memory_usage()
        if self.config.swap_weight:
            weight = swap_weight_to_gpu(self.weight.data, self.config.profile)
            output = super().forward(inp, weight)
            del weight
        else:
            output = super().forward(inp)

        return output
        
class SwapTPColumnParallelLinear(tensor_parallel.ColumnParallelLinear):
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
        self._parameters.pop("weight")
        self.weight = None
        self.config = config
        del self.weight
        if config.swap_weight:
            device = "cpu"
        else:
            device = torch.cuda.current_device()
        self.weight = torch.nn.Parameter(torch.empty(self.output_size_per_partition, self.input_size, device=device, requires_grad=False, dtype=config.params_dtype))
    
    def forward(self, inp: torch.Tensor, weight=None):
        
        if weight is None:
            if self.weight is None:
                raise RuntimeError(
                    "weight was not supplied to ColumnParallelLinear forward pass "
                    "and skip_weight_param_allocation is True."
                )
            weight = self.weight

        
        log_gpu_memory_usage()
        if self.config.swap_weight:
            weight = swap_weight_to_gpu(weight, self.config.profile)

        output = super().forward(inp, weight)
        del weight

        return output