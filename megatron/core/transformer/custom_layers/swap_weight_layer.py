from megatron.core.transformer.custom_layers.transformer_engine import TELinear, _get_extra_te_kwargs
from megatron.core import ModelParallelConfig
from typing import Callable, Optional
import torch
import transformer_engine as te
from megatron.core.transformer.transformer_config import TransformerConfig


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
        tp_comm_buffer_name: str = None
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
        if config.swap_weight:
            self.register_forward_pre_hook(self.swap_weight_to_device)
            self.register_forward_hook(self.swap_weight_to_cpu)
            self.register_full_backward_pre_hook(self.swap_weight_to_device)
            self.register_full_backward_hook(self.swap_weight_to_cpu)
    
    def swap_weight_to_device(self, *args, **kwargs):
        if self.config.profile:
            torch.cuda.nvtx.range_push("swap weight to gpu")
        device = torch.cuda.current_device()
        if self.weight_tensor.device != device:
            self.weight_tensor.to(device)
        if self.config.profile:
            torch.cuda.nvtx.range_pop()
    
    def swap_weight_to_cpu(self, *args, **kwargs):
        if self.config.profile:
            torch.cuda.nvtx.range_push("swap weight to cpu")
        if str(self.weight_tensor.device) != "cpu":
            self.weight_tensor.to("cpu")
        if self.config.profile:
            torch.cuda.nvtx.range_pop()
            
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
        self.config = config
        if config.swap_weight:
            self.register_forward_pre_hook(self.swap_weight_to_device)
            self.register_forward_hook(self.swap_weight_to_cpu)
            self.register_full_backward_pre_hook(self.swap_weight_to_device)
            self.register_full_backward_hook(self.swap_weight_to_cpu)
    
    def swap_weight_to_device(self, *args, **kwargs):
        if self.config.profile:
            torch.cuda.nvtx.range_push("swap weight to gpu")
        device = torch.cuda.current_device()
        if self.weight.device != device:
            self.weight.to(device)
        if self.config.profile:
            torch.cuda.nvtx.range_pop()
    
    def swap_weight_to_cpu(self, *args, **kwargs):
        if self.config.profile:
            torch.cuda.nvtx.range_push("swap weight to cpu")
        if str(self.weight.device) != "cpu":
            self.weight.to("cpu")
        if self.config.profile:
            torch.cuda.nvtx.range_pop()
            
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
        self.config = config
        if config.swap_weight:
            self.register_forward_pre_hook(self.swap_weight_to_device)
            self.register_forward_hook(self.swap_weight_to_cpu)
            self.register_full_backward_pre_hook(self.swap_weight_to_device)
            self.register_full_backward_hook(self.swap_weight_to_cpu)
    
    def swap_weight_to_device(self, *args, **kwargs):
        if self.config.profile:
            torch.cuda.nvtx.range_push("swap weight to gpu")
        device = torch.cuda.current_device()
        if self.weight.device != device:
            self.weight.to(device)
        if self.config.profile:
            torch.cuda.nvtx.range_pop()
    
    def swap_weight_to_cpu(self, *args, **kwargs):
        if self.config.profile:
            torch.cuda.nvtx.range_push("swap weight to cpu")
        if str(self.weight.device) != "cpu":
            self.weight.to("cpu")
        if self.config.profile:
            torch.cuda.nvtx.range_pop()
            
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