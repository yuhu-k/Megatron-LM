from megatron.core.transformer.custom_layers.transformer_engine import (
    condition_init_method,
    TEColumnParallelLinear,
    TERowParallelLinear
)
from megatron.core.transformer.custom_layers.swap_weight_layer import SwapWeightLinear
from megatron.core import ModelParallelConfig
from typing import Callable
from megatron.core.transformer.utils import make_sharded_tensors_for_checkpoint
from megatron.core.transformer.module import MegatronModule

class LoRAOriginalLinear(SwapWeightLinear):
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

class LoRALinear(MegatronModule):
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
        finetune_weight: bool = True,
    ):
        super().__init__(config)
        self.weight = LoRAOriginalLinear(
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
        self.weight.requires_grad_(False)
        self.rank = config.finetune_lora_rank
        self.alpha = config.finetune_lora_alpha
        self.finetune_weight = finetune_weight
        if config.tensor_model_parallel_size != None:
            if parallel_mode == "column":
                output_size = int(output_size / config.tensor_model_parallel_size)
            elif config.tensor_model_parallel_size != None:
                input_size = int(input_size / config.tensor_model_parallel_size)
            self.rank = int(self.rank / config.tensor_model_parallel_size)
        
        # if finetune_weight:
        #     # self.lora_a = SwapWeightLinear(
        #     #         input_size=input_size,
        #     #         output_size=self.rank,
        #     #         parallel_mode="column",
        #     #         config=config,
        #     #         init_method=config.init_method,
        #     #         bias=bias,
        #     #         skip_bias_add=skip_bias_add,
        #     #         skip_weight_param_allocation=skip_weight_param_allocation,
        #     #         tp_comm_buffer_name=tp_comm_buffer_name,
        #     #         swap_weight=False
        #     #     )
        #     # self.lora_b = SwapWeightLinear(
        #     #     input_size=self.rank,
        #     #     output_size=output_size,
        #     #     parallel_mode="row",
        #     #     config=config,
        #     #     init_method=init_method,
        #     #     bias=bias,
        #     #     skip_bias_add=skip_bias_add,
        #     #     skip_weight_param_allocation=False,
        #     #     tp_comm_buffer_name=tp_comm_buffer_name,
        #     #     swap_weight=False
        #     # )
        #     self.lora_a = TEColumnParallelLinear(
        #         input_size=input_size,
        #         output_size=self.rank,
        #         config=config,
        #         init_method=config.init_method,
        #         bias=bias,
        #         gather_output=False,
        #         skip_bias_add=skip_bias_add,
        #         is_expert=False,
        #         tp_comm_buffer_name=tp_comm_buffer_name
        #     )
        #     self.lora_b = TERowParallelLinear(
        #         input_size=self.rank,
        #         output_size=output_size,
        #         config=config,
        #         init_method=init_method,
        #         bias=bias,
        #         input_is_parallel=True,
        #         skip_bias_add=skip_bias_add,
        #         is_expert=False,
        #         tp_comm_buffer_name=tp_comm_buffer_name
        #     )
            
        
    
    def forward(self, x):
        out = self.weight(x)[0]
        # if self.finetune_weight:
        #     lora_out = (self.alpha / self.rank) * self.lora_b(self.lora_a(x)[0])[0]
        #     out = out + lora_out
        return out, None

class LoRAColumnParallelLinear(LoRALinear):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        config: ModelParallelConfig,
        init_method: Callable,
        gather_output: bool,
        bias: bool,
        skip_bias_add: bool,
        is_expert: bool,
        skip_weight_param_allocation: bool = False,
        tp_comm_buffer_name: str = None,
        finetune_weight: bool = True,
    ):
        if gather_output:
            raise ValueError('Transformer Engine linear layers do not support gather_output = True')

        if is_expert:
            raise ValueError('Transformer Engine linear layers do not yet support MoE')
        
        super().__init__(
            input_size=input_size,
            output_size=output_size,
            parallel_mode="column",
            config=config,
            init_method=condition_init_method(config, init_method),
            bias=bias,
            skip_bias_add=skip_bias_add,
            skip_weight_param_allocation=skip_weight_param_allocation,
            tp_comm_buffer_name=tp_comm_buffer_name,
            finetune_weight=finetune_weight,
        )

    def sharded_state_dict(self, prefix='', sharded_offsets=(), metadata=None):
        """ Sharding along axis 0, bias sharded """
        state_dict = self.state_dict(prefix='', keep_vars=True)
        return make_sharded_tensors_for_checkpoint(
            state_dict, prefix, {'weight': 0, 'bias': 0}, sharded_offsets
        )
            
class LoRARowParallelLinear(LoRALinear):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        config: ModelParallelConfig,
        init_method: Callable,
        bias: bool,
        input_is_parallel: bool,
        skip_bias_add: bool,
        is_expert: bool,
        tp_comm_buffer_name: str = None,
        finetune_weight: bool = True
    ):
        
        if not input_is_parallel:
            raise ValueError(
                "Transformer Engine linear layers do not support input_is_parallel = False"
            )

        if is_expert:
            raise ValueError('Transformer Engine linear layers do not yet support MoE')
        
        super().__init__(
            input_size=input_size,
            output_size=output_size,
            parallel_mode="row",
            config=config,
            init_method=condition_init_method(config, init_method),
            bias=bias,
            skip_bias_add=skip_bias_add,
            skip_weight_param_allocation=False,  # We don't currently use this for row parallel layers
            tp_comm_buffer_name=tp_comm_buffer_name,
            finetune_weight=finetune_weight,
        )
    
    def sharded_state_dict(self, prefix='', sharded_offsets=(), metadata=None):
        """ Sharding along axis 1, bias not sharded """
        state_dict = self.state_dict(prefix='', keep_vars=True)
        return make_sharded_tensors_for_checkpoint(
            state_dict, prefix, {'weight': 1}, sharded_offsets
        )

