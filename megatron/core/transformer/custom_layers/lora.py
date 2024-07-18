from megatron.core.transformer.custom_layers.transformer_engine import (
    TEColumnParallelLinear,
    TERowParallelLinear
)
from megatron.core import ModelParallelConfig
from typing import Callable


class LoRAColumnParallelLinear(TEColumnParallelLinear):
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
        mlp_layer_num: int = None
    ):
        super().__init__(
            input_size=input_size,
            output_size=output_size,
            config=config,
            init_method=init_method,
            gather_output=gather_output,
            bias=bias,
            skip_bias_add=skip_bias_add,
            is_expert=is_expert,
            skip_weight_param_allocation=skip_weight_param_allocation,
            tp_comm_buffer_name=tp_comm_buffer_name
        )
        
        self.mlp_layer_num = mlp_layer_num
        self.config = config
        
        if config.finetune_mlp and mlp_layer_num == 1:
            self.rank = config.finetune_lora_rank
            self.alpha = config.finetune_lora_alpha
            if config.tensor_model_parallel_size != None:
                output_size = int(output_size / config.tensor_model_parallel_size)
                self.rank = int(self.rank / config.tensor_model_parallel_size)
            
            self.gate_lora_a = TEColumnParallelLinear(
                input_size=input_size,
                output_size=self.rank,
                config=config,
                init_method=config.init_method,
                gather_output=gather_output,
                bias=bias,
                skip_bias_add=skip_bias_add,
                is_expert=is_expert,
                skip_weight_param_allocation=skip_weight_param_allocation,
                tp_comm_buffer_name=tp_comm_buffer_name
            )
            self.gate_lora_b = TERowParallelLinear(
                input_size=self.rank,
                output_size=output_size,
                config=config,
                init_method=init_method,
                bias=bias,
                input_is_parallel=True,
                skip_bias_add=skip_bias_add,
                is_expert=is_expert,
                tp_comm_buffer_name=tp_comm_buffer_name,
            )
            self.up_lora_a = TEColumnParallelLinear(
                input_size=input_size,
                output_size=self.rank,
                config=config,
                init_method=config.init_method,
                gather_output=gather_output,
                bias=bias,
                skip_bias_add=skip_bias_add,
                is_expert=is_expert,
                skip_weight_param_allocation=skip_weight_param_allocation,
                tp_comm_buffer_name=tp_comm_buffer_name
            )
            self.up_lora_b = TERowParallelLinear(
                input_size=self.rank,
                output_size=output_size,
                config=config,
                init_method=init_method,
                bias=bias,
                input_is_parallel=True,
                skip_bias_add=skip_bias_add,
                is_expert=is_expert,
                tp_comm_buffer_name=tp_comm_buffer_name,
            )
        elif (self.mlp_layer_num == None or config.finetune_mlp) and config.finetune_method == "lora":
            self.rank = config.finetune_lora_rank
            self.alpha = config.finetune_lora_alpha
            if config.tensor_model_parallel_size != None:
                output_size = int(output_size / config.tensor_model_parallel_size)
                self.rank = int(self.rank / config.tensor_model_parallel_size)
            
            self.lora_a = TEColumnParallelLinear(
                input_size=input_size,
                output_size=self.rank,
                config=config,
                init_method=config.init_method,
                gather_output=gather_output,
                bias=bias,
                skip_bias_add=skip_bias_add,
                is_expert=is_expert,
                skip_weight_param_allocation=skip_weight_param_allocation,
                tp_comm_buffer_name=tp_comm_buffer_name
            )
            self.lora_b = TERowParallelLinear(
                input_size=self.rank,
                output_size=output_size,
                config=config,
                init_method=init_method,
                bias=bias,
                input_is_parallel=True,
                skip_bias_add=skip_bias_add,
                is_expert=is_expert,
                tp_comm_buffer_name=tp_comm_buffer_name,
            )
        else:
            self.lora_a = None
            self.lora_b = None
        
    def forward(self, x):
        out = super().forward(x)[0]
        if self.config.finetune_mlp and self.mlp_layer_num == 1:
            import torch
            out = torch.chunk(out, 2, dim=-1)
            gate = (self.alpha / self.rank) * self.gate_lora_b(self.gate_lora_a(out[0]))[0]
            up = (self.alpha / self.rank) * self.up_lora_b(self.up_lora_a(out[1]))[1]
            out = torch.cat([gate,up], dim=-1)
        elif (self.mlp_layer_num == None or self.config.finetune_mlp) and self.config.finetune_method == "lora":
            lora_out = self.lora_a(x)[0]
            lora_out = (self.alpha / self.rank) * self.lora_b(lora_out)[0]
            out = out + lora_out

        return out,None
            
class LoRARowParallelLinear(TERowParallelLinear):
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
        is_mlp: bool = False
    ):
        super().__init__(
            input_size=input_size,
            output_size=output_size,
            config=config,
            init_method=init_method,
            bias=bias,
            input_is_parallel=input_is_parallel,
            skip_bias_add=skip_bias_add,
            is_expert=is_expert,
            tp_comm_buffer_name=tp_comm_buffer_name,
        )
        
        self.is_mlp = is_mlp
        self.config = config
        
        
        if (not is_mlp or config.finetune_mlp) and config.finetune_method == "lora":
            self.rank = config.finetune_lora_rank
            self.alpha = config.finetune_lora_alpha
            if config.tensor_model_parallel_size != None:
                input_size = int(input_size / config.tensor_model_parallel_size)
                self.rank = int(self.rank / config.tensor_model_parallel_size)
            
            self.lora_a = TEColumnParallelLinear(
                input_size=input_size,
                output_size=self.rank,
                config=config,
                init_method=config.init_method,
                gather_output=False,
                bias=bias,
                skip_bias_add=skip_bias_add,
                is_expert=is_expert,
                skip_weight_param_allocation=False,
                tp_comm_buffer_name=tp_comm_buffer_name
            )
            self.lora_b = TERowParallelLinear(
                input_size=self.rank,
                output_size=output_size,
                config=config,
                init_method=init_method,
                bias=bias,
                input_is_parallel=input_is_parallel,
                skip_bias_add=skip_bias_add,
                is_expert=is_expert,
                tp_comm_buffer_name=tp_comm_buffer_name,
            )
        else:
            self.lora_a = None
            self.lora_b = None
        
        
    def forward(self, x):
        out = super().forward(x)[0]
        if (not self.is_mlp or self.config.finetune_mlp) and self.config.finetune_method == "lora":
            lora_out = self.lora_a(x)[0]
            lora_out = (self.alpha / self.rank) * self.lora_b(lora_out)[0]
            out = out + lora_out

        return out,None