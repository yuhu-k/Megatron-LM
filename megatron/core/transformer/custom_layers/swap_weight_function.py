"""Linear API"""
from typing import Union, Optional, Callable, Tuple, Dict, Any

import torch
import torch.distributed
from megatron.core import ModelParallelConfig
from megatron.core.transformer.custom_layers.transformer_engine import _get_extra_te_kwargs, _te_version, condition_init_method
from pkg_resources import packaging
from megatron.core.parallel_state import get_tensor_model_parallel_group

from megatron.core.tensor_parallel import get_cuda_rng_tracker


import transformer_engine_extensions as tex

from transformer_engine.pytorch.module.base import (
    get_workspace,
    _prepare_backward,
    get_ub,
    TransformerEngineBaseModule,
    _2X_ACC_FPROP,
    _2X_ACC_DGRAD,
    _2X_ACC_WGRAD,
)
from transformer_engine.pytorch.fp8 import get_fp8_te_dtype
from transformer_engine.pytorch.utils import (
    cast_if_needed,
    assert_dim_for_fp8_exec,
)
from transformer_engine.pytorch.distributed import (
    get_distributed_world_size,
    allreduce,
    reduce_scatter_along_first_dim,
    gather_along_first_dim,
    gather_along_last_dim,
)
from transformer_engine.pytorch.cpp_extensions import (
    fp8_gemm,
    gemm,
    fp8_cast_transpose_fused,
    cast_to_fp8,
)
from transformer_engine.pytorch.constants import dist_group_type
from megatron.training import get_args
from transformer_engine.pytorch.module.linear import Linear
from swap_manager import get_weight_swapper
from large_model_gpu import get_pack_hook
from tensor_manager import chk_tensor_registered


class _Linear(torch.autograd.Function):
    """Linear semi-top level module
    Calls custom cuda extensions.
    """

    @staticmethod
    def forward(
        ctx,
        weight: torch.Tensor,
        weight_fp8: Union[torch.Tensor, None],
        weight_t_fp8: Union[torch.Tensor, None],
        inp: torch.Tensor,
        bias: torch.Tensor,
        use_bias: bool,
        is_first_microbatch: Union[bool, None],
        fp8: bool,
        fp8_calibration: bool,
        fp8_meta: Dict[str, Any],
        fuse_wgrad_accumulation: bool,
        tp_group: Union[dist_group_type, None],
        tp_size: int,
        sequence_parallel: bool,
        tensor_parallel: bool,
        activation_dtype: torch.dtype,
        parallel_mode: Union[str, None],
        is_grad_enabled: bool,
        ub_split_rs: bool,
        ub_split_ag: bool,
    ) -> torch.Tensor:
        if chk_tensor_registered(weight):
            weight_for_save = weight
            weight = weight.get_computable_form()
            torch.cuda.current_stream().synchronize()
            freeze_weight = True
        else:
            freeze_weight = False


        
        # Make sure input dimensions are compatible
        in_features = weight.shape[-1]
        assert inp.shape[-1] == in_features, "GEMM not possible"
        inputmat = inp.view((-1, in_features))
        if fp8:
            assert_dim_for_fp8_exec(inputmat)
            assert_dim_for_fp8_exec(weight)

        update_fp8_weights = is_first_microbatch is None or is_first_microbatch

        if ub_split_rs:
            tp_world_size = get_distributed_world_size(tp_group)
            if tp_world_size == 1:
                ub_split_rs = False
        # Cast for native AMP
        inputmat = cast_if_needed(inputmat, activation_dtype)
        inputmat_no_fp8 = inputmat

        if fp8:
            fp8_dtype_forward = get_fp8_te_dtype(fp8_meta["recipe"], fprop_tensor=True)

            if not fp8_meta["recipe"].override_linear_precision.wgrad:
                if is_grad_enabled:
                    inputmat, inputmat_t = fp8_cast_transpose_fused(
                        inputmat,
                        fp8_meta["scaling_fwd"],
                        tex.FP8FwdTensors.GEMM1_INPUT,
                        fp8_dtype_forward,
                    )
                else:
                    inputmat = cast_to_fp8(
                        inputmat,
                        fp8_meta["scaling_fwd"],
                        tex.FP8FwdTensors.GEMM1_INPUT,
                        fp8_dtype_forward,
                    )
            else:
                inputmat, inputmat_t = cast_to_fp8(
                    inputmat,
                    fp8_meta["scaling_fwd"],
                    tex.FP8FwdTensors.GEMM1_INPUT,
                    fp8_dtype_forward,
                ), None

        # Column Parallel Linear
        if parallel_mode == "column" and sequence_parallel:
            inputmat_total, _ = gather_along_first_dim(inputmat, tp_group)
        else:
            inputmat_total = inputmat

        if fp8:
            bias_dtype = (
                torch.bfloat16
                if activation_dtype == torch.float32
                else activation_dtype
            )
            bias = cast_if_needed(bias, bias_dtype) if use_bias else bias

            if update_fp8_weights:
                if is_grad_enabled:
                    fp8_cast_transpose_fused(
                        weight,
                        fp8_meta["scaling_fwd"],
                        tex.FP8FwdTensors.GEMM1_WEIGHT,
                        fp8_dtype_forward,
                        cast_out=weight_fp8,
                        transpose_out=weight_t_fp8,
                    )
                else:
                    weight_t_fp8 = None
                    weight_fp8 = cast_to_fp8(
                        weight,
                        fp8_meta["scaling_fwd"],
                        tex.FP8FwdTensors.GEMM1_WEIGHT,
                        fp8_dtype_forward,
                    )

            if ub_split_rs:
                ub_obj_projout = get_ub("proj_fprop")
                out = ub_obj_projout.get_ubuf_output(1)
                dim_size = list(inputmat_total.size())
                dim_size[0] = dim_size[0] // tp_world_size
                dim_size[1] = weight.size(0)
                rs_out = torch.empty(dim_size, dtype=activation_dtype, device=inputmat_total.device)
            else:
                dim_size = list(inputmat_total.size())
                dim_size[1] = weight.size(0)
                out = torch.empty(dim_size, dtype=activation_dtype, device=inputmat_total.device)

            _ = fp8_gemm(
                weight_fp8,
                fp8_meta["scaling_fwd"].scale_inv,
                tex.FP8FwdTensors.GEMM1_WEIGHT,
                fp8_dtype_forward,
                inputmat_total,
                fp8_meta["scaling_fwd"].scale_inv,
                tex.FP8FwdTensors.GEMM1_INPUT,
                fp8_dtype_forward,
                activation_dtype,
                get_workspace(),
                bias=bias,
                use_bias=use_bias,
                use_split_accumulator=_2X_ACC_FPROP,
                out=out,
                ub_algo=tex.UbufOverlapAlgo.SPLIT_PIPELINED_RS if ub_split_rs else None,
                ub=ub_obj_projout if ub_split_rs else None,
                extra_output_tensor=rs_out if ub_split_rs else None,
            )
        else:
            # Cast for native AMP
            weight = cast_if_needed(weight, activation_dtype)
            bias = cast_if_needed(bias, activation_dtype) if use_bias else bias

            if fp8_calibration:
                # amax of input
                fp8_meta["scaling_fwd"].amax_history[0][tex.FP8FwdTensors.GEMM1_INPUT] = \
                    torch.amax(inputmat_total).float()
                # amax of weight
                fp8_meta["scaling_fwd"].amax_history[0][tex.FP8FwdTensors.GEMM1_WEIGHT] = \
                    torch.amax(weight).float()

            if ub_split_rs:
                ub_obj_projout = get_ub("proj_fprop")
                out = ub_obj_projout.get_ubuf_output(1)
                dim_size = list(inputmat_total.size())
                dim_size[0] = dim_size[0] // tp_world_size
                dim_size[1] = weight.size(0)
                rs_out = torch.empty(dim_size, dtype=activation_dtype, device=inputmat_total.device)
            else:
                dim_size = list(inputmat_total.size())
                dim_size[1] = weight.size(0)
                out = torch.empty(dim_size, dtype=activation_dtype, device=inputmat_total.device)

            _, _, _ = gemm(
                weight,
                inputmat_total,
                activation_dtype,
                get_workspace(),
                bias=bias,
                use_bias=use_bias,
                out=out,
                ub_algo=tex.UbufOverlapAlgo.SPLIT_PIPELINED_RS if ub_split_rs else None,
                ub=ub_obj_projout if ub_split_rs else None,
                extra_output_tensor=rs_out if ub_split_rs else None,
            )
        #args = get_args()
        # if args.lms:
        #     weight = weight.cpu()
            
        if is_grad_enabled:
            fp8_wgrad = fp8 and not fp8_meta["recipe"].override_linear_precision.wgrad
            hook = get_pack_hook()
            ctx.save_for_backward(
                *hook.my_pack_hook(
                inputmat_no_fp8 if weight.requires_grad and not fp8_wgrad else None,
                inputmat_t if weight.requires_grad and fp8_wgrad else None,
                weight_for_save if freeze_weight else weight,
                weight_t_fp8 if fp8 else None,
                fp8_meta["scaling_fwd"].scale_inv.clone() if fp8 else None,)
            )
            ctx.activation_dtype = activation_dtype
            ctx.fp8 = fp8
            ctx.fp8_meta = fp8_meta
            ctx.fuse_wgrad_accumulation = fuse_wgrad_accumulation
            ctx.is_first_microbatch = is_first_microbatch
            ctx.use_bias = use_bias
            ctx.sequence_parallel = sequence_parallel
            ctx.tensor_parallel = tensor_parallel
            ctx.inp_shape = inp.shape
            ctx.parallel_mode = parallel_mode
            ctx.tp_group = tp_group
            ctx.ub_split_ag = ub_split_ag
            ctx.tp_size = tp_size
            ctx.requires_dgrad = inp.requires_grad

        # Row Parallel Linear
        if ub_split_rs:
            out = rs_out
        elif parallel_mode == "row" and sequence_parallel:
            out, _ = reduce_scatter_along_first_dim(out, tp_group)
        elif parallel_mode == "row" and tensor_parallel:
            out, _ = allreduce(out, tp_group)
        # [*, in_features] -> [*, out_features] except first dimension changes for SP
        return out.view(-1, *inp.shape[1:-1], out.shape[-1])


    @staticmethod
    def backward(
        ctx, grad_output: torch.Tensor
    ) -> Tuple[Union[torch.Tensor, None], ...]:
        with _prepare_backward(
            ctx.fp8, ctx.fp8_meta, ctx.tp_group, ctx.tp_size, name="_Linear"
        ):
            hook = get_pack_hook()
            (
                inputmat,
                inputmat_t,
                weight,
                weight_t_fp8,
                fwd_scale_inverses,
            ) = hook.my_unpack_hook(*ctx.saved_tensors)

            if chk_tensor_registered(weight):
                weight = weight.get_computable_form()
                torch.cuda.current_stream().synchronize()

            
            # if args.lms:
            #     weight = weight.to(torch.cuda.current_device())
            
            if ctx.ub_split_ag:
                tp_world_size = get_distributed_world_size(ctx.tp_group)
                if tp_world_size == 1:
                    ctx.ub_split_ag = False
            if ctx.ub_split_ag:
                dim_size = list(grad_output.size())
                dim_size[0] = dim_size[0] * tp_world_size
                ctx.ub_obj_gradout = get_ub("proj_dgrad")
            (
                grad_output,
                grad_output_c,
                grad_output_t,
                grad_bias,
            ) = TransformerEngineBaseModule.grad_output_preprocess(
                ctx, grad_output, ctx.parallel_mode == "row"
            )
            handle = None
            # Column Parallel Linear
            # Overlap input AG with dgrad
            if ctx.parallel_mode == "column" and ctx.sequence_parallel:
                if ctx.fp8 and not ctx.fp8_meta["recipe"].override_linear_precision.wgrad:
                    inputmat_t_total, handle = gather_along_last_dim(
                        inputmat_t, ctx.tp_group, async_op=ctx.requires_dgrad
                    )
                else:
                    inputmat_total, handle = gather_along_first_dim(
                        inputmat, ctx.tp_group, async_op=ctx.requires_dgrad
                    )
            else:
                inputmat_t_total = inputmat_t
                inputmat_total = inputmat

            if ctx.is_first_microbatch is not None:
                accumulate_wgrad_into_param_main_grad = (
                    ctx.fuse_wgrad_accumulation and not ctx.is_first_microbatch
                )
            else:
                accumulate_wgrad_into_param_main_grad = ctx.fuse_wgrad_accumulation

            if ctx.fp8:
                fp8_dtype_forward = get_fp8_te_dtype(
                    ctx.fp8_meta["recipe"], fprop_tensor=True
                )
                fp8_dtype_backward = get_fp8_te_dtype(
                    ctx.fp8_meta["recipe"], fprop_tensor=False
                )

            if ctx.requires_dgrad:
                if ctx.fp8:
                    dgrad = fp8_gemm(
                        weight_t_fp8,
                        fwd_scale_inverses,
                        tex.FP8FwdTensors.GEMM1_WEIGHT,
                        fp8_dtype_forward,
                        grad_output_c,
                        ctx.fp8_meta["scaling_bwd"].scale_inv,
                        tex.FP8BwdTensors.GRAD_OUTPUT1,
                        fp8_dtype_backward,
                        ctx.activation_dtype,
                        get_workspace(),
                        use_split_accumulator=_2X_ACC_DGRAD,
                        ub_algo=tex.UbufOverlapAlgo.SPLIT_PIPELINED_AG if ctx.ub_split_ag else None,
                        ub=ctx.ub_obj_gradout if ctx.ub_split_ag else None,
                    )
                else:
                    dgrad, _, _ = gemm(
                        weight,
                        grad_output,
                        ctx.activation_dtype,
                        get_workspace(),
                        layout="NN",
                        grad=True,
                        ub_algo=tex.UbufOverlapAlgo.SPLIT_PIPELINED_AG if ctx.ub_split_ag else None,
                        ub=ctx.ub_obj_gradout if ctx.ub_split_ag else None,
                    )

                # Overlap dgrad-RS/AR with wgrad
                if ctx.parallel_mode == "column" and ctx.sequence_parallel:
                    handle.wait()
                    dgrad, handle = reduce_scatter_along_first_dim(
                        dgrad, ctx.tp_group, async_op=True
                    )
                elif ctx.parallel_mode == "column" and ctx.tensor_parallel:
                    dgrad, handle = allreduce(dgrad, ctx.tp_group, async_op=True)

            if weight.requires_grad:
                if ctx.fp8:
                    # WGRAD
                    if not ctx.fp8_meta["recipe"].override_linear_precision.wgrad:
                        if ctx.ub_split_ag:
                            grad_output_t = tex.fp8_transpose(grad_output_c, fp8_dtype_backward)
                        wgrad = fp8_gemm(
                            inputmat_t_total,
                            fwd_scale_inverses,
                            tex.FP8FwdTensors.GEMM1_INPUT,
                            fp8_dtype_forward,
                            grad_output_t,
                            ctx.fp8_meta["scaling_bwd"].scale_inv,
                            tex.FP8BwdTensors.GRAD_OUTPUT1,
                            fp8_dtype_backward,
                            ctx.activation_dtype,
                            get_workspace(),
                            accumulate=accumulate_wgrad_into_param_main_grad,
                            out=weight.main_grad if ctx.fuse_wgrad_accumulation else None,
                            use_split_accumulator=_2X_ACC_WGRAD,
                        )
                    else:
                        wgrad, _, _ = gemm(
                            inputmat_total,
                            grad_output,
                            ctx.activation_dtype,
                            get_workspace(),
                            layout="NT",
                            grad=True,
                            accumulate=accumulate_wgrad_into_param_main_grad,
                            out=weight.main_grad if ctx.fuse_wgrad_accumulation else None,
                        )
                else:
                    # WGRAD
                    wgrad, grad_bias, _ = gemm(
                        inputmat_total,
                        grad_output,
                        ctx.activation_dtype,
                        get_workspace(),
                        layout="NT",
                        grad=True,
                        use_bias=ctx.use_bias,
                        accumulate=accumulate_wgrad_into_param_main_grad,
                        out=weight.main_grad if ctx.fuse_wgrad_accumulation else None,
                    )

            require_grad = weight.requires_grad

            
            # Column Parallel Linear
            if ctx.parallel_mode == "column" and ctx.tensor_parallel and handle is not None:
                handle.wait()

            if not ctx.use_bias:
                grad_bias = None
        # if ctx.offload_weight:
        #     weight_tmp.offload()

        return (
            wgrad if require_grad else None,
            None,
            None,
            dgrad.view(ctx.inp_shape) if ctx.requires_dgrad else None,
            grad_bias,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )

class SwapLinear(Linear):
    def forward(self,
        inp: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
        is_first_microbatch: Optional[bool] = None,
        weight_id: Optional[int] = None):
        # if weight is not None or bias is not None:
        #     raise RuntimeError(
        #         "Arguments `weight` and `bias` are deprecated and "
        #         "will be fully removed in future releases."
        #     )

        with self.prepare_forward(inp, is_first_microbatch) as inp:
            bias_tensor = (
                self.bias if self.parameters_split is None
                else self.bias_tensor if not torch.is_grad_enabled()
                else self.noop_cat("bias_tensor", self.bias_names)
            )
            if weight_id == None:
                weight_tensor = ( weight if weight != None 
                    else self.weight if self.parameters_split is None
                    else self.weight_tensor if not torch.is_grad_enabled()
                    else self.noop_cat("weight_tensor", self.weight_names)
                )
            else:
                weight_tensor = None

            # Fetch the fp8 weights placeholders (for linear/gemm)
            weight1_fp8, weight1_t_fp8 = self.get_fp8_weights_scratchpad(
                is_first_microbatch
            )

            if torch.is_grad_enabled():
                linear_fn = _Linear.apply
                args = []
            else:
                linear_fn = _Linear.forward
                args = [None]
            args += (
                weight_tensor,
                weight1_fp8,
                weight1_t_fp8,
                inp,
                bias_tensor,
                self.apply_bias and not self.gemm_bias_unfused_add,
                is_first_microbatch,
                self.fp8,
                self.fp8_calibration,
                self.fp8_meta,
                self.fuse_wgrad_accumulation,
                self.tp_group,
                self.tp_size,
                self.sequence_parallel,
                self.tp_size > 1,
                self.activation_dtype,
                self.parallel_mode,
                torch.is_grad_enabled(),
                self.ub_split_rs,
                self.ub_split_ag,
            )
            out = linear_fn(*args)
        
        if self.gemm_bias_unfused_add:
            out = out + cast_if_needed(bias_tensor, self.activation_dtype)

        if self.return_bias:
            return out, cast_if_needed(bias_tensor, self.activation_dtype)
        return out
    
class SwapTELinear(SwapLinear):
    """
    Wrapper for the Transformer-Engine's `Linear` layer.

    Note that if Megatron's parallel_state has not been initialized
    yet, the tp_group passed to TE will be None and must be set later
    via set_tensor_parallel_group().
    """

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
        self.config = config

        # TE returns a zero length Tensor when bias=False and
        # return_bias=True, but we prefer None.  So in that case we
        # tell TE to not return the bias, and return None
        # ourselves. This way our forward always returns two values
        # and we don't have to deal with the zero length Tensor.
        self.te_return_bias = skip_bias_add and bias
        self.is_first_microbatch = True
        self.disable_parameter_transpose_cache = self.config.disable_parameter_transpose_cache
        if skip_weight_param_allocation:
            raise ValueError(
                'Transformer Engine linear layers do not support skip_weight_param_allocation'
            )

        extra_kwargs = _get_extra_te_kwargs(config)

        if _te_version >= packaging.version.Version("0.8.0"):
            if self.config.tp_comm_overlap:
                if _te_version > packaging.version.Version("1.5.0"):
                    # Use old overlap flags if they were supplied instead
                    extra_kwargs["ub_overlap_ag"] = (
                        self.config.tp_comm_overlap_ag
                        if hasattr(self.config, "tp_comm_overlap_ag")
                        else self.config.tp_comm_split_ag or self.config.tp_comm_atomic_ag
                    )
                    extra_kwargs["ub_overlap_rs"] = (
                        self.config.tp_comm_overlap_rs
                        if hasattr(self.config, "tp_comm_overlap_rs")
                        else self.config.tp_comm_split_rs or self.config.tp_comm_atomic_rs
                    )
                else:
                    extra_kwargs["ub_split_ag"] = self.config.tp_comm_split_ag
                    extra_kwargs["ub_atomic_gemm_ag"] = self.config.tp_comm_atomic_ag
                    extra_kwargs["ub_split_rs"] = self.config.tp_comm_split_rs
                    extra_kwargs["ub_atomic_gemm_rs"] = self.config.tp_comm_atomic_rs
                if _te_version > packaging.version.Version("1.0.0"):
                    assert (
                        tp_comm_buffer_name is not None
                    ), "Buffer name should be set to configure communication overlap settings"
                    extra_kwargs["ub_name"] = tp_comm_buffer_name

        super().__init__(
            in_features=input_size,
            out_features=output_size,
            sequence_parallel=self.config.sequence_parallel,
            fuse_wgrad_accumulation=self.config.gradient_accumulation_fusion,
            tp_group=get_tensor_model_parallel_group(check_initialized=False),
            tp_size=self.config.tensor_model_parallel_size,
            get_rng_state_tracker=get_cuda_rng_tracker
            if get_cuda_rng_tracker().is_initialized()
            else None,
            init_method=condition_init_method(config, init_method),
            bias=bias,
            return_bias=self.te_return_bias,
            parallel_mode=parallel_mode,
            **extra_kwargs,
        )

    def forward(self, x, weight=None, weight_id=None):
        _is_first_microbatch = (
            None if self.disable_parameter_transpose_cache else self.is_first_microbatch
        )
        out = super().forward(x, weight=weight, is_first_microbatch=_is_first_microbatch, weight_id=weight_id)
        self.is_first_microbatch = False

        # TE only returns a tuple when return_bias is True, otherwise
        # it returns a single Tensor, we always want to return two
        # values regardless of the arguments.
        if self.te_return_bias:
            return out
        return out, None