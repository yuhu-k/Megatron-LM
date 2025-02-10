from megatron.core import mpu
from .split_model import get_layer_num as gln, get_global_layer_offset as gglo, get_all_stage_layer_num as gasln, get_rank_pp_dp_rank as grppdpr

def get_all_stage_layer_num(spec_list_path):
    return gasln(spec_list_path)

def get_layer_num(spec_list_path, pp_rank=None):
    return gln(spec_list_path, mpu.get_pipeline_model_parallel_rank() if pp_rank is None else pp_rank)

def get_global_layer_offset(spec_list_path):
    return gglo(spec_list_path, mpu.get_pipeline_model_parallel_rank(), mpu.get_pipeline_model_parallel_world_size())

def get_rank_pp_dp_rank(spec_list_path, global_rank, tp_degree):
    return grppdpr(spec_list_path, global_rank, tp_degree)