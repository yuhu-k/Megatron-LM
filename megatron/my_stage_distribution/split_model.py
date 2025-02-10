import os.path as osp
from typing import List

def get_all_stage_layer_num(spec_list_path) -> List[int]:
    try:
        if osp.exists(spec_list_path):
            with open(spec_list_path, 'r') as f:
                nums = f.read().strip()
            if ',' in nums:
                nums = nums.split(',')
            elif ' ' in nums:
                nums = nums.split()
            else:
                nums = nums.split('\n')
            num_layers = [int(x.strip()) for x in nums]
        else:
            num_layers = [int(x.strip()) for x in str(spec_list_path).split(',')]
    except:
        raise ValueError(f"Error in reading the layer number from {spec_list_path}")
        
    return num_layers

def get_layer_num(spec_list_path, pp_rank):
    num_list = get_all_stage_layer_num(spec_list_path)
    return num_list[pp_rank]

def get_global_layer_offset(spec_list_path, pp_rank, pp_size):
    num_list = get_all_stage_layer_num(spec_list_path)
    return sum(num_list[:pp_rank])
    
def get_rank_pp_dp_rank(spec_list_path, global_rank, tp_degree):
    num_list = get_all_stage_layer_num(spec_list_path)
    layer_offset = 0
    for i, num in enumerate(num_list):
        if layer_offset + num * tp_degree > global_rank:
            return i, num
        layer_offset += num * tp_degree
    return None, None