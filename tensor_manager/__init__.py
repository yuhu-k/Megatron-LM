from torch import Tensor
from torchao.dtypes.nf4tensor import NF4Tensor
from typing import Union
from functools import partial
from .tensor_manager import TensorManager

__TENSOR_MANAGER = None

# def update_tensor_data(tensor:Union[Tensor, NF4Tensor]):
#     if __TENSOR_MANAGER == None:
#         assert "Error, Tensor_Manager is not yet initialized."
#     __TENSOR_MANAGER.update_tensor_data(tensor, tensor.tensor_id)

def register_tensor(tensor:Union[Tensor, NF4Tensor]):
    if __TENSOR_MANAGER == None:
        assert "Error, Tensor_Manager is not yet initialized."
    tensor.tensor_id = __TENSOR_MANAGER.register(tensor)
    tensor.get_computable_form = partial(__TENSOR_MANAGER.get_tensor, tensor_id=tensor.tensor_id)
    tensor.offload = partial(__TENSOR_MANAGER.offload_tensor, tensor_id=tensor.tensor_id)
    
# def get_computable_tensor(tensor:Union[Tensor, NF4Tensor]):
#     if __TENSOR_MANAGER == None:
#         assert "Error, Tensor_Manager is not yet initialized."
#     return __TENSOR_MANAGER.get_tensor(tensor.tensor_id)

# def offload_tensor(tensor:Union[Tensor, NF4Tensor]):
#     if __TENSOR_MANAGER == None:
#         assert "Error, Tensor_Manager is not yet initialized."
#     return __TENSOR_MANAGER.offload_tensor(tensor.tensor_id)

def finish_warmup():
    if __TENSOR_MANAGER == None:
        assert "Error, Tensor_Manager is not yet initialized."
    __TENSOR_MANAGER.finish_warmup()
    
def get_in_gpu_ratio():
    if __TENSOR_MANAGER == None:
        assert "Error, Tensor_Manager is not yet initialized."
    __TENSOR_MANAGER.get_in_gpu_tensor_ratio()

def chk_tensor_registered(tensor:Union[Tensor, NF4Tensor]):
    if __TENSOR_MANAGER == None:
        assert "Error, Tensor_Manager is not yet initialized."
    return hasattr(tensor, "tensor_id") and __TENSOR_MANAGER.chk_id_availibility(tensor.tensor_id)
    
def init_tensor_manager():
    global __TENSOR_MANAGER
    if __TENSOR_MANAGER == None:
        __TENSOR_MANAGER = TensorManager(1)
        