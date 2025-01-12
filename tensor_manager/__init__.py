from torch import Tensor
from torchao.dtypes.nf4tensor import NF4Tensor
from typing import Union
from functools import partial
from .tensor_manager import TensorManager
from torch.nn.parameter import Parameter

__TENSOR_MANAGER = None

class PackTensor:
    def __init__(self, tensor_manager, tensor_id):
        if tensor_manager == None:
            raise ValueError("Error, Tensor_Manager is not yet initialized.")
        # self.get_computable_form = partial(tensor_manager.get_tensor, tensor_id=tensor_id)
        self.delete = partial(tensor_manager.delete_tensor, tensor_id=tensor_id)
        self.tensor_id = tensor_id
        self.mgr = tensor_manager
        
    def get_computable_form(self):
        # print("get", self.tensor_id)
        return self.mgr.get_tensor(self.tensor_id)
        

class PackTensorList:
    def __init__(self, tensor_manager, *tensor_ids):
        if tensor_manager == None:
            raise ValueError("Error, Tensor_Manager is not yet initialized.")
        self.packed_tensors = []
        for tensor_id in tensor_ids:
            if type(tensor_id) == PackTensorList:
                self.packed_tensors.append(tensor_id)
            elif type(tensor_id) == int:
                self.packed_tensors.append(PackTensor(tensor_manager, tensor_id))
            else:
                self.packed_tensors.append(tensor_id)
                # raise TypeError("Tensor ID should be an integer. But got type: {}".format(type(tensor_id)))
    
    def get_computable_form(self):
        # print("get", self.packed_tensors)
        result = tuple([packed_tensor.get_computable_form() if type(packed_tensor) == PackTensor or type(packed_tensor) == PackTensorList else packed_tensor for packed_tensor in self.packed_tensors])
        # print("get", result)
        # self.delete()
        if len(result) == 1:
            return result[0]
        return result

    def delete(self):
        for packed_tensor in self.packed_tensors:
            if type(packed_tensor) == PackTensorList or type(packed_tensor) == PackTensor:
                packed_tensor.delete()
                
    # def get_ids(self):
    #     return [packed_tensor.tensor_id if type(packed_tensor) == PackTensor else packed_tensor for packed_tensor in self.packed_tensors]

def register_tensor(tensor:Union[Tensor, NF4Tensor]):
    if __TENSOR_MANAGER == None:
        assert "Error, Tensor_Manager is not yet initialized."
    tensor.tensor_id = __TENSOR_MANAGER.register(tensor)
    tensor.get_computable_form = partial(__TENSOR_MANAGER.get_tensor, tensor_id=tensor.tensor_id)
    tensor.offload = partial(__TENSOR_MANAGER.offload_tensor, tensor_id=tensor.tensor_id)

def finish_warmup():
    if __TENSOR_MANAGER == None:
        assert "Error, Tensor_Manager is not yet initialized."
    __TENSOR_MANAGER.finish_warmup()
    
def get_in_gpu_ratio():
    if __TENSOR_MANAGER == None:
        assert "Error, Tensor_Manager is not yet initialized."
    return __TENSOR_MANAGER.get_weight_count_in_gpu(), __TENSOR_MANAGER.get_activation_numel_in_gpu()

def chk_tensor_registered(tensor:Union[Tensor, NF4Tensor]):
    if __TENSOR_MANAGER == None:
        assert "Error, Tensor_Manager is not yet initialized."
    return hasattr(tensor, "tensor_id") and __TENSOR_MANAGER.chk_id_availibility(tensor.tensor_id)
    
def init_tensor_manager(stage_num:int = 1, batch_size:int = 1, swap_activation:bool = False):
    global __TENSOR_MANAGER
    if __TENSOR_MANAGER == None:
        __TENSOR_MANAGER = TensorManager(stage_num, batch_size, swap_activation)
    
    print("Tensor Manager Initialized")
        
def register_activation(*activations):
    if __TENSOR_MANAGER == None:
        assert "Error, Tensor_Manager is not yet initialized."
    tensor_ids = []
    for activation in activations:
        if type(activation) == Tensor or type(activation) == NF4Tensor or type(activation) == Parameter:
            if hasattr(activation, "tensor_id"):
                tensor_ids.append(activation.tensor_id)
            else:
                tensor_ids.append(__TENSOR_MANAGER.register(activation, "activation"))
                activation.tenson_id = tensor_ids[-1]
        elif type(activation) == tuple or type(activation) == list:
            tensor_ids.append(register_activation(*activation))
        else:
            tensor_ids.append(activation)
            #raise TypeError("Activation should be a tensor. But got type: {}".format(type(activation)))

    return PackTensorList(__TENSOR_MANAGER, *tensor_ids)

def set_stage_and_batch_id(stage_id:int, batch_id:int = 0, operation:str = "forward"):
    if __TENSOR_MANAGER == None:
        assert "Error, Tensor_Manager is not yet initialized."
    __TENSOR_MANAGER.set_stage_and_batch_id(stage_id, batch_id, operation)
    
def pause_tensor_manager():
    if __TENSOR_MANAGER == None:
        assert "Error, Tensor_Manager is not yet initialized."
    __TENSOR_MANAGER.pause()
    
def resume_tensor_manager():
    if __TENSOR_MANAGER == None:
        assert "Error, Tensor_Manager is not yet initialized."
    __TENSOR_MANAGER.resume()

def total_moving_time():
    if __TENSOR_MANAGER == None:
        assert "Error, Tensor_Manager is not yet initialized."
    return __TENSOR_MANAGER.total_moving_time()