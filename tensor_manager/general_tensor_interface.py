import torch
from .base_tensor_interface import TensorInterfaceBase

class GeneralTensorInterface(TensorInterfaceBase):
    def __init__(self, tensor:torch.Tensor, numel, size, device, dtype):
        super().__init__(numel, size, device, dtype)
        self.packed = tensor
    
    @classmethod
    def compress(cls, tensor:torch.Tensor):
        return cls(tensor, tensor.numel(), tensor.size(), tensor.device, tensor.dtype)
    
    def decompress(self):
        return self.packed
    
    def to(self, device, non_blocking=False):
        return GeneralTensorInterface(self.packed.to(device, non_blocking=non_blocking), self.numel(), self.size(), device, self.dtype)
    
def to_general_interface(tensor):
    return GeneralTensorInterface.compress(tensor)