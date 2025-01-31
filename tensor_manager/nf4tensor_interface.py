import torch
from .nf4tensor import to_nf4, NF4Tensor
from .base_tensor_interface import TensorInterfaceBase
from typing import overload, Optional, Union

class NF4TensorInterface(TensorInterfaceBase):
    def __init__(self, tensor:NF4Tensor, numel, size, device, dtype):
        super().__init__(numel, size, device, dtype)
        self.packed = tensor
    
    @classmethod
    def compress(cls, tensor):
        tensor_nf4 = to_nf4(tensor)
        return cls(tensor_nf4, tensor.numel(), tensor.size(), tensor.device, tensor.dtype)
    
    def decompress(self, dtype:Optional[torch.dtype]=None):
        return self.packed.get_original_weight().to(dtype if dtype is not None else self.dtype)
    
    @overload
    def to(self, device:Union[torch.device,int,str], non_blocking:bool=False) -> NF4Tensor:
        if device == self.device or device == str(self.device):
            return self
        tensor = self.packed.clone()
        tensor.quantized_data = tensor.quantized_data.to(device, non_blocking=False)
        tensor.quantized_scalers = tensor.quantized_scalers.to(device, non_blocking=False)
        tensor.quantization_factor = tensor.quantization_factor.to(device, non_blocking=False)
        tensor.nf4 = tensor.nf4.to(device, non_blocking=False)
        tensor = tensor.cuda() if device.type == "cuda" else tensor.cpu()
        
        return NF4TensorInterface(tensor, self.numel(), self.size(), device, self.dtype)
    
    @overload
    def to(self, dtype:torch.dtype, non_blocking:bool) -> torch.Tensor:
        return self.decompress(dtype)
        
    
def to_nf4_interface(tensor):
    return NF4TensorInterface.compress(tensor)