import torch
from .base_tensor_interface import TensorInterfaceBase

class RTN4Bit(TensorInterfaceBase):
    def __init__(self, min_val, max_val, packed_tensor, numel, size, device, dtype):
        super().__init__(numel, size, device, dtype)
        self.min_val = min_val
        self.max_val = max_val
        self.packed = packed_tensor
    
    @classmethod
    @torch.no_grad() 
    def compress(cls, tensor):
        # 紀錄張量的屬性
        min_val = tensor.min()
        max_val = tensor.max()

        # 壓縮：量化和打包
        quantized = torch.round((tensor - min_val) / (max_val - min_val) * 15).to(torch.uint8)
        
        # 確保展平為 1D 張量進行打包
        quantized = quantized.view(-1)
        packed = (quantized[::2] << 4) | quantized[1::2]
        
        return cls(min_val, max_val, packed, tensor.numel(), tensor.size(), tensor.device, tensor.dtype)

    def decompress(self):
        # 解壓縮
        packed = self.packed
        num_quantized = packed.size(0) * 2

        # 初始化解壓縮張量，並確保形狀匹配
        quantized = torch.empty(num_quantized, dtype=torch.uint8, device=packed.device)
        
        # 從 packed 張量中解碼
        quantized[::2] = packed >> 4      # 高 4 位
        quantized[1::2] = packed & 0x0F   # 低 4 位
        
        # 還原到浮點值
        matrix = quantized.float() / 15 * (self.max_val - self.min_val) + self.min_val

        # 恢復為原始形狀
        return matrix.view(self.size()).to(self.dtype)
    
    def to(self, device, non_blocking=False):
        out = RTN4Bit(self.min_val, self.max_val, self.packed.to(device, non_blocking=non_blocking), self.numel(), self.size(), device, self.dtype)
        return out
    
def decompress_4bit_to_bf16(packed:RTN4Bit) -> torch.Tensor:
    return packed.decompress()

def to_rtn_4bit(tensor:torch.Tensor) -> RTN4Bit:
    return RTN4Bit.compress(tensor)

# # 壓縮 BF16 矩陣到 4-bit
# def compress_bf16_to_4bit(matrix):
#     # 找到矩陣的最小值和最大值
#     min_val = matrix.min()
#     max_val = matrix.max()
    
#     # 量化到 [0, 15]
#     quantized = torch.round((matrix - min_val) / (max_val - min_val) * 15).to(torch.uint8)
    
#     # 打包到 4-bit（每 2 個值存到 1 個字節）
#     packed = (quantized[::2] << 4) | quantized[1::2]
    
#     return packed, min_val, max_val

# # 解壓縮 4-bit 回 BF16 矩陣
# def decompress_4bit_to_bf16(packed, min_val, max_val):
#     # 解包 4-bit 數據
#     quantized = torch.empty(packed.size(0) * 2, dtype=torch.uint8, device=packed.device)
#     quantized[::2] = packed >> 4
#     quantized[1::2] = packed & 0x0F
    
#     # 解量化
#     matrix = quantized.float() / 15 * (max_val - min_val) + min_val
#     return matrix

