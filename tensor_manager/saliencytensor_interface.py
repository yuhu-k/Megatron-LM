import torch
from .rtn4bit_interface import RTN4Bit, to_rtn_4bit
from typing import Optional
from .base_tensor_interface import TensorInterfaceBase

class SaliencyChannelTensor(TensorInterfaceBase):
    def __init__(self, packed_tensor:Optional[RTN4Bit], tensor:torch.Tensor, numel, size, device, dtype, warmup_iters_threshold, metadata=None):
        super().__init__(numel, size, device, dtype)
        self.warmup_iters_thres = warmup_iters_threshold
        self.packed = packed_tensor
        self.origin_tensor = tensor
        self.metadata = metadata if metadata is not None else {"salient_channels": None, "warmup_iter": 0}

    def to(self, device, non_blocking=False):
        if self.metadata['warmup_iter'] < self.warmup_iters_thres:
            return SaliencyChannelTensor(self.packed, self.origin_tensor.to(device, non_blocking=non_blocking), self.numel(), self.size(), device, self.dtype, self.warmup_iters_thres, self.metadata)
        # print("Don't swap device for SaliencyChannelTensor. This tensor should always be on GPU.")

    def warmup(self, input):
        tmp = find_outlier_channels(input)
        self.metadata["salient_channels"] = tmp if self.metadata["salient_channels"] is None else self.metadata["salient_channels"] + tmp
        self.metadata["warmup_iter"] += 1
        if self.metadata["warmup_iter"] == self.warmup_iters_thres:
            self.finish_warmup()
        
    def decompress(self):
        if self.metadata['warmup_iter'] < self.warmup_iters_thres:
            return self.origin_tensor
        
        saliency = self.metadata["salient_channels"]
        size = saliency.size(0)
        _, salient_channels_idx = torch.topk(saliency, k=int(size * 0.1), largest=True, sorted=False)
        high_precision_tensor = self.origin_tensor
        tensor = self.packed.decompress()
        tensor[:, salient_channels_idx] = high_precision_tensor.to(torch.cuda.current_device())[:, salient_channels_idx]
        return tensor
    
    def pin_memory(self):
        self.tensor = self.tensor.pin_memory()
        return self
    
    def finish_warmup(self):
        self.packed = to_rtn_4bit(self.origin_tensor.to(torch.cuda.current_device(), non_blocking=True))
        # self.origin_tensor = self.origin_tensor.to("cpu")
        
    @classmethod
    def compress(cls, tensor: torch.Tensor, warmup_iter_threshold):
        return cls(None, tensor, tensor.numel(), tensor.size(), tensor.device, tensor.dtype, warmup_iter_threshold)

def find_outlier_channels(tensor: torch.Tensor, outlier_percent=1) -> torch.Tensor:
    """
    找出張量中的 outlier channel。

    參數：
        tensor (torch.Tensor): 輸入的張量，形狀為 (N, C, H, W) 或 (N, C)。
        outlier_percent (float): 判定為 outlier 的百分比 (0-100)。預設為 1%。

    返回：
        torch.Tensor: 一個向量，形狀為 (C,)。若某 channel 為 outlier 則對應值為 1，否則為 0。
    """
    if tensor.dim() == 3:
        out_tensor = torch.zeros(tensor.size(2), dtype=torch.int16, device=tensor.device)
        # 使用 torch.unbind 在第二維度分割成 n 個二維張量
        split_tensors = torch.unbind(tensor, dim=1)

        # 檢查結果
        for idx, t in enumerate(split_tensors):
            # print(t.dim())
            out_tensor += find_outlier_channels(t, outlier_percent)
        return out_tensor
            
    if tensor.dim() not in [2, 4]:
        raise ValueError(f"輸入張量必須是 2D (N, C) 或 4D (N, C, H, W) 的形式, 但得到的形狀為 {tensor.shape}")

    # 若為 4D 張量，先對 H 和 W 維度進行平均，得到每個 channel 的特徵值
    if tensor.dim() == 4:
        tensor = tensor.mean(dim=(-1, -2))  # 現在形狀為 (N, C)

    # 對每個 channel 計算其平均值（跨樣本 N 維）
    channel_means = tensor.float().abs().mean(dim=0)  # (C,)
    all_means = channel_means.mean()

    # # 計算百分位範圍
    # low_percentile = outlier_percent / 2  # 下百分位
    # high_percentile = 100 - low_percentile  # 上百分位

    # # 使用 torch.quantile 計算閾值
    # threshold_low = torch.quantile(channel_means, low_percentile / 100)
    # threshold_high = torch.quantile(channel_means, high_percentile / 100)

    # 判斷每個 channel 是否為 outlier
    # outliers = (channel_means < threshold_low) | (channel_means > threshold_high)
    outliers = channel_means >= all_means*2

    # 將布林值轉換為整數類型 (1: outlier, 0: not outlier)
    outlier_vector = outliers.to(torch.int16).to(torch.cuda.current_device())
    
    return outlier_vector

def decompress_salient_tensor(tensor: SaliencyChannelTensor):
    return tensor.decompress()

def to_saliency_channel_tensor(tensor: torch.Tensor, warmup_iters=100):
    return SaliencyChannelTensor.compress(tensor, warmup_iters)