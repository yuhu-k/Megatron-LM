import torch

from .spectator import spectator
from .utils import report_to_torch_profiler
from .context import context
from .storage_manager import storage_manager
from .multi_gpu import multigpu_iteration_guard

class PackedTensor:
    def __init__(self, tensor, tid):
        self.tensor = tensor
        self.shape = tensor.size()
        self.size_ = tensor.numel() * tensor.element_size()
        self.ptr_ = tensor.storage().data_ptr()
        self.tid_ = tid
        self.original_device = self.tensor.device
        self.reshaped = False

        self.ref_cnt_ = 1

        spectator.set_tid_size(tid, self.size_)

    def swap_out(self):
        tensor = self.tensor
        context.offload_stream.wait_stream(context.default_stream)

        spectator.increase_swap_out_size(self.size())
        spectator.timer_begin('swap_out', self.tid(), context.offload_stream)
        multigpu_iteration_guard.swap_out()
        
        with torch.cuda.stream(context.offload_stream):
            tensor.record_stream(context.offload_stream)
            packed = torch.empty(
                tensor.size(),
                dtype=tensor.dtype,
                layout=tensor.layout,
                pin_memory=(not tensor.is_sparse))
                
            packed.copy_(tensor, non_blocking=True)

        spectator.timer_end('swap_out', self.tid())
        
        # context.offload_stream.synchronize()
        
        storage_manager.alloc(packed.storage().data_ptr(), -1, self.size(), self.tid)
                
        self.tensor = packed
        
    def swap_in(self):
        packed = self.tensor

        spectator.timer_begin('swap_in', self.tid(), context.prefetch_stream)
        
        multigpu_iteration_guard.swap_in()

        cpu_ptr = packed.storage().data_ptr()
        with torch.cuda.stream(context.prefetch_stream):
            tensor = packed.to(self.original_device, non_blocking=True)
            context.default_stream.wait_stream(context.prefetch_stream)
        tensor.record_stream(context.default_stream)

        spectator.timer_end('swap_in', self.tid())
        storage_manager.delete(cpu_ptr, -1, self.size())

        self.tensor = tensor

    def inc(self):
        self.ref_cnt_ += 1

    def dec(self):
        self.ref_cnt_ -= 1
                
    def is_swapped_out(self):
        return self.original_device != self.tensor.device

    def get(self):
        return self.tensor

    def size(self):
        return self.size_

    def tid(self):
        return self.tid_

    def ptr(self):
        return self.ptr_

    def ref_cnt(self):
        return self.ref_cnt_


# prevent multiple packed call on same tensor
class TensorCache:
    def __init__(self):
        self.reset()

    def reset(self):
        self.current_tid = 0
        self.cache:dict[int, PackedTensor] = dict() # tid -> PT, for tensor in GPU and CPU
        self.location = dict() # addr -> tid, only for tensor in GPU
        self.small_tensors = dict() # tid -> tensor, only for tensors whose size is smaller than threshold

    def pack(self, tensor) -> int: # tensor -> packed tensor
        if (tensor.numel() * tensor.element_size()) < context.basic_size_threshold:
            self.small_tensors[self.current_tid] = tensor
            self.current_tid += 1
            return self.current_tid - 1
        report_to_torch_profiler(f"_swap_out|{self.current_tid}")
        tid = self.location.get(tensor.storage().data_ptr())
        if tid is not None:
            packed = self.cache.get(tid)
            if packed is not None:
                if packed.shape != tensor.size():
                    # a new PT (share same storage with a PT already in tensor cache)
                    packed = PackedTensor(tensor, self.current_tid)
                    self.current_tid = self.current_tid + 1

                    packed.reshaped = True 
                if packed.is_swapped_out():
                    print("buggy, the packed tensor is swapped out")
                    exit(1)
            else:
                print("buggy, cannot find tid in tensor cache")
                exit(1)
            
        else: 
            packed = PackedTensor(tensor, self.current_tid)
            self.location[tensor.storage().data_ptr()] = self.current_tid
            self.current_tid = self.current_tid + 1

            if context.enable_swap:
                if context.policy.is_swap(packed.tid(), packed.size()):
                    self.location.pop(tensor.storage().data_ptr(), None)
                    packed.swap_out()
        
        self.__add(packed)
        return packed.tid()
        
    def unpack(self, packed): # packed tensor -> tensor
        if self.small_tensors.get(packed) != None:
            tensor = self.small_tensors.pop(packed)
            return tensor
        packed = self.cache.get(packed) ## 0513
        report_to_torch_profiler(f"_swap_in|{packed.tid()}")
        # first time swap in
        if packed.is_swapped_out():
            packed.swap_in()
            if context.policy.is_last_tensor(packed.tid()):
                #print("last swap in", packed.tid())
                event = torch.cuda.Event(enable_timing=False)
                event.record(context.prefetch_stream)
                context.last_swap_event = event

        tensor = packed.get()
        tensor_cache.location[tensor.storage().data_ptr()] = packed.tid()
        packed.dec()
        
        if packed.ref_cnt() == 0:
            self.__remove(packed)
            self.location.pop(tensor.storage().data_ptr(), None)

        return tensor

    def __add(self, packed):
        self.cache[packed.tid()] = packed

    def __remove(self, packed):
        self.cache.pop(packed.tid(), None)

tensor_cache = TensorCache()

def pack_hook(tensor):
    # if (tensor.numel() * tensor.element_size()) < context.basic_size_threshold:
    #     return tensor
    spectator.record_memory_footprint()
    
    return tensor_cache.pack(tensor)

def unpack_hook(tensor):
    spectator.record_memory_footprint()

    #if isinstance(tensor, PackedTensor):
    return tensor_cache.unpack(tensor)
    #return tensor

class PackHookInitializer:
    def __init__(self, profile=False, is_enable=False, non_blocking = False) -> None:
        self.__enable = is_enable
        self.__profile = profile
        self.blocking = not non_blocking
        self.name_list = {}
    
    def enable(self):
        self.__enable = True
    
    def disable(self):
        self.__enable = False

    @torch.no_grad()
    def my_pack_hook(self, *tensors):
        if self.__enable:
            l:list[torch.Tensor] = []
            for tensor in tensors:
                if self.__profile:
                    torch.cuda.nvtx.range_push("Pack tensor")
                l.append(torch.tensor([[pack_hook(tensor)]]))
                if self.blocking:
                    self.wait()
                if self.__profile:
                    torch.cuda.nvtx.range_pop()
            return tuple(l)
        else:
            return tensors

    @torch.no_grad()
    def my_unpack_hook(self, *tensors) -> tuple:
        if self.__enable:
            l = []
            for tensor in tensors:
                if self.__profile:
                    torch.cuda.nvtx.range_push("Unpack tensor")
                l.append(unpack_hook(int(tensor[0][0])))
                if self.blocking:
                    self.wait()
                if self.__profile:
                    torch.cuda.nvtx.range_pop()
            return tuple(l)
        else:
            return tensors

    def wait(self):
        torch.cuda.synchronize()
        
    def pack_tensor_with_name(self, tensor, name):
        if self.name_list.get(name) != None:
            print(f"Error, tensor{name} is in cpu")
            exit(1)
        self.name_list[name] = self.my_pack_hook(tensor)
    
    def unpack_tensor_with_name(self, name):
        if self.name_list.get(name) == None:
            print(f"Error, tensor{name} is not in cpu")
            exit(1)
        unpack = self.my_unpack_hook(self.name_list[name])
        return unpack[0]
        