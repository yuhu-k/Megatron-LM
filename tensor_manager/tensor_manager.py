from torch import Tensor
import threading
import torch
# from torchao.dtypes.nf4tensor import NF4Tensor
import copy
from .nf4tensor import NF4Tensor

class OrderList:
    def __init__(self, order: list):
        self.pointer = 0
        self.order = order
        pass
    
    def get(self):
        self.pointer += 1
        return self.order[self.pointer-1]

class TensorManager:
            
    def __init__(self, warmup_iteration_times = 100) -> None:
        self.tensor_list:list[list[torch.Tensor,torch.Tensor]] = []
        self.tensor_id_count = 0
        
        self.warmup = True
        self.use_order = [[]]
        self.use_order_pointer = 0
        self.max_tensor_num_in_gpu = 1
        self.num_tensor_in_gpu = 0
        self.warmup_iteration_times = warmup_iteration_times
        self.load_order = []
        self.load_pointer = 0
        self.weight_use_time = []
        self.tensor_use_threshold = 1

        self.stream = torch.cuda.Stream(device=torch.cuda.current_device()) 
        # self.streams = [torch.cuda.Stream(device=torch.cuda.current_device())  for _ in range(4)]
        
        # self.mtx = None
        # self.thread_output = None
        # self.thread = None
        
    def chk_id_availibility(self, tensor_id:int) -> bool:
        if type(tensor_id) != int:
            return False
        return self.tensor_id_count > tensor_id
        
    def register(self, t:Tensor):
        tensor_id = self.tensor_id_count
        self.tensor_id_count += 1
        self.tensor_list.append([t, None])

        self.weight_use_time.append(0)
        return tensor_id
        
    def get_tensor(self, tensor_id):
        torch.cuda.nvtx.range_push("synchronize dequantize")
        self.stream.synchronize()
        torch.cuda.nvtx.range_pop()
        self.get_computable_form(tensor_id, False)
        if self.warmup:
            self.use_order[-1].append((tensor_id, "load"))
        self.weight_use_time[tensor_id] += 1
        output = self.tensor_list[tensor_id][1].clone().detach().requires_grad_(False)
        if self.weight_use_time[tensor_id] >= self.tensor_use_threshold:
            self.offload_tensor(tensor_id)
            self.weight_use_time[tensor_id] = 0
            

        return output
            
    def offload_tensor(self, tensor_id):
        self.use_order_pointer += 1
        if self.warmup:
            self.use_order[-1].append((tensor_id, "offload"))
        else:
            if not self.decide_offload_tensor(tensor_id, self.use_order_pointer, self.num_tensor_in_gpu):
                return
            
        self.tensor_list[tensor_id][1] = None
        self.num_tensor_in_gpu -= 1

            
    # def dump_user_order(self):
    #     with open("/tmp2/tests.txt", "w") as f:
    #         i = 1
    #         for l in self.order_list:
    #             f.write(f"{i}: " + str(l) + "\n")
    #             i += 1
            
    def finish_warmup(self):
        if self.warmup:
            self.warmup_iteration_times -= 1
            if self.warmup_iteration_times <= 0:
                self.warmup = False
                self.make_use_order()
            else:
                self.use_order.append([])
        self.use_order_pointer = 0
        self.load_pointer = 0
                
    def make_use_order(self):
        self.use_order = self.use_order[0] + self.use_order[0]
        self.load_order = [x[0] for x in self.use_order if x[1] == "load"]
    
    def get_computable_form(self, tensor_id, non_blocking=True):
        if self.tensor_list[tensor_id][1] is None:
            tensor = self.tensor_list[tensor_id][0]
            if tensor != None:
                torch.cuda.nvtx.range_push(f"get_computable_form {tensor_id}")
                with torch.cuda.stream(self.stream):
                    if str(tensor.device) == "cpu":
                        if type(tensor) == NF4Tensor:
                            torch.cuda.nvtx.range_push(f"swap {tensor_id}")
                            tensor = tensor.clone()
                            tensor.quantized_data = tensor.quantized_data.to(torch.cuda.current_device(), non_blocking=False)
                            tensor.quantized_scalers = tensor.quantized_scalers.to(torch.cuda.current_device(), non_blocking=False)
                            tensor.quantization_factor = tensor.quantization_factor.to(torch.cuda.current_device(), non_blocking=False)
                            tensor.nf4 = tensor.nf4.to(torch.cuda.current_device(), non_blocking=False)
                            tensor = tensor.cuda()
                            torch.cuda.nvtx.range_pop()
                        else:
                            tensor = tensor.to(torch.cuda.current_device(), non_blocking=non_blocking)
                    if type(tensor) == NF4Tensor:
                        torch.cuda.nvtx.range_push(f"dequantize {tensor_id}")
                        tensor = tensor.get_original_weight().to(tensor.dtype, non_blocking=non_blocking)
                        torch.cuda.nvtx.range_pop()
                    if hasattr(tensor, "tensor_id"):
                        del tensor.tensor_id
                    
                    self.tensor_list[tensor_id][1] = tensor.detach()
                    self.num_tensor_in_gpu += 1
                if not non_blocking:
                    self.stream.synchronize()
                torch.cuda.nvtx.range_pop()

                # self.tensor_list[tensor_id][1] = tensor.detach()
            else:
                RuntimeError(f"Error, missing tensor id {tensor_id}")
            
            # if not self.warmup:
            #     with torch.cuda.stream(self.stream):
                    
            #         for i in range(4):
            #             with torch.cuda.stream(self.streams[i]):
            #                 self.get_computable_form(self.get_next_tensor_id(), True)
            #         for stream in self.streams:
            #             stream.synchronize()
        # return self.tensor_list[tensor_id][1]
    
    def get_next_tensor_id(self):
        self.load_pointer = (self.load_pointer + 1) % len(self.load_order)
        return self.load_order[self.load_pointer]
            
            
        
    def decide_offload_tensor(self, tensor_id, pointer, num_tensors_in_gpu) -> bool:
        if self.warmup or num_tensors_in_gpu > self.max_tensor_num_in_gpu:
            return True

        # 預先計算每個張量的最後使用位置
        if not hasattr(self, 'last_use_index'):
            self.last_use_index = {}
            for i in range(len(self.use_order)-1, -1, -1):
                tid, _ = self.use_order[i]
                if tid not in self.last_use_index:
                    self.last_use_index[tid] = i

        tmp_num_in_gpu = num_tensors_in_gpu
        offloaded_tensors = set()

        for i in range(pointer, len(self.use_order)):
            current_tensor_id, action = self.use_order[i]
            if current_tensor_id == tensor_id:
                # 如果張量即將被使用，不應該卸載
                return False
            if action == "load":
                tmp_num_in_gpu += 1
            elif action == "offload":
                if current_tensor_id in offloaded_tensors:
                    continue
                # 如果張量不再被使用，可以卸載
                if self.last_use_index.get(current_tensor_id, -1) <= i:
                    tmp_num_in_gpu -= 1
                    offloaded_tensors.add(current_tensor_id)
            # 如果GPU中的張量數超過限制，需要卸載
            if tmp_num_in_gpu > self.max_tensor_num_in_gpu:
                return True
        return True
                        
    # def decide_offload_tensor(self, tensor_id, pointer, num_tensors_in_gpu) -> bool:
    #     if self.warmup or num_tensors_in_gpu > self.max_tensor_num_in_gpu:
    #         return True
    #     tmp_num_in_gpu = self.num_tensor_in_gpu
    #     for i in range(pointer,self.use_order.__len__()):
    #         if tensor_id == self.use_order[i][0]:
    #             return False
    #         else:
    #             if self.use_order[i][1] == "load":
    #                 tmp_num_in_gpu += 1
    #             elif self.use_order[i][1] == "offload":
    #                 if self.decide_offload_tensor(self.use_order[i][0], i, tmp_num_in_gpu):
    #                     tmp_num_in_gpu -= 1
    #         if tmp_num_in_gpu > self.max_tensor_num_in_gpu:
    #             return True
    #     return True
        
    # def update_tmp_list(self):
    #     for id in range(self.tmp_list.__len__()):
    #         self.tmp_list[id].append(self.tmp_list[id].pop(0))
    
    # def reset_tmp_list(self):
    #     if not self.warmup:
    #         self.tmp_list = self.order_list.copy()
    #         self.update_tmp_list()
        
    def get_in_gpu_tensor_ratio(self):
        count = 0
        for w in self.tensor_list:
            if w[1] is not None:
                count += 1
                
        return count / self.tensor_list.__len__()
    
    # def deal_thread_result(self):
    #     with open("/tmp2/a.txt", "a") as f:
    #         f.write(f"c {self.thread}\n")
    #     if self.thread != None:
    #         self.thread.join()
    #         self.thread = None
    #         if self.mtx != None:
    #             self.mtx.acquire()
    #             tensor_id, tensor = self.thread_output
    #             self.thread_output = None
    #             self.mtx.release()
    #         if hasattr(tensor, "tensor_id"):
    #             del tensor.tensor_id
    #         with open("/tmp2/a.txt", "a") as f:
    #             f.write(f"b {tensor_id} {tensor}\n")
    #         self.tensor_list[tensor_id][1] = tensor.detach()

    # def transform_tensor_threading(self, tensor_id):
    #     tensor = self.tensor_list[tensor_id][0]
    #     self.mtx = threading.Lock()
    #     def thread_func():
    #         with open("/tmp2/a.txt", "a") as f:
    #             f.write(f"e {type(tensor)}\n")
    #         tensor = tensor.to(tensor.dtype)
    #         with open("/tmp2/a.txt", "a") as f:
    #             f.write(f"a {tensor_id} {tensor}\n")
    #         self.mtx.acquire()
    #         self.thread_output = (tensor_id, tensor)
    #         self.mtx.release()
            
    #     self.thread = threading.Thread(target=thread_func)
    #     self.thread.start()
        
    #     with open("/tmp2/a.txt", "a") as f:
    #         f.write(f"d {self.thread} {type(tensor)}\n")
    
    
    # def add_swap_thread(self, tensor_id):
    #     def swap_to_gpu(tensor_id):
    #         self.lock_queue.acquire()
    #         queue = self.swap_queue
    #         self.lock_queue.release()
    #         for x in queue:
    #             if x == tensor_id:
    #                 break
    #             self.lock_threads.acquire()
    #             thread = self.swap_thread[x]
    #             self.lock_threads.release()
    #             thread.join()
            
    #         self.swapping_now = tensor_id
            
    #         self.lock_tensor_dict.acquire()
    #         tensor = self.tensor_dict[tensor_id]
    #         self.gpu_tensor_dict[tensor_id] = tensor.to(self.device_dict[tensor_id], non_blocking=False).detach()
    #         self.lock_tensor_dict.release()
        
    #     self.lock_tensor_dict.acquire()
    #     tensor = self.gpu_tensor_dict[tensor_id]
    #     self.lock_tensor_dict.release()
    #     if tensor is None:
    #         thread = threading.Thread(target=swap_to_gpu, args=(tensor_id,))
    #         self.lock_threads.acquire()
    #         self.swap_thread[tensor_id] = thread
    #         self.lock_threads.release()
    #         self.lock_queue.acquire()
    #         self.swap_queue.append(tensor_id)
    #         self.lock_queue.release()
    #         thread.start()
            # self.lock_main_thread.acquire()
            # main_thread = self.main_thread
            # self.lock_main_thread.release()
            # if main_thread == None:
            #     self.start_swap_main_thread()
            

    # def start_swap_main_thread(self):
    #     def manage_thread():
    #         self.lock_queue.acquire()            
    #         while(self.swap_queue.__len__() != 0):
    #             thread_id = self.swap_queue.pop(0)
    #             self.lock_queue.release()
    #             self.lock_threads.acquire()
    #             thread = self.swap_thread[thread_id]
    #             self.lock_threads.release()
    #             thread.start()
    #             thread.join()
    #             self.lock_threads.acquire()
    #             self.swap_thread[thread_id] = None
    #             self.lock_threads.release()
    #             self.lock_queue.acquire()
    #         self.lock_main_thread.acquire()
    #         self.main_thread = None
    #         self.lock_main_thread.release()
            
    #     self.lock_main_thread.acquire()
    #     self.main_thread = threading.Thread(target=manage_thread)
    #     self.main_thread.start()
    #     self.lock_main_thread.release()
        
    # def cut_in_swap_queue(self, tensor_id):
    #     self.swap_queue.insert(0, tensor_id)
    
