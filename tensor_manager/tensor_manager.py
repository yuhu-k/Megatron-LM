from torch import Tensor
import threading
import torch
# from torchao.dtypes.nf4tensor import NF4Tensor
from .nf4tensor import NF4Tensor
import json
import numpy as np
from typing import Optional
from functools import partial

class TensorsBucket:
    def __init__(self):
        self.tensor_list:list[Optional[torch.Tensor], Optional[torch.Tensor]] = [] # [CPU tensor, GPU tensor]
        self.tensor_count = 0
        self.non_used_tensor_id = []
        self.tensor_metadata = []
        self.computable_tensor_list:list[Optional[torch.Tensor]] = []
        self.stream = torch.cuda.Stream(device=torch.cuda.current_device())
        
        self.events = []
        
            
    def register(self, tensor:Tensor):
        tensor_id = None
        for i in range(self.non_used_tensor_id.__len__()):
            if self.tensor_list[self.non_used_tensor_id[i]] != None and self.tensor_metadata[self.non_used_tensor_id[i]]["size"] == tensor.size():
                tensor_id = self.non_used_tensor_id.pop(i)
                self.tensor_metadata[tensor_id]["status"] = True
                break
                # self.tensor_list[tensor_id].copy_(tensor, non_blocking=True)
                # self.tensor_metadata[tensor_id]["status"] = True
        if tensor_id == None:
            tensor_id = self.tensor_count
            self.tensor_count += 1
            self.tensor_list.append([None, None])
            self.tensor_metadata.append({"size": tensor.size(), "status": True, "synchronous": None, "numel": tensor.numel()})
        #self.tensor_list.append(tensor.pin_memory() if tensor.device.type == "cpu" else tensor.to("cpu", non_blocking=True))
        if tensor.device.type == "cpu":
            self.tensor_list[tensor_id][0] = tensor.pin_memory()
            self.tensor_metadata[tensor_id]["synchronous"] = "C" # C means the tensor is in CPU
        else:
            self.tensor_list[tensor_id][1] = tensor
            self.tensor_metadata[tensor_id]["synchronous"] = "G"
        return tensor_id
    
    def get_gpu_tensor(self, tensor_id, non_blocking=True) -> Optional[torch.Tensor]:
        if self.tensor_metadata[tensor_id]["status"]:
            self.move_tensor(tensor_id, torch.cuda.current_device(), non_blocking)
            return self.tensor_list[tensor_id][1]
        else:
            return None
    
    def get_cpu_tensor(self, tensor_id, non_blocking=True) -> Optional[torch.Tensor]:
        if self.tensor_metadata[tensor_id]["status"]:
            self.move_tensor(tensor_id, "cpu", non_blocking)
            return self.tensor_list[tensor_id][0]
        else:
            return None
        
    def move_tensor(self, tensor_id, device, non_blocking=True):
        # torch.cuda.synchronize()
        
        torch.cuda.nvtx.range_push(f"move_tensor {tensor_id} to {device}")
        if device == "cpu":
            if self.tensor_list[tensor_id][0] == None:
                with torch.cuda.stream(self.stream):
                    self.tensor_list[tensor_id][0] = self.tensor_list[tensor_id][1].to("cpu", non_blocking=non_blocking).pin_memory()
                self.tensor_metadata[tensor_id]["synchronous"] = "ALL"
                # if not non_blocking:
                #     self.stream.synchronize()
            elif self.tensor_metadata[tensor_id]["synchronous"] == "G":
                with torch.cuda.stream(self.stream):
                    self.tensor_list[tensor_id][0].copy_(self.tensor_list[tensor_id][1], non_blocking=non_blocking)
                self.tensor_metadata[tensor_id]["synchronous"] = "ALL"
                # if not non_blocking:
                #     self.stream.synchronize()
        else:
            if self.tensor_list[tensor_id][1] == None or self.tensor_metadata[tensor_id]["synchronous"] == "C":
                with torch.cuda.stream(self.stream):
                    self.tensor_list[tensor_id][1] = self.tensor_list[tensor_id][0].to(device, non_blocking=non_blocking)
                self.tensor_metadata[tensor_id]["synchronous"] = "ALL"
                # if not non_blocking:
                #     self.stream.synchronize()
        # torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()

    def delete(self, tensor_id):
        self.tensor_list[tensor_id][1] = None
        self.non_used_tensor_id.append(tensor_id)
        self.tensor_metadata[tensor_id]["status"] = False
        
    def offload_tensor(self, tensor_id, non_blocking=True):
        if self.tensor_metadata[tensor_id]["status"]:
            if self.tensor_list[tensor_id][1] != None:
                self.move_tensor(tensor_id, "cpu", non_blocking)
                self.tensor_list[tensor_id][1] = None
                self.tensor_metadata[tensor_id]["synchronous"] = "C"
    
    def get_tensor_size(self, tensor_id):
        return self.tensor_metadata[tensor_id]["size"]
    
    def chk_tid_in_use(self, tensor_id) -> bool:
        return self.tensor_metadata[tensor_id]["status"]
    
    def get_tensor_numel(self, tensor_id):
        return self.tensor_metadata[tensor_id]["numel"]
    
    def chk_tensor_in_gpu(self, tensor_id):
        return self.chk_tid_in_use(tensor_id) and (self.tensor_metadata[tensor_id]["synchronous"] == "G" or self.tensor_metadata[tensor_id]["synchronous"] == "ALL")

    def total_move_tensors_time(self):
        time = 0
        for start_event, end_event in self.events:
            if not end_event.query():
                break
            time += start_event.elapsed_time(end_event)
        self.events = []
        
        return time

class TensorManager:
            
    def __init__(self, stage_num, batch_num, activation_swapping) -> None:
        self.tensor_bucket = TensorsBucket()
        # self.tensor_list:list[list[Optional[int],torch.Tensor]] = []
        # self.tensor_id_count = 0
        
        # self.use_order = [[]]
        # self.use_order_pointer = 0
        self.max_tensor_num_in_gpu = 1
        self.num_tensor_in_gpu = 0
        # self.load_order = []
        # self.load_pointer = 0
        # self.weight_use_time = []
        self.metadata = {}
        self.tensor_use_threshold = batch_num
        self.swap_threshold = 1024 * 1024
        self.stage_to_weight = {i:[] for i in range(stage_num)}
        self.stage_batch_to_activation = {(i, j):[] for i in range(stage_num) for j in range(batch_num)}
        
        self.stage_num = stage_num # virtual pipeline degree
        self.batch_num = batch_num
        self.pause_flag = True    # To avoid prefetching when loading the weights, pause tensor manager and activate after loading the all weights
        self.activation_swapping = activation_swapping

        self.stream = torch.cuda.Stream(device=torch.cuda.current_device())
        self.stage_id = 0
        self.batch_id = 0
        self.operation = "forward"
        
        self.id_to_stage = {}
        self.stage_id_to_numels = {}
        
        # # 開啟並讀取 JSON 檔案
        # with open('/tmp2/Megatron-LM/results-5.json', 'r', encoding='utf-8') as file:
        #     data = json.load(file)  # 將 JSON 內容轉換為 Python 字典
        # self.swap_setting = {}
        # for i in range(len(data)):
        #     self.swap_setting[data[i]["layer_idx"]//2, data[i]["batch_idx"], data[i]["stage"]] = {"prefetch_ration":data[i]["layer_prefetch_ratio"], "activation_offload":data[i]["activation_offload_result"]}
            
        # self.swap_list = np.full((40, 32, 2), 0)

        # self.thread = None
    
    def get_weight_count_in_gpu(self):
        counts = []
        for stage_id in range(self.stage_num):
            count = 0
            if self.stage_to_weight[stage_id].__len__() != 0:
                tid = self.stage_to_weight[stage_id][0]
                count += 1 if self.tensor_bucket.chk_tensor_in_gpu(tid) else 0
            counts.append(count)
        return counts
    
    def get_activation_numel_in_gpu(self):
        numels = []
        total_numel = 0
        for stage_id in range(self.stage_num):
            numel = 0
            for batch_id in range(self.batch_num):
                if self.stage_batch_to_activation[(stage_id, batch_id)].__len__() != 0:
                    for tid in self.stage_batch_to_activation[(stage_id, batch_id)]:
                        numel += self.tensor_bucket.get_tensor_numel(tid) if self.tensor_bucket.chk_tensor_in_gpu(tid) else 0
                        total_numel += self.tensor_bucket.get_tensor_numel(tid)
            numels.append(numel/1024/1024)
        numels.append(total_numel/1024/1024)
        return numels
    
    def set_stage_and_batch_id(self, stage_id, batch_id, operation):
        self.stage_id = stage_id
        self.batch_id = batch_id
        self.operation = operation
        if self.pause_flag:
            return
        if self.activation_swapping:
            if operation == "forward":
                if stage_id == self.stage_num - 1 and batch_id == self.batch_num - 1:
                    self.prefetch_activation([stage_id], [0])
            if operation == "backward":
                if batch_id != self.batch_num - 1:
                    self.prefetch_activation([stage_id], [batch_id+1])
                elif stage_id > 0:
                    self.prefetch_activation([stage_id-1], [0])
        
        if operation == "forward":
            if stage_id != self.stage_num-1 and batch_id == 0:
                self.prefetch_weight([stage_id + 1], 1)
        if operation == "backward":
            if stage_id != 0 and batch_id == 0:
                self.prefetch_weight([stage_id - 1], 1)
    
    # To avoid unnecessary activation swap when using activation checkpointing, tensormanger has to pause in the recomputation proceding
    def pause(self):
        self.pause_flag = True
        
    def resume(self):
        self.pause_flag = False
        
    def chk_id_availibility(self, tensor_id:int) -> bool:
        # if type(tensor_id) != int:
        #     return False
        # return self.tensor_id_count > tensor_id
        return self.tensor_bucket.chk_tid_in_use(tensor_id)
        
    def register(self, t:Tensor, tensor_type = "weight") -> int:
        tid = self.tensor_bucket.register(t)
        self.metadata[tid] = {'type': tensor_type, "event": None, "use_time": 0}
        if tensor_type == "weight":
            self.stage_to_weight[self.stage_id].append(tid)
            self.id_to_stage[tid] = (self.stage_id, None)
        elif tensor_type == "activation":
            self.stage_batch_to_activation[(self.stage_id, self.batch_id)].append(tid)
            self.id_to_stage[tid] = (self.stage_id, self.batch_id)
        
        if self.tensor_bucket.get_tensor_numel(tid) <= self.swap_threshold and ((tensor_type == "activation" and self.activation_swapping) or tensor_type == "weight"):
            with torch.enable_grad():
                self.tensor_bucket.offload_tensor(tid)
        # print(f"Register tensor {tid}")
        return tid
    
    def prefetch_activation(self, stage_ids, batch_ids):
        for stage_id in stage_ids:
            for batch_id in batch_ids:
                if self.stage_batch_to_activation.get((stage_id, batch_id)) != None:
                    for tensor_id in self.stage_batch_to_activation[(stage_id, batch_id)]:
                        self.tensor_bucket.get_gpu_tensor(tensor_id, True)
    
    def prefetch_weight(self, stage_ids, percent = 1):
        total_numels = 0
        for stage_id in stage_ids:
            if self.stage_id_to_numels.get(stage_id) == None:
                self.stage_id_to_numels[stage_id] = 0
                for tensor_id in self.stage_to_weight[stage_id]:
                    self.stage_id_to_numels[stage_id] += self.tensor_bucket.get_tensor_numel(tensor_id)
            total_numels += self.stage_id_to_numels[stage_id]
        target_numels = total_numels * percent
        current_numels = 0
        for stage_id in stage_ids:
            if self.stage_to_weight.get(stage_id) != None:
                for tensor_id in self.stage_to_weight[stage_id]:
                    current_numels += self.tensor_bucket.get_tensor_numel(tensor_id)
                    if current_numels > target_numels:
                        break
                    # self.get_computable_form(tensor_id, True)
                    self.tensor_bucket.get_gpu_tensor(tensor_id, True)
                    
    def fetch_rest_weight(self, stage_ids):
        for stage_id in stage_ids:
            if self.stage_to_weight.get(stage_id) != None:
                for tensor_id in self.stage_to_weight[stage_id]:
                    # self.get_computable_form(tensor_id, False)
                    self.tensor_bucket.get_gpu_tensor(tensor_id, True)
                
        
    def get_tensor(self, tensor_id):
        # if self.swap_list[self.stage_id, self.batch_id] == 0:
        #     self.swap_list[self.stage_id, self.batch_id] = 1
        #     self.fetch_rest_weight([self.stage_id])
        #     self.prefetch_activation([self.stage_id], [self.batch_id])

        output = self.tensor_bucket.get_gpu_tensor(tensor_id, False)
        self.metadata[tensor_id]['use_time'] += 1

        if self.metadata[tensor_id]['type'] == "weight":
            output = output.detach().requires_grad_(False)
        if (self.operation == "forward" and self.metadata[tensor_id]['type'] == "weight" and self.metadata[tensor_id]["use_time"] == self.tensor_use_threshold and self.id_to_stage[tensor_id][0] != self.stage_num-1) or \
            (self.operation == "backward" and self.metadata[tensor_id]['type'] == "weight" and self.metadata[tensor_id]["use_time"] == 2*self.tensor_use_threshold and self.id_to_stage[tensor_id][0] != 0 and self.id_to_stage[tensor_id][0] != self.stage_num-1) or \
            (self.metadata[tensor_id]["use_time"] >= 3*self.tensor_use_threshold):
            self.tensor_bucket.offload_tensor(tensor_id)
            self.metadata[tensor_id]["use_time"] = 0
            # print(f"1 Offload stage {self.id_to_stage[tensor_id][0]}")
        elif self.metadata[tensor_id]['type'] == "activation":
            self.delete_tensor(tensor_id)
            
        if output == None:
            RuntimeError(f"Error, missing tensor id {tensor_id}")
        # else:
        #     print(f"Get tensor {tensor_id}")
        #     print(f"Tensor size: {output.size()}")
        #     print(f"Tensor type: {data}")

        return output
    
    def delete_tensor(self, tensor_id):
        # self.tensor_list[tensor_id][0] = None
        # self.tensor_list[tensor_id][1] = None
        # self.weight_use_time[tensor_id] = 0
        self.tensor_bucket.delete(tensor_id)
        self.metadata.pop(tensor_id)
        self.stage_batch_to_activation[(self.stage_id, self.batch_id)].remove(tensor_id)
            
    def offload_tensor(self, tensor_id):
        # self.use_order_pointer += 1
        # if self.warmup:
        #     self.use_order[-1].append((tensor_id, "offload"))
            # if not self.decide_offload_tensor(tensor_id, self.use_order_pointer, self.num_tensor_in_gpu):
            #     return
        # self.tensor_list[tensor_id][1] = None
        # self.num_tensor_in_gpu -= 1
        assert "This function is not in use"

    def finish_warmup(self):
        # if self.warmup:
        #     self.warmup_iteration_times -= 1
        #     if self.warmup_iteration_times <= 0:
        #         self.warmup = False
        #         self.make_use_order()
        #     else:
        #         self.use_order.append([])
        # self.use_order_pointer = 0
        # self.load_pointer = 0
        self.resume()
                
    # def make_use_order(self):
    #     self.use_order = self.use_order[0] + self.use_order[0]
    #     self.load_order = [x[0] for x in self.use_order if x[1] == "load"]
    
    def get_computable_form(self, tensor_id, non_blocking=True):
        if self.tensor_list[tensor_id][1] is None:
            tensor = self.tensor_list[tensor_id][0]
            if tensor != None:
                main_stream = torch.cuda.current_stream()
                main_stream.synchronize()
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
                            if self.metadata[tensor_id]['type'] == "activation":
                                with torch.enable_grad():
                                    tensor = tensor.to(torch.cuda.current_device(), non_blocking=non_blocking)
                            else:
                                tensor = tensor.to(torch.cuda.current_device(), non_blocking=non_blocking)
                    if type(tensor) == NF4Tensor:
                        torch.cuda.nvtx.range_push(f"dequantize {tensor_id}")
                        tensor = tensor.get_original_weight().to(tensor.dtype, non_blocking=non_blocking)
                        torch.cuda.nvtx.range_pop()
                    if hasattr(tensor, "tensor_id"):
                        del tensor.tensor_id
                    
                    if non_blocking:
                        if self.metadata[tensor_id]["event"] == None:
                            self.metadata[tensor_id]["event"] = torch.cuda.Event(enable_timing=True)
                            self.metadata[tensor_id]["event"].record()
                    else:
                        self.stream.synchronize()
                
                    self.tensor_list[tensor_id][1] = tensor
                    self.num_tensor_in_gpu += 1
                # if not non_blocking:
                #     self.stream.synchronize()
                main_stream.synchronize()
                torch.cuda.nvtx.range_pop()

                # self.tensor_list[tensor_id][1] = tensor.detach()
            else:
                RuntimeError(f"Error, missing tensor id {tensor_id}")
        else:
            # if self.metadata[tensor_id]["event"] != None:
            #     self.metadata[tensor_id]["event"].synchronize()
            #     self.metadata[tensor_id]["event"] = None
            pass
            
            # if not self.warmup:
            #     with torch.cuda.stream(self.stream):
                    
            #         for i in range(4):
            #             with torch.cuda.stream(self.streams[i]):
            #                 self.get_computable_form(self.get_next_tensor_id(), True)
            #         for stream in self.streams:
            #             stream.synchronize()
        # return self.tensor_list[tensor_id][1]
    def total_moving_time(self):
        return self.tensor_bucket.total_move_tensors_time()
    # def get_next_tensor_id(self):
    #     self.load_pointer = (self.load_pointer + 1) % len(self.load_order)
    #     return self.load_order[self.load_pointer]
            
    # def decide_offload_tensor(self, tensor_id, pointer, num_tensors_in_gpu) -> bool:
    #     if self.pause_flag or num_tensors_in_gpu > self.max_tensor_num_in_gpu:
    #         return True

    #     # 預先計算每個張量的最後使用位置
    #     if not hasattr(self, 'last_use_index'):
    #         self.last_use_index = {}
    #         for i in range(len(self.use_order)-1, -1, -1):
    #             tid, _ = self.use_order[i]
    #             if tid not in self.last_use_index:
    #                 self.last_use_index[tid] = i

    #     tmp_num_in_gpu = num_tensors_in_gpu
    #     offloaded_tensors = set()

    #     for i in range(pointer, len(self.use_order)):
    #         current_tensor_id, action = self.use_order[i]
    #         if current_tensor_id == tensor_id:
    #             # 如果張量即將被使用，不應該卸載
    #             return False
    #         if action == "load":
    #             tmp_num_in_gpu += 1
    #         elif action == "offload":
    #             if current_tensor_id in offloaded_tensors:
    #                 continue
    #             # 如果張量不再被使用，可以卸載
    #             if self.last_use_index.get(current_tensor_id, -1) <= i:
    #                 tmp_num_in_gpu -= 1
    #                 offloaded_tensors.add(current_tensor_id)
    #         # 如果GPU中的張量數超過限制，需要卸載
    #         if tmp_num_in_gpu > self.max_tensor_num_in_gpu:
    #             return True
    #     return True
                        
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
    
