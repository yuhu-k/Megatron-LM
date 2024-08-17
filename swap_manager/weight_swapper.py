from torch import Tensor
import threading
import torch



class WeightSwapper:
    def __init__(self, warmup_iteration_times = 1) -> None:
        self.weight_dict = []
        self.device_dict = []
        self.gpu_weight_dict = []
        self.weight_id_count = 0
        self.warmup = True
        self.use_order = []
        self.order_list = []
        self.tmp_list = []
        self.warmup_iteration_times = warmup_iteration_times
        
        # self.swap_thread = []
        # self.active_threads = 0
        # self.swap_queue = []
        # self.lock_queue = threading.Lock()
        # self.lock_weight_dict = threading.Lock()
        # self.lock_threads = threading.Lock()
        # self.lock_order_list = threading.Lock()
        
    def register(self, weight:Tensor = None, device=None):
        weight_id = self.weight_id_count
        self.weight_id_count += 1
        self.weight_dict.append(weight)
        self.device_dict.append(device)
        self.gpu_weight_dict.append(None)
        # self.swap_thread.append(None)
        
        return weight_id
    
    def update_weight(self, weight:Tensor, weight_id):
        if self.weight_id_count < weight_id:
            RuntimeError(f"Error, missing weight id {weight_id}")
            
        if not isinstance(weight, Tensor):
            RuntimeError(f"Weight needs to be a tensor, but get {weight}")
        
        if self.weight_id_count <= weight_id or self.device_dict[weight_id] == None:
            self.device_dict[weight_id] = weight.device
        
        self.weight_dict[weight_id] = weight.cpu().detach()
        
    def get_weight(self, weight_id):
        if self.warmup:
            self.use_order.append((weight_id, "get"))
        
        if self.gpu_weight_dict[weight_id] is None:
            weight = self.weight_dict[weight_id]
            if weight != None:
                # if self.warmup:
                #     self.gpu_weight_dict[weight_id] = weight.to(self.device_dict[weight_id], non_blocking=False).detach()
                # else:
                    #self.add_swap_thread(self.order_list[1])
                    # self.add_swap_thread(weight_id)
                    # self.lock_threads.acquire()
                    # thread = self.swap_thread[weight_id]
                    # self.lock_threads.release()
                    # if thread != None:
                    #     print(weight_id, self.gpu_weight_dict[weight_id], self.swap_queue)
                    #     thread.join()
                        
                #self.lock_weight_dict.acquire()
                self.gpu_weight_dict[weight_id] = weight.to(self.device_dict[weight_id], non_blocking=False).detach()
                #self.lock_weight_dict.release()
            else:
                RuntimeError(f"Error, missing weight id {weight_id}")
        return self.gpu_weight_dict[weight_id]
            
    def offload_weight(self, weight_id):
        if self.warmup:
            self.use_order.append((weight_id, "offload"))
            del self.gpu_weight_dict[weight_id]
            self.gpu_weight_dict.insert(weight_id, None)
        else:
            if self.decide_offload_weight(weight_id):
                self.gpu_weight_dict[weight_id] = None
                torch.cuda.empty_cache()
            
    def get_weight_cpu(self, weight_id):
        weight = self.weight_dict[weight_id]
        if weight != None:
            return weight
        else:
            RuntimeError(f"Error, missing weight id {weight_id}")
            
    def dump_user_order(self):
        with open("/tmp2/tests.txt", "w") as f:
            i = 1
            for l in self.order_list:
                f.write(f"{i}: " + str(l) + "\n")
                i += 1
            
    def finish_warmup(self):
        if self.warmup:
            self.warmup_iteration_times -= 1
            self.build_weight_order()
            if self.warmup_iteration_times <= 0:
                self.warmup = False
                self.dump_user_order()
        
    def build_weight_order(self):
        tmp_order_list = []
        for id, act in self.use_order:
            if act == "offload":
                tmp_order_list.append(id)
        self.use_order = []
        self.order_list.append(tmp_order_list)
                        
    def decide_offload_weight(self, weight_id) -> bool:
        offload = True
        for l in self.tmp_list:
            order_list = l
            len = 0#int(order_list.__len__() * 1/2)
            half_list = order_list[:len]
            if weight_id in half_list:
                offload = False
                break
        self.update_tmp_list()
        return offload
        
    def update_tmp_list(self):
        for id in range(self.tmp_list.__len__()):
            self.tmp_list[id].append(self.tmp_list[id].pop(0))
    
    def reset_tmp_list(self):
        if not self.warmup:
            self.tmp_list = self.order_list.copy()
            self.update_tmp_list()
        
    def get_in_gpu_weight_ratio(self):
        count = 0
        for w in self.gpu_weight_dict:
            if w is not None:
                count += 1
                
        return count / self.gpu_weight_dict.__len__()
    
    # def add_swap_thread(self, weight_id):
    #     def swap_to_gpu(weight_id):
    #         self.lock_queue.acquire()
    #         queue = self.swap_queue
    #         self.lock_queue.release()
    #         for x in queue:
    #             if x == weight_id:
    #                 break
    #             self.lock_threads.acquire()
    #             thread = self.swap_thread[x]
    #             self.lock_threads.release()
    #             thread.join()
            
    #         self.swapping_now = weight_id
            
    #         self.lock_weight_dict.acquire()
    #         weight = self.weight_dict[weight_id]
    #         self.gpu_weight_dict[weight_id] = weight.to(self.device_dict[weight_id], non_blocking=False).detach()
    #         self.lock_weight_dict.release()
        
    #     self.lock_weight_dict.acquire()
    #     weight = self.gpu_weight_dict[weight_id]
    #     self.lock_weight_dict.release()
    #     if weight is None:
    #         thread = threading.Thread(target=swap_to_gpu, args=(weight_id,))
    #         self.lock_threads.acquire()
    #         self.swap_thread[weight_id] = thread
    #         self.lock_threads.release()
    #         self.lock_queue.acquire()
    #         self.swap_queue.append(weight_id)
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
        
    # def cut_in_swap_queue(self, weight_id):
    #     self.swap_queue.insert(0, weight_id)