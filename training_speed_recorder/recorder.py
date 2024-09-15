import time

class recorder:
    def __init__(self):
        self.count = 0
        self.start_time = None
        self.end_time = None
        self.leng = 0
        
    def start(self):
        if self.start_time == None:
            self.start_time = time.time()
        
    def end(self):
        self.end_time = time.time()
        
    def add_tokens_count(self, tokens_num):
        if self.start_time != None and self.end_time == None:
            self.count += tokens_num
            
    def get_speed(self):
        # if self.start_time != None and self.end_time != None:
        #     print(f"Training speed: {self.count/(self.end_time - self.start_time)} tokens/s")
        return self.count/(self.end_time - self.start_time)
    
class nf4_expand_speed_recorder:
    def __init__(self):
        self.max_speed = 0
        self.min_speed = None
        self.acc_speed = 0
        self.count = 0
    
    def start_record(self):
        self.start_time = time.time()
    
    def end_record(self, tensor_element_num):
        end_time = time.time()
        expand_time = end_time - self.start_time
        
        speed = expand_time / tensor_element_num
        self.max_speed = speed if speed > self.max_speed else self.max_speed
        self.min_speed = speed if self.min_speed == None or speed < self.min_speed else self.min_speed
        self.acc_speed += speed
        self.count += 1
    
    def get_speed(self):
        return {
            "average": self.acc_speed / self.count,
            "maximum": self.max_speed,
            "minimum": self.min_speed
        }
    