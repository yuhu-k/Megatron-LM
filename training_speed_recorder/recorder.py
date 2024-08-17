import time

class recorder:
    def __init__(self):
        self.count = 0
        self.start_time = None
        self.end_time = None
        
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