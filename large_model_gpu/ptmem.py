class PTMem:
    def __init__(self):
        self.total_mem = 0
        self.tensor_cache = {}

    def reset(self):
        self.total_mem = 0
        for key in self.tensor_cache:
            del key
        self.tensor_cache = {}
        

ptmem = PTMem()
