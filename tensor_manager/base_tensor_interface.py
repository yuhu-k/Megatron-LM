class TensorInterfaceBase:
    def __init__(self, numel, size, device, dtype):
        self.__numel = numel
        self.__size = size
        self.device = device
        self.dtype = dtype
        
    @classmethod
    def compress(cls, tensor):
        raise NotImplementedError
    
    def decompress(self):
        raise NotImplementedError

    def get_computable_form(self):
        return self.decompress()
    
    def numel(self):
        return self.__numel
    
    def size(self):
        return self.__size
    
    def to(cls, *args, **kwargs):
        raise NotImplementedError
    
    def pin_memory(self):
        raise NotImplementedError
    
