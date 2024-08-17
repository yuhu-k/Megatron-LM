from .recorder import recorder


_RECORDER = None

def init_recorder():
    global _RECORDER
    _RECORDER = recorder()
    
def get_recorder():
    return _RECORDER