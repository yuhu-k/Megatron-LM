from .recorder import recorder, nf4_expand_speed_recorder


_RECORDER = None
_NF4_RECORDER = None

def init_recorder():
    global _RECORDER
    _RECORDER = recorder()

def init_nf4_recorder():
    global _NF4_RECORDER
    _NF4_RECORDER = nf4_expand_speed_recorder()
    
def get_recorder():
    return _RECORDER

def get_nf4_recorder():
    global _NF4_RECORDER
    if _NF4_RECORDER == None:
        init_nf4_recorder()
    return _NF4_RECORDER