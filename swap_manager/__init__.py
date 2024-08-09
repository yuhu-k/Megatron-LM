from .weight_swapper import WeightSwapper

_WEIGHTSWAPPER = None


def init_weight_swapper():
    global _WEIGHTSWAPPER
    _WEIGHTSWAPPER = WeightSwapper()

def get_weight_swapper():
    return _WEIGHTSWAPPER