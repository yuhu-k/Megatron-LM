import torch
import os
import argparse

from .swap_manager import SwapManager
from .utils import human_readable_size, generate_report, redirect_print, count_parameters
from .policy import init_policy
from .policy import add_arguments as policy_add_arguments
from .multi_gpu import init_multi, IpcObject
from .multi_gpu import add_arguments as multi_gpu_add_arguments
from .context import context

from collections import defaultdict, namedtuple
from .profiler import Kernel, TimeSegment, ProfilerInformation
from .packed_tensor import PackHookInitializer

_GLOBAL_HOOK = None

def add_arguments(parser):
    policy_add_arguments(parser)
    multi_gpu_add_arguments(parser)

    # Profiler Args
    parser.add_argument('--profile', default=False, action=argparse.BooleanOptionalAction, help='Profile and Export Trace')
    parser.add_argument('--path', type=str, default='/root/share')
    parser.add_argument('--filename', type=str)

def init(device, args, enable_multi_gpu=False, ipc_object=None, skip_sync_iterations=-1):
    print("===================")
    print(args)
    print("===================")
    context.device = device
    context.offload_stream = torch.cuda.Stream(device)
    context.prefetch_stream = torch.cuda.Stream(device)
    context.default_stream = torch.cuda.default_stream(device)
    
    if enable_multi_gpu:
        context.enable_multi_gpu = enable_multi_gpu
        init_multi(args, ipc_object, skip_sync_iterations)

    context.total_steps = args.lms_iterations
    context.enable_swap = args.lms_swap   
    if context.enable_swap:
        init_policy(args)

    context.profile = args.lms_profile
    if context.profile:
        if args.lms_path and args.lms_filename:
            context.path = args.lms_path
            context.filename = args.lms_filename
        else:
            raise Exception("The profile export path and filename is required")

    redirect_print()
    generate_report("Initialize Large Model Support", {
        #"CUDA_VISIBLE_DEVICES": os.environ['CUDA_VISIBLE_DEVICES'],
        "device": device,
    })

def init_pack_hook(is_profile, is_enable, nonblocking):
    global _GLOBAL_HOOK
    _GLOBAL_HOOK = PackHookInitializer(is_profile, is_enable, nonblocking)
    
def get_pack_hook():
    return _GLOBAL_HOOK