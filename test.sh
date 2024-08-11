export PYTHONPATH="$PYTHONPATH:$(pwd)"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_DEBUG=INFO && export NCCL_SOCKET_IFNAME=enp181s0 && export GLOO_SOCKET_IFNAME=enp181s0

NSYS_ARGS="
    --force-overwrite true \
    -o /tmp/test \
    --gpu-metrics-device all \
    --capture-range cudaProfilerApi \
    --trace=nvtx,cuda,cudnn \
    --backtrace=none
"

nsys profile $NSYS_ARGS \
torchrun test.py