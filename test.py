

    
#!/usr/bin/env python
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def run(rank, size):
    req = None
    if rank == 0:
        tensor = torch.randn(1024,1024,1024).to("cuda:0")
        # Send the tensor to process 1
        req = dist.isend(tensor=tensor, dst=1)
    else:
        tensor = torch.zeros(1024,1024,1024).to("cuda:1")
        # Receive tensor from process 0
        req = dist.irecv(tensor=tensor, src=0)
        
    req.wait()
    print('Rank ', rank, ' has data ', tensor[0])

def init_process(rank, size, fn, backend='nccl'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


if __name__ == "__main__":
    torch.cuda.cudart().cudaProfilerStart()
    torch.autograd.profiler.emit_nvtx(record_shapes=True).__enter__()
    size = 2
    processes = []
    mp.set_start_method("spawn")
    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
    torch.cuda.cudart().cudaProfilerStop()