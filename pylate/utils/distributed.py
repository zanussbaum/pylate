import torch
import torch.distributed.nn
import torch.distributed as dist


def print_in_order(msg):
    # helpful function for debugging distributed training
    if dist.is_initialized():
        for i in range(dist.get_world_size()):
            if i == dist.get_rank():
                print(f"{i}: {msg}")
            dist.barrier()
    else:
        print(msg)


def print_rank_zero(msg):
    # helpful function for debugging distributed training
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(msg)
        dist.barrier()
    else:
        print(msg)


def gather_with_grad(t):
    # gather with gradients: https://github.com/mlfoundations/open_clip/issues/616#issuecomment-1996291468
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return t

    if t.ndim == 0:
        t = t.unsqueeze(0)

    return torch.cat(torch.distributed.nn.all_gather(t), dim=0)
