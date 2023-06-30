import torch
import os
import torch.distributed as dist
import gc

QUANTIZE_GROUP_SIZE=128

def quantize(w, n_bits=4, group_size=128):
    if group_size is not None:
        w_shape = w.shape
        w = w.reshape(-1, group_size)
    w_max = torch.max(torch.abs(w), dim=-1).values
    # print(w_max.shape)
    w_min = -w_max
    q_max = 2 ** (n_bits - 1) - 1
    q_min = -q_max - 1
    scale = (w_max - w_min) / (q_max - q_min)
    scale = scale.reshape(-1, 1)
    q_w = torch.round(w / scale)
    q_w = torch.clamp(q_w, q_min, q_max)
    if group_size is not None:
        w = w.reshape(w_shape)
        q_w = q_w.reshape(w_shape)
    q_w = q_w.to(dtype=torch.int8)
    return q_w, scale

def dequantize(w, scale, group_size=128):
    w_shape = w.shape
    w = w.reshape(-1, group_size).float()
    w = w * scale
    w = w.reshape(w_shape)
    return w

# transfer quantized tensors into compact bit tensors
def serialize(w, n_bits=4):
    w_shape = w.shape
    # add (2**(n_bits-1)) to make sure the value is positive
    w = w + (2**(n_bits-1))
    w = w.to(dtype=torch.uint8)
    pack_ratio = 8 // n_bits
    packed = torch.zeros(w.numel() // pack_ratio, dtype=torch.uint8, device=w.device)
    for i in range(pack_ratio):
        packed |= w[i::pack_ratio].long() << (n_bits * i)
    packed = packed.to(dtype=torch.uint8)
    return packed, w_shape

def deserialize(packed, w_shape, n_bits=4):
    pack_ratio = 8 // n_bits
    w = torch.zeros(packed.numel() * pack_ratio, dtype=torch.uint8, device=packed.device)
    for i in range(pack_ratio):
        w[i::pack_ratio] = (packed >> (n_bits * i)) & ((1 << n_bits) - 1)
    w = w.to(dtype=torch.float32).reshape(w_shape)
    w = w - (2**(n_bits-1))
    return w

def butterfly_all_reduce(grad, comm_group, num_bits=4):
    # all reduce the gradients with butterfly pattern

    rank = dist.get_rank(group=comm_group)
    world_size = dist.get_world_size(group=comm_group)

    q_grad, scale = quantize(grad, n_bits=num_bits, group_size=QUANTIZE_GROUP_SIZE)
    q_grad, w_shape = serialize(q_grad, n_bits=num_bits)

    q_w_tag = 0
    scale_tag = 1

    reqs = []
    q_grad_send_buffer = q_grad
    q_grad_recv_buffer = torch.zeros_like(q_grad, dtype=q_grad.dtype)
    scale_send_buffer = scale
    scale_recv_buffer = torch.zeros_like(scale, dtype=scale.dtype)
    
    # world size must be power of 2
    assert world_size & (world_size - 1) == 0
    import math
    power = int(math.log2(world_size))
    for i in range(power):
        target = rank ^ (1 << i)

        if rank < target:
            reqs.append(dist.isend(tensor=q_grad_send_buffer, dst=target, tag=q_w_tag))
            reqs.append(dist.irecv(tensor=q_grad_recv_buffer, src=target, tag=q_w_tag))
            reqs.append(dist.isend(tensor=scale_send_buffer, dst=target, tag=scale_tag))
            reqs.append(dist.irecv(tensor=scale_recv_buffer, src=target, tag=scale_tag))
        else:
            reqs.append(dist.irecv(tensor=q_grad_recv_buffer, src=target, tag=q_w_tag))
            reqs.append(dist.isend(tensor=q_grad_send_buffer, dst=target, tag=q_w_tag))
            reqs.append(dist.irecv(tensor=scale_recv_buffer, src=target, tag=scale_tag))
            reqs.append(dist.isend(tensor=scale_send_buffer, dst=target, tag=scale_tag))
        
        for req in reqs:
            req.wait()
        reqs = []

        recv_q_w = deserialize(q_grad_recv_buffer, w_shape, n_bits=num_bits)
        recv_scale = scale_recv_buffer
        recv_w = dequantize(recv_q_w, recv_scale)
        grad.add_(recv_w)
        q_grad, scale = quantize(grad, n_bits=num_bits, group_size=QUANTIZE_GROUP_SIZE)
        q_grad, w_shape = serialize(q_grad, n_bits=num_bits)
        q_grad_send_buffer = q_grad
        scale_send_buffer = scale
        
    return grad

def print_rank(*args):
    print(f'rank {dist.get_rank()}:', *args)

def print_rank0(*args):
    if dist.get_rank() == 0:
        print(f'rank {dist.get_rank()}:', *args)

def test():
    import time
    # test quantize and dequantize
    w = torch.randn(128, 128)
    q_w, scale = quantize(w, n_bits=4, group_size=128)
    w_recover = dequantize(q_w, scale, group_size=128)
    print_rank0(torch.mean(torch.abs(w - w_recover)))

    # test butterfly_all_reduce
    comm_group = dist.group.WORLD
    print_rank0(comm_group)
    local_rank = dist.get_rank(group=comm_group)
    grad = torch.ones(42800, 52800) * (local_rank + 1)
    print(f'gradient size: {grad.numel() * 32 / 8 / 1024 / 1024} MB')
    grad = grad.to(local_rank)
    _grad = grad.clone()
    dist.barrier(group=comm_group)
    torch.cuda.synchronize()
    start = time.time()
    my_grad = butterfly_all_reduce(grad, comm_group)
    end = time.time()
    print_rank0(f'time: {end - start}')
    dist.barrier(group=comm_group)
    torch.cuda.synchronize()
    start = time.time()
    dist.all_reduce(_grad, group=comm_group)
    end = time.time()
    print_rank0(f'time: {end - start}')
    dist.barrier(group=comm_group)
    torch.cuda.synchronize()
    print_rank0(grad, my_grad, _grad)

if __name__ == '__main__':
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
    )
    test()