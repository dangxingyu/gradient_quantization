import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
import os
from utils import (
    butterfly_all_reduce,
    print_rank0
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from transformers.models.opt.modeling_opt import (
    OPTDecoderLayer,
)
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    FullStateDictConfig,
    StateDictType,
    BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import ( 
    transformer_auto_wrap_policy,
    enable_wrap,
    wrap,
)
from functools import partial

class ToyModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.in_proj = nn.Linear(input_size, hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, input_size, bias=False)
        # self.dummy_proj = nn.Linear(128, 128, bias=False)
    
    def forward(self, x):
        x = self.in_proj(x)
        x = F.relu(x)
        x = self.out_proj(x)
        return x

def setup():
    local_rank = int(os.environ['LOCAL_RANK'])
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
    )
    torch.cuda.set_device(local_rank)

def cleanup():
    dist.destroy_process_group()

def print_rank0(*args, **kwargs):
    rank = dist.get_rank()
    if rank == 0:
        print(*args, **kwargs)

def int4_compress_hook(communication_group, grad, *args, **kwargs):
    # print_rank0(f'before: ', grad)
    # assert torch.all(args[0] == 0)
    _grad = grad.detach().clone()
    butterfly_all_reduce(grad, communication_group)
    # print_rank0(f'after: ', grad)
    world_size = dist.get_world_size()
    _grad *= world_size
    error = torch.abs(_grad - grad).mean() / torch.abs(_grad).mean()
    print_rank0(f'error: ', error)
    return None

def main():
    setup()
    torch.manual_seed(0)
    rank = dist.get_rank()
    input_size = 128
    print(f"Rank {rank} is running")
    # model = ToyModel(input_size, input_size).cuda()
    print(f"Rank {rank} start loading model")
    model = AutoModelForCausalLM.from_pretrained('facebook/opt-2.7b').cuda()
    model.train()
    for param in model.parameters():
        param.requires_grad = True
        param = param.float()

    bfSixteen = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16
    )

    import functools
    opt_auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            OPTDecoderLayer
        },
    )
    sharding_strategy: ShardingStrategy = ShardingStrategy.FULL_SHARD
    model = FSDP(model,
        auto_wrap_policy=opt_auto_wrap_policy,
        # mixed_precision=bfSixteen,
        sharding_strategy=sharding_strategy,
        device_id=torch.cuda.current_device(),
    )
    print(f"Rank {rank} finish loading model")

    tokenizer = AutoTokenizer.from_pretrained('facebook/opt-2.7b')
    dummy_input = 'The quick brown fox jumps over the lazy dog.'
    inputs = tokenizer(dummy_input, return_tensors='pt')
    for k, v in inputs.items():
        inputs[k] = v.cuda()
    if 'labels' not in inputs:
        inputs['labels'] = inputs['input_ids'].clone()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    # criterion = nn.MSELoss()
    # x = torch.randn(1, input_size).cuda()
    # y = x * 10 + 1

    communication_group = dist.group.WORLD
    model.register_comm_hook(communication_group, int4_compress_hook)
    print(model, file=open('model.txt', 'w'))

    for i in range(2):
        loss = model(**inputs).loss
        loss.backward()
        # check_grad = model.module.model.decoder.layers[0].self_attn.k_proj.weight.grad
        # print_rank0(f"check_grad: {check_grad}")
        print_rank0(f"Iter {i} loss: {loss.item()}")
        optimizer.step()
        optimizer.zero_grad()


if __name__ == '__main__':
    main()
    cleanup()