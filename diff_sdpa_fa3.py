import torch
import torch.nn.functional as F
from flash_attn import flash_attn_func
import time
import numpy as np 
import copy
from torch.profiler import profile, record_function, ProfilerActivity
#torch.manual_seed(10)
configs = [
        #((2,32,36480,64),torch.float16),
        #((1, 32, 36480, 72),torch.float16), # dit-megatron F.sdpa
        #((1,16,2048,64),torch.float16), # benchmark
        #((1,24,2048,64),torch.float16), # benchmark
        #((2,24,2048,64),torch.float16), # benchmark
        #((1,24,4250,64),torch.float16), # sd3 F.sdpa
        #((2,24,4250,64),torch.float16), # sd3 F.sdpa
        #((1,24,4096,128),torch.float16), # benchmark
        #((32,40,1024,128),torch.bfloat16), # qwen-14b fa
        #((7,32,2048,128),torch.bfloat16), # llama7b te-fa
        #((2,32,2048,128),torch.bfloat16), # llama7b te-fa
        #((1,24,4096,128),torch.bfloat16), # moe-18b fa
        #((192,16,577,64),torch.bfloat16), # clip F.sdpa
        ((40,18,632,128),torch.bfloat16), # dit-pytorch F.sdpa
        ((2,18,8840,128),torch.bfloat16), # dit-pytorch F.sdpa
        ((64,28,1024,128),torch.bfloat16), # qwen-57b fa
        ] 
 
def fwd_bwd(q, k, v, attn_mask):
    times = []
    torch.cuda.synchronize()
    forward_start = time.time()
    output = flash_attn_func(q,k,v,causal=True)
    torch.cuda.synchronize()
    forward_end = time.time()
    #q2=q.clone().detach().requires_grad_(True)
    #k2=k.clone().detach().requires_grad_(True)
    #v2=v.clone().detach().requires_grad_(True)
    q2=copy.deepcopy(q)
    k2=copy.deepcopy(k)
    v2=copy.deepcopy(v)
    q22 = q2.permute(0,2,1,3)
    k22 = k2.permute(0,2,1,3)
    v22 = v2.permute(0,2,1,3)
    output2 = F.scaled_dot_product_attention(q22, k22, v22, attn_mask=attn_mask,is_causal=True)
    output2 = output2.permute(0,2,1,3)
    print('fwd ',torch.abs(output2-output).mean().detach(),torch.abs(output2-output).max().detach())
 
    loss = output.sum()
    torch.cuda.synchronize()
    backward_start = time.time()
    loss.backward()
    torch.cuda.synchronize()
    loss2=output2.sum()
    loss2.backward()
    backward_end = time.time()
    #from IPython import embed;embed()
    print('bwd ',torch.abs(q.grad-q2.grad).mean(),torch.abs(q.grad-q2.grad).max())
    return forward_end - forward_start, backward_end - backward_start
 
def call(q, k, v, attn_mask, warmup, cnt):
    for _ in range(warmup):
        fwd_time,bwd_time = fwd_bwd(q, k, v, attn_mask)
    fwd_times = []
    bwd_times = []
    if True:
        for _ in range(cnt):
            fwd_time,bwd_time = fwd_bwd(q, k, v, attn_mask)
            fwd_times.append(fwd_time)
            bwd_times.append(bwd_time)
    return ('fwd ', float(np.round(np.mean(fwd_times),4)), 'bwd ' ,float(np.round(np.mean(bwd_times),4)))

if __name__ == '__main__':
    for config in configs:
        b,n,e,d = config[0]
        q = torch.randn(b,e,n,d, dtype=config[1], device='cuda', requires_grad=True)
        k = torch.randn(b,e,n,d, dtype=config[1], device='cuda', requires_grad=True)
        v = torch.randn(b,e,n,d, dtype=config[1], device='cuda', requires_grad=True)
        print(config, call(q, k, v, None,3, 2))