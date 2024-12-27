import torch
import torch.nn.functional as F
from flash_attn import flash_attn_func
import time
import numpy as np 
from torch.profiler import profile, record_function, ProfilerActivity
torch.manual_seed(0)
configs = [
        #((2,32,36480,64),torch.float16),
        #((1, 32, 36480, 72),torch.float16),
        #((1,16,2048,64),torch.float16),
        #((1,24,2048,64),torch.float16),
        #((2,24,2048,64),torch.float16),
        #((1,24,4250,64),torch.float16),
        #((2,24,4250,64),torch.float16),
        #((7,32,2048,128),torch.bfloat16),
        ((42,25,1024,64),torch.bfloat16),
        ] 
 
def fwd(q, k, v, attn_mask):
    times = []
    torch.cuda.synchronize()
    forward_start = time.time()
    output = flash_attn_func(q,k,v,causal=True)
    #q = q.permute(0,2,1,3)
    #k = k.permute(0,2,1,3)
    #v = v.permute(0,2,1,3)
    #output2 = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
    #output2 = output2.permute(0,2,1,3)
    #print(torch.abs(output2-output).mean())
    torch.cuda.synchronize()
    forward_end = time.time()
    return output, forward_end - forward_start
 
def bwd(output):
    loss = output.sum()
    torch.cuda.synchronize()
    backward_start = time.time()
    loss.backward()
    torch.cuda.synchronize()
    backward_end = time.time()
    return backward_end - backward_start
 
def call(q, k, v, attn_mask, warmup, cnt):
    for _ in range(warmup):
        out, fwd_time = fwd(q, k, v, attn_mask)
        bwd_time = bwd(out)
    fwd_times = []
    bwd_times = []
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_stack=True) as prof:
    #if True:
        for _ in range(cnt):
            out, fwd_time = fwd(q, k, v, attn_mask)
            bwd_time = bwd(out)
            fwd_times.append(fwd_time)
            bwd_times.append(bwd_time)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    #print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10,max_name_column_width=None))
    return ('fwd ', float(np.round(np.mean(fwd_times),4)), 'bwd ' ,float(np.round(np.mean(bwd_times),4)))

if __name__ == '__main__':
    for config in configs:
        b,n,e,d = config[0]
        q = torch.randn(b,e,n,d, dtype=config[1], device='cuda', requires_grad=True)
        k = torch.randn(b,e,n,d, dtype=config[1], device='cuda', requires_grad=True)
        v = torch.randn(b,e,n,d, dtype=config[1], device='cuda', requires_grad=True)
        print(config, call(q, k, v, None,3, 5))
