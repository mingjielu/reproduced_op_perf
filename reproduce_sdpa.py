import torch
import torch.nn.functional as F
import time
import numpy as np 
torch.manual_seed(0)
torch.backends.cuda.preferred_rocm_fa_library("ck")
configs = [
        #((2,32,36480,64),torch.float16),
        ((1, 32, 36480, 72),torch.float16),
        ((1,16,2048,64),torch.float16),
        ((1,24,2048,64),torch.float16),
        ((2,24,2048,64),torch.float16),
        ((1,24,4250,64),torch.float16),
        ((2,24,4250,64),torch.float16),
        ] 
 
def fwd(q, k, v, attn_mask):
    times = []
    torch.cuda.synchronize()
    forward_start = time.time()
    output = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
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
    for _ in range(cnt):
        out, fwd_time = fwd(q, k, v, attn_mask)
        bwd_time = bwd(out)
        fwd_times.append(fwd_time)
        bwd_times.append(bwd_time)
    return ('fwd ', np.round(np.mean(fwd_times),4), 'bwd ' ,np.round(np.mean(bwd_times),4))

if __name__ == '__main__':
    for config in configs:
        b,n,e,d = config[0]
        q = torch.randn(b,n,e,d, dtype=config[1], device='cuda', requires_grad=True)
        k = torch.randn(b,n,e,d, dtype=config[1], device='cuda', requires_grad=True)
        v = torch.randn(b,n,e,d, dtype=config[1], device='cuda', requires_grad=True)
        print(config, call(q, k, v, None,3, 5))
