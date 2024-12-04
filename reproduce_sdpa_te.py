import torch
import torch.nn.functional as F
from transformer_engine.pytorch.attention import DotProductAttention
from transformer_engine.pytorch.attention import AttnMaskType
import time
import numpy as np 
torch.manual_seed(0)
#torch.backends.cuda.preferred_rocm_fa_library("ck")
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
    core_attention = DotProductAttention(q.shape[2],k.shape[-1])
    print(vars(core_attention))
    torch.cuda.synchronize()
    forward_start = time.time()
    output = core_attention(q,k,v,'causal')
    #print(q.shape,k.shape,v.shape,output.shape)
    q = q.permute(1,2,0,3)
    k = k.permute(1,2,0,3)
    v = v.permute(1,2,0,3)
    output2 = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
    output2 = output2.permute(2,0,1,3).view(output.shape[0],output.shape[1],output.shape[2])
    print((torch.abs(output2-output).mean()))
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
        q = torch.randn(e,b,n,d, dtype=config[1], device='cuda', requires_grad=True)
        k = torch.randn(e,b,n,d, dtype=config[1], device='cuda', requires_grad=True)
        v = torch.randn(e,b,n,d, dtype=config[1], device='cuda', requires_grad=True)
        print(config, call(q, k, v, None,3, 5))
