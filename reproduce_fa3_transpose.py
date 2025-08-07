import torch
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity
from flash_attn import flash_attn_func
b, s, h, d = 2, 8192, 32, 128
q = torch.randn(b,s,h,d, dtype=torch.bfloat16, device='cuda', requires_grad=True)
k = torch.randn(b,s,h,d, dtype=torch.bfloat16, device='cuda', requires_grad=True)
v = torch.randn(b,s,h,d, dtype=torch.bfloat16, device='cuda', requires_grad=True)
print('q: ',q.shape,q.stride())

def hook_fn(grad):
    print('*'*100)
    print(f'{grad.shape=},{grad.stride()=},{grad.is_contiguous()=}')
for i in range(5): #warmup
    a = flash_attn_func(q,k,v,causal=True)
    a.sum().backward()

#with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_stack=True) as prof:
#    aa = flash_attn_func(q,k,v,causal=True)
#    aa.retain_grad()
#    aa.register_hook(hook_fn)
#    bb = aa.transpose(0,1)
#    bb.retain_grad()
#    bb.register_hook(hook_fn)
#    #bb.backward(torch.ones((s,b,h,d)).to('cuda'))
#    bb.backward(torch.ones_like(bb).contiguous())
#print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
#
#
#with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_stack=True) as prof:
#    aa = flash_attn_func(q,k,v,causal=True)
#    aa.retain_grad()
#    aa.register_hook(hook_fn)
#    bb = aa.transpose(0,1).contiguous()
#    bb.retain_grad()
#    bb.register_hook(hook_fn)
#    bb.backward(torch.ones_like(bb).contiguous())
#print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))


my_linear = nn.Linear(4096,4096,dtype=torch.bfloat16).to('cuda')
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_stack=True) as prof:
    aa = flash_attn_func(q,k,v,causal=True)
    aa.retain_grad()
    aa.register_hook(hook_fn)
    bb = aa.transpose(0,1).contiguous()
    #bb.retain_grad()
    #bb.register_hook(hook_fn)
    cc = bb.reshape(s,b,-1)
    dd = my_linear(cc)
    dd.sum().backward()

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
