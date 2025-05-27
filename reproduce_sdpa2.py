import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

configs = [
        #((2,32,36480,64),torch.float16),
        #((1,16,2048,64),torch.float16), # benchmark
        #((1,24,2048,64),torch.float16), # benchmark
        #((2,24,2048,64),torch.float16), # benchmark
        #((1,24,4250,64),torch.float16), # sd3 F.sdpa
        ((2,24,4250,64),torch.float16), # sd3 F.sdpa
        ((1,24,4096,128),torch.float16), # benchmark
        ((32,40,1024,128),torch.bfloat16), # qwen-14b fa
        ((7,32,2048,128),torch.bfloat16), # llama7b te-fa
        ((2,32,2048,128),torch.bfloat16), # llama7b te-fa
        ((1,24,4096,128),torch.bfloat16), # moe-18b fa
        ((192,16,577,64),torch.bfloat16), # clip F.sdpa
        ((40,18,632,128),torch.bfloat16), # dit-pytorch F.sdpa
        ((2,18,8840,128),torch.bfloat16), # dit-pytorch F.sdpa
        ] 


for config in configs:
    print(config)
    b,n,e,d = config[0]
    q = torch.randn(b,n,e,d, dtype=config[1], device='cuda', requires_grad=True)
    k = torch.randn(b,n,e,d, dtype=config[1], device='cuda', requires_grad=True)
    v = torch.randn(b,n,e,d, dtype=config[1], device='cuda', requires_grad=True)

    attn_output = F.scaled_dot_product_attention(q, k, v)
    attn_output.backward(torch.ones_like(attn_output))
    print(f'q.grad nan : {q.grad.isnan().any()}, {q.grad.mean().item()=}, {q.grad.std().item()=}')
    #print(f'k.grad nan : {k.grad.isnan().any()}, {k.grad.mean().item()=}, {k.grad.std().item()=}')
    #print(f'v.grad nan : {v.grad.isnan().any()}, {v.grad.mean().item()=}, {v.grad.std().item()=}')
    
    
    q.grad.zero_()
    k.grad.zero_()
    v.grad.zero_()
    
    with sdpa_kernel(SDPBackend.MATH):
        attn_output = F.scaled_dot_product_attention(q, k, v)
    
    attn_output.backward(torch.ones_like(attn_output))
    print('using MATH backend')
    print(f'q.grad nan : {q.grad.isnan().any()}, {q.grad.mean().item()=}, {q.grad.std().item()=}')
    #print(f'k.grad nan : {k.grad.isnan().any()}, {k.grad.mean().item()=}, {k.grad.std().item()=}')
    #print(f'v.grad nan : {v.grad.isnan().any()}, {v.grad.mean().item()=}, {v.grad.std().item()=}')
    print('\n')
