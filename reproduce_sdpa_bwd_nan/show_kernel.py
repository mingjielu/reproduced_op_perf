import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.profiler import profile, record_function, ProfilerActivity


q = torch.load('q.pt')
k = torch.load('k.pt')
v = torch.load('v.pt')
mask = torch.load('mask.pt')
print(f'q.shape : {q.shape}, q.dtype: {q.dtype}')
print(f'k.shape : {k.shape}, k.dtype: {k.dtype}')
print(f'v.shape : {v.shape}, v.dtype: {v.dtype}')

print('using default sdpa backend, which in this case would be EFFICIENT_ATTENTION')
with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
    with record_function("sdpa_forward"):
        attn_output = F.scaled_dot_product_attention(q, k, v, mask)
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))


attn_output.backward(torch.ones_like(attn_output))
print(f'q.grad nan : {q.grad.isnan().any()}')
print(f'k.grad nan : {k.grad.isnan().any()}')
print(f'v.grad nan : {v.grad.isnan().any()}')


q.grad.zero_()
k.grad.zero_()
v.grad.zero_()

with sdpa_kernel(SDPBackend.MATH):
    attn_output = F.scaled_dot_product_attention(q, k, v, mask)

attn_output.backward(torch.ones_like(attn_output))
print('using MATH backend')
print(f'q.grad nan : {q.grad.isnan().any()}')
print(f'k.grad nan : {k.grad.isnan().any()}')
print(f'v.grad nan : {v.grad.isnan().any()}')
