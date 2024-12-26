import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


q = torch.load('q.pt')
k = torch.load('k.pt')
v = torch.load('v.pt')
mask = torch.load('mask.pt')
print(f'q.shape : {q.shape}, q.dtype: {q.dtype}, q.max:{q.max()}, q.min:{q.min()}')
print(f'k.shape : {k.shape}, k.dtype: {k.dtype}, k.max:{k.max()}, k.min:{k.min()}')
print(f'v.shape : {v.shape}, v.dtype: {v.dtype}, v.max:{v.max()}, v.min:{v.min()}')
print(f'mask.shape: {mask.shape}, mask.dtype: {mask.dtype}')

print('using default sdpa backend, which in this case would be EFFICIENT_ATTENTION')
attn_output = F.scaled_dot_product_attention(q, k, v, mask)
print(attn_output)
attn_output.backward(torch.ones_like(attn_output))
print(f'q.grad nan : {q.grad.isnan().any()}')
print(f'k.grad nan : {k.grad.isnan().any()}')
print(f'v.grad nan : {v.grad.isnan().any()}')


q.grad.zero_()
k.grad.zero_()
v.grad.zero_()

with sdpa_kernel(SDPBackend.MATH):
    attn_output = F.scaled_dot_product_attention(q, k, v, mask)

print(attn_output)
attn_output.backward(torch.ones_like(attn_output))
print('using MATH backend')
print(f'q.grad nan : {q.grad.isnan().any()}')
print(f'k.grad nan : {k.grad.isnan().any()}')
print(f'v.grad nan : {v.grad.isnan().any()}')
