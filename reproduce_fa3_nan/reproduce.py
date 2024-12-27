from flash_attn.modules.mha import FlashSelfAttention
import torch
qkv = torch.load('qkv.tensor')
#print(qkv.isnan().any(),qkv.max())
print(qkv.isnan().any())
print(qkv.shape)
attn = FlashSelfAttention(attention_dropout=0.0)
out = attn(qkv, causal=True)
loss = out.sum()
loss.backward()
print('bwd is nan:',qkv.grad.isnan().any())
