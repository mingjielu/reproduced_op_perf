import torch
from torch import Tensor
import transformer_engine as te

class TEDotProductAttention(te.pytorch.DotProductAttention):
    cp_stream: torch.cuda.Stream = None
    
    def __init__(self):
        TEDotProductAttention.cp_stream = torch.cuda.Stream()

        extra_kwargs = {}
        extra_kwargs['num_gqa_groups'] = 4
        extra_kwargs['attention_type'] = 'self'
        extra_kwargs["cp_stream"] = TEDotProductAttention.cp_stream

        super().__init__(
            num_attention_heads=40,
            kv_channels=128,
            **extra_kwargs
        )


    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attn_mask_type
    ):

        core_attn_out = super().forward(
            query,
            key,
            value,
            None,
            attn_mask_type=attn_mask_type.name
            #attn_mask_type=None
        )


        return core_attn_out

q = torch.load('q.pt')#.requires_grad_(True)
k = torch.load('k.pt')#.requires_grad_(True)
v = torch.load('v.pt')#.requires_grad_(True)
print(q,k,v)
attn_mask_type = torch.load('attn_mask_type.pt')
q = q.cpu().cuda()
k = k.cpu().cuda()
v = v.cpu().cuda()
q=q.requires_grad_(True)
k=k.requires_grad_(True)
v=v.requires_grad_(True)
attention_layer = TEDotProductAttention()

attention_output = attention_layer(
    query=q,
    key=k,
    value=v,
    attn_mask_type=attn_mask_type
)

attention_output.backward(torch.ones_like(attention_output))
print(q.grad.isnan().any())

print("Attention output shape:", attention_output.shape)
