import torch
from torch.profiler import profile, record_function, ProfilerActivity
from flash_attn import flash_attn_varlen_func
from aiter import flash_attn_varlen_func as flash_attn_varlen_func_aiter

device = torch.device("cuda:0")
q = torch.randn((68496, 16, 72), dtype = torch.bfloat16, device = device, requires_grad = True)
k = torch.randn((68496, 16, 72), dtype = torch.bfloat16, device = device, requires_grad = True)
v = torch.randn((68496, 16, 72), dtype = torch.bfloat16, device = device, requires_grad = True)

cu_q = torch.tensor([0,  3928,  6111,  9866, 11515, 14854, 17020, 19084, 22679, 25888,
        29416, 32193, 35665, 39053, 42166, 45280, 49252, 53121, 56984, 60892, 63234, 66823, 68496], dtype=torch.int32).to(device)
cu_k = torch.tensor([0,  3928,  6111,  9866, 11515, 14854, 17020, 19084, 22679, 25888,
        29416, 32193, 35665, 39053, 42166, 45280, 49252, 53121, 56984, 60892, 63234, 66823, 68496], dtype=torch.int32).to(device)


causal = True
softmax_scale = 0.08838834764831845

for i in range(5): #warmup
    out = flash_attn_varlen_func(q, k, v, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k, max_seqlen_q=4096, max_seqlen_k=4096, softmax_scale=softmax_scale, causal=True)
    out.sum().backward()

    out2, _ = flash_attn_varlen_func_aiter(q, k, v, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k, max_seqlen_q=4096, max_seqlen_k=4096, softmax_scale=softmax_scale, causal=True, return_lse=True)
    out2.sum().backward()


with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_stack=True) as prof:
    out = flash_attn_varlen_func(q, k, v, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k, max_seqlen_q=4096, max_seqlen_k=4096, softmax_scale=softmax_scale, causal=True)
    out.sum().backward()
    q_grad = q.grad.clone()
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_stack=True) as prof:
    out2, _ = flash_attn_varlen_func_aiter(q, k, v, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k, max_seqlen_q=4096, max_seqlen_k=4096, softmax_scale=softmax_scale, causal=True, return_lse=True)
    out2.sum().backward()
    q_grad2 = q.grad.clone()

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

print('q_grad/q_grad2 is nan: ', q_grad.isnan().any(),q_grad2.isnan().any())
print('diff output, mean/max: ', (out2-out).abs().mean().item(),(out2-out).abs().max().item())
print('diff grad,  mean/max: ', (q_grad2-q_grad).abs().mean().item(),(q_grad2-q_grad).abs().max().item())
