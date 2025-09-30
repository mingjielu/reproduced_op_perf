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
    q.grad = None

    out2, _ = flash_attn_varlen_func_aiter(q, k, v, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k, max_seqlen_q=4096, max_seqlen_k=4096, softmax_scale=softmax_scale, causal=True, return_lse=True)
    out2.sum().backward()
    q.grad = None


with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_stack=True) as prof:
    out = flash_attn_varlen_func(q, k, v, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k, max_seqlen_q=4096, max_seqlen_k=4096, softmax_scale=softmax_scale, causal=True)
    out.sum().backward()
    q_grad = q.grad.clone()
    q.grad = None
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_stack=True) as prof:
    out2, _ = flash_attn_varlen_func_aiter(q, k, v, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k, max_seqlen_q=4096, max_seqlen_k=4096, softmax_scale=softmax_scale, causal=True, return_lse=True)
    out2.sum().backward()
    q_grad2 = q.grad.clone()
    q.grad = None
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

print('q_grad/q_grad2 is nan: ', q_grad.isnan().any(),q_grad2.isnan().any())
print('diff output, mean/max: ', (out2-out).abs().mean().item(),(out2-out).abs().max().item())
print('diff grad,  mean/max: ', (q_grad2-q_grad).abs().mean().item(),(q_grad2-q_grad).abs().max().item())

'''
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
autograd::engine::evaluate_function: FlashAttnVarlen...         0.00%      15.141us         0.09%     576.444us     576.444us       0.000us         0.00%     110.681ms     110.681ms             1
                            FlashAttnVarlenFuncBackward         0.02%     142.468us         0.09%     561.303us     561.303us       0.000us         0.00%     110.681ms     110.681ms             1
                flash_attn::_flash_attn_varlen_backward         0.04%     267.994us         0.06%     384.262us     384.262us     110.451ms        89.44%     110.681ms     110.681ms             1
void ck_tile::kentry<256, 1, ck_tile::FmhaBwdDQDKDVK...         0.00%       0.000us         0.00%       0.000us       0.000us     106.862ms        86.53%     106.862ms     106.862ms             1
                                    FlashAttnVarlenFunc         0.02%     109.943us         0.06%     381.231us     381.231us       0.000us         0.00%      12.296ms      12.296ms             1
                 flash_attn::_flash_attn_varlen_forward         0.03%     168.633us         0.04%     259.962us     259.962us      12.296ms         9.96%      12.296ms      12.296ms             1
void ck_tile::kentry<256, 2, ck_tile::FmhaFwdKernel<...         0.00%       0.000us         0.00%       0.000us       0.000us      12.296ms         9.96%      12.296ms      12.296ms             1
void ck_tile::kentry<64, 2, ck_tile::FmhaBwdOGradDot...         0.00%       0.000us         0.00%       0.000us       0.000us       3.055ms         2.47%       3.055ms       3.055ms             1
void ck_tile::kentry<256, 2, ck_tile::FmhaBwdConvert...         0.00%       0.000us         0.00%       0.000us       0.000us     533.723us         0.43%     533.723us     533.723us             1
autograd::engine::evaluate_function: torch::autograd...         0.00%      11.902us         0.01%      65.223us      21.741us       0.000us         0.00%     282.519us      94.173us             3
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
Self CPU time total: 646.944ms
Self CUDA time total: 123.493ms

-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
autograd::engine::evaluate_function: FlashAttnVarlen...         0.07%      23.571us         2.90%     959.529us     959.529us       0.000us         0.00%      24.535ms      24.535ms             1
                            FlashAttnVarlenFuncBackward         1.06%     349.383us         2.83%     935.958us     935.958us       0.000us         0.00%      24.535ms      24.535ms             1
                      aiter::wrapper_fmha_v3_varlen_bwd         0.38%     126.054us         1.29%     427.629us     427.629us      24.296ms        73.07%      24.414ms      24.414ms             1
aiter::fmha_bwd_hd128_bf16_causal_br_a32_rtna_psskdd...         0.00%       0.000us         0.00%       0.000us       0.000us      20.775ms        62.48%      20.775ms      20.775ms             1
                                    FlashAttnVarlenFunc         0.40%     132.688us         0.92%     304.869us     304.869us       0.000us         0.00%       8.199ms       8.199ms             1
                          aiter::wrapper_mha_varlen_fwd         0.29%      97.605us         0.49%     160.639us     160.639us       8.199ms        24.66%       8.199ms       8.199ms             1
_ZN7ck_tile6kentryILi2ENS_13FmhaFwdKernelINS_28Block...         0.00%       0.000us         0.00%       0.000us       0.000us       8.199ms        24.66%       8.199ms       8.199ms             1
_ZN7ck_tile6kentryILi2ENS_22FmhaBwdOGradDotOKernelIN...         0.00%       0.000us         0.00%       0.000us       0.000us       3.031ms         9.12%       3.031ms       3.031ms             1
_ZN7ck_tile6kentryILi2ENS_25FmhaBwdConvertQGradKerne...         0.00%       0.000us         0.00%       0.000us       0.000us     489.842us         1.47%     489.842us     489.842us             1
autograd::engine::evaluate_function: torch::autograd...         0.06%      20.631us         0.28%      94.282us      31.427us       0.000us         0.00%     279.919us      93.306us             3
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
Self CPU time total: 33.094ms
Self CUDA time total: 33.249ms

q_grad/q_grad2 is nan:  tensor(False, device='cuda:0') tensor(False, device='cuda:0')
diff output, mean/max:  0.0002193450927734375 0.03125
diff grad,  mean/max:  8.20159912109375e-05 0.03125
'''
