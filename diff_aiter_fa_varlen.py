import torch
from torch.profiler import profile, record_function, ProfilerActivity
from flash_attn import flash_attn_varlen_func
from aiter import flash_attn_varlen_func as flash_attn_varlen_func_aiter

device = torch.device("cuda:0")
q = torch.randn((6666, 16, 128), dtype = torch.bfloat16, device = device, requires_grad = True)
k = torch.randn((6666, 16, 128), dtype = torch.bfloat16, device = device, requires_grad = True)
v = torch.randn((6666, 16, 128), dtype = torch.bfloat16, device = device, requires_grad = True)
cu_q = torch.tensor([0, 3298, 6666], dtype=torch.int32).to(device)
cu_k = torch.tensor([0, 3298, 6666], dtype=torch.int32).to(device)
causal = True
softmax_scale = 0.08838834764831845

for i in range(5):
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
autograd::engine::evaluate_function: FlashAttnVarlen...         0.29%      23.426us         9.68%     769.486us     769.486us       0.000us         0.00%       6.742ms       6.742ms             1
                            FlashAttnVarlenFuncBackward         2.20%     174.545us         9.39%     746.060us     746.060us       0.000us         0.00%       6.742ms       6.742ms             1
                flash_attn::_flash_attn_varlen_backward         4.99%     396.759us         6.67%     530.165us     530.165us       6.692ms        81.07%       6.742ms       6.742ms             1
void ck_tile::kentry<256, 1, ck_tile::FmhaBwdDQDKDVK...         0.00%       0.000us         0.00%       0.000us       0.000us       6.568ms        79.58%       6.568ms       6.568ms             1
                                    FlashAttnVarlenFunc         1.02%      80.735us         3.96%     314.447us     314.447us       0.000us         0.00%       1.390ms       1.390ms             1
                 flash_attn::_flash_attn_varlen_forward         1.97%     156.668us         2.85%     226.272us     226.272us       1.390ms        16.83%       1.390ms       1.390ms             1
void ck_tile::kentry<256, 2, ck_tile::FmhaFwdKernel<...         0.00%       0.000us         0.00%       0.000us       0.000us       1.390ms        16.83%       1.390ms       1.390ms             1
void ck_tile::kentry<64, 2, ck_tile::FmhaBwdOGradDot...         0.00%       0.000us         0.00%       0.000us       0.000us      94.560us         1.15%      94.560us      94.560us             1
autograd::engine::evaluate_function: torch::autograd...         0.25%      20.216us         1.85%     147.071us      49.024us       0.000us         0.00%      61.719us      20.573us             3
                        torch::autograd::AccumulateGrad         0.95%      75.607us         1.60%     126.855us      42.285us       0.000us         0.00%      61.719us      20.573us             3
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
Self CPU time total: 7.949ms
Self CUDA time total: 8.254ms

-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
autograd::engine::evaluate_function: FlashAttnVarlen...         0.47%      21.059us        16.33%     737.635us     737.635us       0.000us         0.00%       2.614ms       2.614ms             1
                            FlashAttnVarlenFuncBackward         5.72%     258.371us        15.87%     716.576us     716.576us       0.000us         0.00%       2.614ms       2.614ms             1
                      aiter::wrapper_fmha_v3_varlen_bwd         2.40%     108.288us         7.65%     345.356us     345.356us       2.559ms        54.34%       2.583ms       2.583ms             1
aiter::fmha_bwd_hd128_bf16_causal_br_a32_rtna_pssk_g...         0.00%       0.000us         0.00%       0.000us       0.000us       2.443ms        51.87%       2.443ms       2.443ms             1
                                    FlashAttnVarlenFunc         2.05%      92.630us         3.98%     179.785us     179.785us       0.000us         0.00%       1.978ms       1.978ms             1
                      aiter::wrapper_fmha_v3_varlen_fwd         1.05%      47.242us         1.81%      81.796us      81.796us       1.978ms        42.00%       1.978ms       1.978ms             1
           aiter::fmha_fwd_hd128_bf16_causal_rtna_group         0.00%       0.000us         0.00%       0.000us       0.000us       1.978ms        42.00%       1.978ms       1.978ms             1
_ZN7ck_tile6kentryILi2ENS_22FmhaBwdOGradDotOKernelIN...         0.00%       0.000us         0.00%       0.000us       0.000us      88.960us         1.89%      88.960us      88.960us             1
autograd::engine::evaluate_function: torch::autograd...         0.42%      18.947us         1.93%      87.343us      29.114us       0.000us         0.00%      61.118us      20.373us             3
                        torch::autograd::AccumulateGrad         0.56%      25.365us         1.51%      68.396us      22.799us       0.000us         0.00%      61.118us      20.373us             3
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
Self CPU time total: 4.516ms
Self CUDA time total: 4.710ms

q_grad/q_grad2 is nan:  tensor(False, device='cuda:0') tensor(False, device='cuda:0')
diff output, mean/max:  2.7179718017578125e-05 0.0078125
diff grad,  mean/max:  0.00011777877807617188 0.015625


######## h20 fa_varlen
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
autograd::engine::evaluate_function: FlashAttnVarlen...         0.16%       9.682us         6.31%     379.074us     379.074us       0.000us         0.00%       3.039ms       3.039ms             1
                            FlashAttnVarlenFuncBackward         1.64%      98.499us         6.15%     369.392us     369.392us       0.000us         0.00%       3.039ms       3.039ms             1
                flash_attn::_flash_attn_varlen_backward         2.86%     171.752us         4.14%     248.830us     248.830us       3.011ms        68.34%       3.039ms       3.039ms             1
void flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<Fl...         0.00%       0.000us         0.00%       0.000us       0.000us       2.933ms        66.58%       2.933ms       2.933ms             1
                                    FlashAttnVarlenFunc         1.46%      87.955us        36.10%       2.168ms       2.168ms       0.000us         0.00%       1.257ms       1.257ms             1
                 flash_attn::_flash_attn_varlen_forward         2.82%     169.324us        34.54%       2.074ms       2.074ms       1.257ms        28.53%       1.257ms       1.257ms             1
void flash_fwd_kernel<Flash_fwd_kernel_traits<128, 1...         0.00%       0.000us         0.00%       0.000us       0.000us       1.257ms        28.53%       1.257ms       1.257ms             1
autograd::engine::evaluate_function: torch::autograd...         0.13%       7.753us         0.83%      49.772us      16.591us       0.000us         0.00%      72.576us      24.192us             3
                        torch::autograd::AccumulateGrad         0.24%      14.216us         0.70%      42.019us      14.006us       0.000us         0.00%      72.576us      24.192us             3
                                             aten::add_         0.24%      14.407us         0.46%      27.803us       9.268us      72.576us         1.65%      72.576us      24.192us             3
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
Self CPU time total: 6.004ms
Self CUDA time total: 4.405ms
'''

