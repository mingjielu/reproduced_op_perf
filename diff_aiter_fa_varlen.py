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
autograd::engine::evaluate_function: FlashAttnVarlen...         0.27%      24.484us         8.28%     748.040us     748.040us       0.000us         0.00%       7.301ms       7.301ms             1
                            FlashAttnVarlenFuncBackward         1.88%     169.978us         8.00%     723.556us     723.556us       0.000us         0.00%       7.301ms       7.301ms             1
                flash_attn::_flash_attn_varlen_backward         4.19%     379.014us         5.68%     513.395us     513.395us       7.218ms        77.03%       7.301ms       7.301ms             1
void ck_tile::kentry<256, 1, ck_tile::FmhaBwdDQDKDVK...         0.00%       0.000us         0.00%       0.000us       0.000us       7.100ms        75.78%       7.100ms       7.100ms             1
                                    FlashAttnVarlenFunc         0.81%      72.970us         2.97%     268.402us     268.402us       0.000us         0.00%       1.619ms       1.619ms             1
                 flash_attn::_flash_attn_varlen_forward         1.47%     132.711us         2.10%     189.921us     189.921us       1.619ms        17.28%       1.619ms       1.619ms             1
void ck_tile::kentry<256, 2, ck_tile::FmhaFwdKernel<...         0.00%       0.000us         0.00%       0.000us       0.000us       1.619ms        17.28%       1.619ms       1.619ms             1
                                              aten::sum         0.42%      37.633us         0.61%      55.167us      55.167us     358.560us         3.83%     358.560us     358.560us             1
void at::native::reduce_kernel<512, 1, at::native::R...         0.00%       0.000us         0.00%       0.000us       0.000us     350.121us         3.74%     350.121us     350.121us             1
void ck_tile::kentry<64, 2, ck_tile::FmhaBwdOGradDot...         0.00%       0.000us         0.00%       0.000us       0.000us      89.119us         0.95%      89.119us      89.119us             1
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
Self CPU time total: 9.039ms
Self CUDA time total: 9.370ms

-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
autograd::engine::evaluate_function: FlashAttnVarlen...         0.35%      20.274us        12.94%     740.965us     740.965us       0.000us         0.00%       2.959ms       2.959ms             1
                            FlashAttnVarlenFuncBackward         4.17%     238.605us        12.58%     720.691us     720.691us       0.000us         0.00%       2.959ms       2.959ms             1
                      aiter::wrapper_fmha_v3_varlen_bwd         1.86%     106.448us         6.68%     382.666us     382.666us       2.872ms        48.57%       2.895ms       2.895ms             1
aiter::fmha_bwd_hd128_bf16_causal_br_a32_rtna_pssk_g...         0.00%       0.000us         0.00%       0.000us       0.000us       2.752ms        46.56%       2.752ms       2.752ms             1
                                    FlashAttnVarlenFunc         1.61%      92.015us         3.17%     181.337us     181.337us       0.000us         0.00%       2.642ms       2.642ms             1
                      aiter::wrapper_fmha_v3_varlen_fwd         0.85%      48.606us         1.47%      84.115us      84.115us       2.642ms        44.69%       2.642ms       2.642ms             1
           aiter::fmha_fwd_hd128_bf16_causal_rtna_group         0.00%       0.000us         0.00%       0.000us       0.000us       2.642ms        44.69%       2.642ms       2.642ms             1
                                            aten::clone         0.15%       8.626us         0.93%      53.412us      26.706us       0.000us         0.00%     202.119us     101.060us             2
                                            aten::copy_         0.31%      17.704us         0.56%      31.917us      15.958us     202.119us         3.42%     202.119us     101.060us             2
                         Memcpy DtoD (Device -> Device)         0.00%       0.000us         0.00%       0.000us       0.000us     143.880us         2.43%     143.880us     143.880us             1
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
Self CPU time total: 5.728ms
Self CUDA time total: 5.912ms

q_grad/q_grad2 is nan:  tensor(False, device='cuda:0') tensor(False, device='cuda:0')
diff output, mean/max:  2.7179718017578125e-05 0.015625
diff grad,  mean/max:  0.000118255615234375 0.015625


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

