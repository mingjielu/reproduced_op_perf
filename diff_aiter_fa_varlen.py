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

'''

-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
autograd::engine::evaluate_function: FlashAttnVarlen...         0.15%      11.717us         8.10%     641.685us     641.685us       0.000us         0.00%       6.646ms       6.646ms             1
                            FlashAttnVarlenFuncBackward         2.46%     194.821us         7.96%     629.968us     629.968us       0.000us         0.00%       6.646ms       6.646ms             1
                flash_attn::_flash_attn_varlen_backward         3.03%     239.837us         4.90%     387.587us     387.587us       6.565ms        80.02%       6.646ms       6.646ms             1
void ck_tile::kentry<256, 1, ck_tile::FmhaBwdDQDKDVK...         0.00%       0.000us         0.00%       0.000us       0.000us       6.449ms        78.61%       6.449ms       6.449ms             1
                                    FlashAttnVarlenFunc         1.63%     128.732us         5.38%     425.674us     425.674us       0.000us         0.00%       1.340ms       1.340ms             1
                 flash_attn::_flash_attn_varlen_forward         2.18%     172.236us         3.54%     280.588us     280.588us       1.340ms        16.33%       1.340ms       1.340ms             1
void ck_tile::kentry<256, 2, ck_tile::FmhaFwdKernel<...         0.00%       0.000us         0.00%       0.000us       0.000us       1.340ms        16.33%       1.340ms       1.340ms             1
                                              aten::sum         0.95%      74.912us         1.24%      97.835us      97.835us      94.299us         1.15%      94.299us      94.299us             1
autograd::engine::evaluate_function: torch::autograd...         0.08%       6.258us         0.82%      65.306us      21.769us       0.000us         0.00%      93.176us      31.059us             3
                        torch::autograd::AccumulateGrad         0.28%      21.871us         0.75%      59.048us      19.683us       0.000us         0.00%      93.176us      31.059us             3
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
Self CPU time total: 7.917ms
Self CUDA time total: 8.204ms

-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
autograd::engine::evaluate_function: FlashAttnVarlen...         0.12%       9.103us         8.10%     628.896us     628.896us       0.000us         0.00%       5.654ms       5.654ms             1
                            FlashAttnVarlenFuncBackward         2.16%     168.021us         7.98%     619.793us     619.793us       0.000us         0.00%       5.654ms       5.654ms             1
                          aiter::wrapper_mha_varlen_bwd         1.77%     137.795us         4.74%     367.797us     367.797us       5.566ms        70.98%       5.591ms       5.591ms             1
_ZN7ck_tile6kentryILi1ENS_19FmhaBwdDQDKDVKernelINS_2...         0.00%       0.000us         0.00%       0.000us       0.000us       5.445ms        69.44%       5.445ms       5.445ms             1
                                    FlashAttnVarlenFunc         1.67%     129.843us         3.30%     255.901us     255.901us       0.000us         0.00%       1.972ms       1.972ms             1
                      aiter::wrapper_fmha_v3_varlen_fwd         1.11%      86.418us         1.39%     107.681us     107.681us       1.972ms        25.14%       1.972ms       1.972ms             1
           aiter::fmha_fwd_hd128_bf16_causal_rtna_group         0.00%       0.000us         0.00%       0.000us       0.000us       1.972ms        25.14%       1.972ms       1.972ms             1
autograd::engine::evaluate_function: torch::autograd...         0.07%       5.157us         0.50%      38.957us      12.986us       0.000us         0.00%      94.017us      31.339us             3
                        torch::autograd::AccumulateGrad         0.11%       8.883us         0.44%      33.800us      11.267us       0.000us         0.00%      94.017us      31.339us             3
                                             aten::add_         0.16%      12.138us         0.32%      24.917us       8.306us      94.017us         1.20%      94.017us      31.339us             3
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
Self CPU time total: 7.764ms
Self CUDA time total: 7.842ms

q_grad/q_grad2 is nan:  tensor(False, device='cuda:0') tensor(False, device='cuda:0')
diff output, mean/max:  2.7179718017578125e-05 0.015625
diff grad,  mean/max:  0.04150390625 3.25


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

