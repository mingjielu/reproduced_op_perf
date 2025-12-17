import os
import torch
from contextlib import nullcontext
from transformer_engine.pytorch.attention.dot_product_attention.backends import FusedAttention
from torch.profiler import profile, record_function, ProfilerActivity

os.environ['NVTE_CK_USES_BWD_V3'] = '1'

softmax_scale=0.08838834764831843
attention_type='self'
layer_number=1
deterministic=False
attn_kwargs = {'attention_dropout': 0.0,
              'attention_dropout_ctx': nullcontext}
seqlen = 32768
device = torch.device("cuda")
query_layer = torch.randn((seqlen,2,192), dtype = torch.bfloat16, device = device, requires_grad = True)
key_layer = torch.randn((seqlen,2,192), dtype = torch.bfloat16, device = device, requires_grad = True)
value_layer = torch.randn((seqlen,2,128), dtype = torch.bfloat16, device = device, requires_grad = True)
cu_seqlens_q = cu_seqlens_kv = torch.tensor([0,  3602,  5672,  9207, 11229, 14814, 15286, 18924, 19417, 20504, 24106, 26087, 29675, 32768], dtype=torch.int32).to(device)
attention_mask = torch.triu(torch.ones(seqlen, seqlen), diagonal=1).bool().unsqueeze(0).unsqueeze(0).to(device)
# attention_mask = torch.load('attention_mask.pt').to(device)
max_seqlen = 3638

fused_attention = FusedAttention(
    softmax_scale,
    attention_type=attention_type,
    layer_number=layer_number,
    deterministic=deterministic,
    **attn_kwargs,
)

profiler = torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    schedule=torch.profiler.schedule(
        wait=5,
        warmup=0,
        active=10,
        repeat=1
    ),
    on_trace_ready=torch.profiler.tensorboard_trace_handler("./profile"),
    record_shapes=True,
    #profile_memory=True,
    with_stack=True
)



with profiler:
    for i in range(15):
        out = fused_attention(
            query_layer,
            key_layer,
            value_layer,
            qkv_layout='thd_thd_thd',
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
            cu_seqlens_q_padded=None,
            cu_seqlens_kv_padded=None,
            max_seqlen_q=max_seqlen,
            max_seqlen_kv=max_seqlen,
            attn_mask_type='padding_causal',
            attention_mask=attention_mask,
            window_size=(-1,0),
            fused_attention_backend=1, #tex.NVTE_Fused_Attn_Backend.NVTE_CK
            core_attention_bias_type='no_bias',
            core_attention_bias=None,
            fast_zero_fill=True,
            cp_group=None,
            cp_global_ranks=None,
            cp_stream=None,
            cp_comm_type='p2p',
            fp8=False,
            fp8_meta={},
            quantizers={},
            pad_between_seqs=False,
            inference_params=None,
        )
        out.sum().backward()
        q_grad = query_layer.grad.clone()
        query_layer.grad = None
        profiler.step()
print(profiler.key_averages().table(sort_by="cuda_time_total",row_limit=10))


# modify value

profiler2 = torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    schedule=torch.profiler.schedule(
        wait=5,
        warmup=0,
        active=10,
        repeat=1
    ),
    on_trace_ready=torch.profiler.tensorboard_trace_handler("./profile"),
    record_shapes=True,
    #profile_memory=True,
    with_stack=True
)



with profiler2:
    for i in range(15):
        # Pad value_layer from 128 to 192 dimensions
        original_head_dim = value_layer.shape[-1]
        padded_head_dim = 192
        pad_part = torch.zeros((value_layer.shape[0],value_layer.shape[1], padded_head_dim - original_head_dim)).to(value_layer.dtype).to(value_layer.device)
        pad_part = pad_part.detach()   # ~X~N确梯度~M~A~O~G~Y~G~L
        value_layer_padded = torch.cat([value_layer, pad_part], dim=-1)


        # Compute with padded tensors
        out_padded = fused_attention(
            query_layer,
            key_layer,
            value_layer_padded,
            qkv_layout='thd_thd_thd',
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
            cu_seqlens_q_padded=None,
            cu_seqlens_kv_padded=None,
            max_seqlen_q=max_seqlen,
            max_seqlen_kv=max_seqlen,
            attn_mask_type='padding_causal',
            attention_mask=attention_mask,
            window_size=(-1,0),
            fused_attention_backend=1, #tex.NVTE_Fused_Attn_Backend.NVTE_CK
            core_attention_bias_type='no_bias',
            core_attention_bias=None,
            fast_zero_fill=True,
            cp_group=None,
            cp_global_ranks=None,
            cp_stream=None,
            cp_comm_type='p2p',
            fp8=False,
            fp8_meta={},
            quantizers={},
            pad_between_seqs=False,
            inference_params=None,
        )
        # Reshape from (32768, 384) to (32768, 2, 192), take first 128 dims, reshape back

        out_padded = out_padded.reshape(value_layer.shape[0],value_layer.shape[1], -1)
        out_padded = out_padded[:,:, :original_head_dim]
        out_padded = out_padded.reshape(out_padded.shape[0],-1)
        out_padded.sum().backward()
        q_grad_padded = query_layer.grad.clone()
        query_layer.grad = None
        profiler2.step()
print(profiler2.key_averages().table(sort_by="cuda_time_total",row_limit=10))





# Calculate differences
abs_diff = torch.abs(out - out_padded)
max_diff = abs_diff.max().item()
mean_diff = abs_diff.mean().item()
relative_diff = (abs_diff / (torch.abs(out) + 1e-8)).mean().item()

print("=" * 50)
print("Comparison between out and out_padded:")
print(f"  Max absolute difference: {max_diff}")
print(f"  Mean absolute difference: {mean_diff}")
print(f"  Mean relative difference: {relative_diff}")
print(f"  out shape: {out.shape}")
print(f"  out_padded shape: {out_padded.shape}")
print("=" * 50)


# Calculate q_grad differences
abs_diff = torch.abs(q_grad - q_grad_padded)
max_diff = abs_diff.max().item()
mean_diff = abs_diff.mean().item()
relative_diff = (abs_diff / (torch.abs(q_grad) + 1e-8)).mean().item()

print("=" * 50)
print("Comparison between q_grad and q_grad_padded:")
print(f"  Max absolute difference: {max_diff}")
print(f"  Mean absolute difference: {mean_diff}")
print(f"  Mean relative difference: {relative_diff}")
print(f"  q_grad shape: {q_grad.shape}")
print(f"  q_grad_padded shape: {q_grad_padded.shape}")
print("=" * 50)



'''
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                                          ProfilerStep*         0.00%       0.000us         0.00%       0.000us       0.000us      93.158ms       100.08%      93.158ms       9.316ms            10
autograd::engine::evaluate_function: FusedAttnFuncBa...         0.11%     108.201us         2.36%       2.265ms     226.527us       0.000us         0.00%      75.095ms       7.510ms            10
                                  FusedAttnFuncBackward         1.32%       1.272ms         2.24%       2.157ms     215.706us      74.547ms        80.08%      75.095ms       7.510ms            10
_ZN7ck_tile6kentryILi1ENS_19FmhaBwdDQDKDVKernelINS_2...         0.00%       0.000us         0.00%       0.000us       0.000us      66.794ms        71.75%      66.794ms       6.679ms            10
                                          ProfilerStep*         6.40%       6.155ms         8.68%       8.352ms     835.151us       0.000us         0.00%      17.477ms       1.748ms            10
                                          FusedAttnFunc         0.96%     926.044us         1.38%       1.327ms     132.657us      16.984ms        18.24%      16.984ms       1.698ms            10
_ZN7ck_tile6kentryILi2ENS_13FmhaFwdKernelINS_28Block...         0.00%       0.000us         0.00%       0.000us       0.000us      16.537ms        17.76%      16.537ms       1.654ms            10
_ZN7ck_tile6kentryILi2ENS_22FmhaBwdOGradDotOKernelIN...         0.00%       0.000us         0.00%       0.000us       0.000us       6.514ms         7.00%       6.514ms     651.448us            10
_ZN7ck_tile6kentryILi2ENS_25FmhaBwdConvertQGradKerne...         0.00%       0.000us         0.00%       0.000us       0.000us     574.531us         0.62%     574.531us      57.453us            10
autograd::engine::evaluate_function: torch::autograd...         0.06%      53.389us         0.33%     317.074us      10.569us       0.000us         0.00%     516.195us      17.206us            30
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
Self CPU time total: 96.173ms
Self CUDA time total: 93.088ms

-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                                          ProfilerStep*         0.00%       0.000us         0.00%       0.000us       0.000us      54.927ms       103.06%      54.927ms       5.493ms            10
autograd::engine::evaluate_function: FusedAttnFuncBa...         0.19%     117.126us         3.13%       1.957ms     195.680us       0.000us         0.00%      31.813ms       3.181ms            10
                                  FusedAttnFuncBackward         2.03%       1.267ms         2.95%       1.840ms     183.968us      31.314ms        58.76%      31.813ms       3.181ms            10
aiter::fmha_bwd_hd192_bf16_causal_a32_rtna_psskddv_g...         0.00%       0.000us         0.00%       0.000us       0.000us      26.174ms        49.11%      26.174ms       2.617ms            10
                                          ProfilerStep*        13.34%       8.328ms        87.14%      54.413ms       5.441ms       0.000us         0.00%      19.781ms       1.978ms            10
                                          FusedAttnFunc         1.41%     877.550us         1.96%       1.221ms     122.083us      15.719ms        29.49%      15.719ms       1.572ms            10
_ZN7ck_tile6kentryILi1ENS_13FmhaFwdKernelINS_28Block...         0.00%       0.000us         0.00%       0.000us       0.000us      15.218ms        28.56%      15.218ms       1.522ms            10
_ZN7ck_tile6kentryILi2ENS_22FmhaBwdOGradDotOKernelIN...         0.00%       0.000us         0.00%       0.000us       0.000us       3.900ms         7.32%       3.900ms     389.980us            10
                                            aten::copy_         2.32%       1.452ms        60.04%      37.493ms     535.612us       3.714ms         6.97%       3.714ms      53.050us            70
                                               aten::to         0.05%      31.380us        59.56%      37.194ms       1.860ms       0.000us         0.00%       2.741ms     137.045us            20
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
Self CPU time total: 62.445ms
Self CUDA time total: 53.294ms

==================================================
Comparison between out and out_padded:
  Max absolute difference: 0.0
  Mean absolute difference: 0.0
  Mean relative difference: 0.0
  out shape: torch.Size([32768, 256])
  out_padded shape: torch.Size([32768, 256])
==================================================
==================================================
Comparison between q_grad and q_grad_padded:
  Max absolute difference: 0.001953125
  Mean absolute difference: 6.705522537231445e-08
  Mean relative difference: 2.4557113647460938e-05
  q_grad shape: torch.Size([32768, 2, 192])
  q_grad_padded shape: torch.Size([32768, 2, 192])
==================================================
'''
