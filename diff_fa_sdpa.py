import torch
import time
import flash_attn
from flash_attn import flash_attn_func

# 检查是否支持 Flash Attention（需要 A100 / H100 或 MI300 系列）
assert torch.cuda.is_available(), "CUDA is required for benchmarking"
device = torch.device("cuda")

print(f"flash_attn using version: {flash_attn.__version__}")
print(f"torch using version: {torch.__version__}")

is_causal = False

def benchmark(func, args, repeat=10, warmup=3, backward=False):
    """Benchmark function with warmup and averaging."""
    # Warmup
    for _ in range(warmup):
        output = func(*args)
        if backward:
            output.sum().backward()
    
    # Timing
    torch.cuda.synchronize()
    start_time = time.time()
    
    for _ in range(repeat):
        output = func(*args)
        if backward:
            output.sum().backward()
    
    torch.cuda.synchronize()
    avg_time = (time.time() - start_time) / repeat
    return avg_time

def test_attention(batch_size, num_heads, seq_len, head_dim):
    """Test SDPA vs FlashAttention for given input shapes."""
    print(f"Testing: B={batch_size}, H={num_heads}, S={seq_len}, D={head_dim}")
    
    # Random inputs
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.bfloat16, requires_grad=True)

    # --- Torch SDPA (B, H, S, D) ---
    def torch_sdpa(q, k, v):
        return torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=is_causal)

    # --- FlashAttention (B, S, H, D), 需要 permute ---
    def flash_attn(q, k, v):
        return flash_attn_func(q.permute(0, 2, 1, 3),  # B, S, H, D
                               k.permute(0, 2, 1, 3),
                               v.permute(0, 2, 1, 3),
                               dropout_p=0.0, causal=is_causal).permute(0, 2, 1, 3)  # 转回 B, H, S, D

    # 测试 Forward
    torch_sdpa_fwd_time = benchmark(torch_sdpa, (q, k, v), backward=False)
    flash_attn_fwd_time = benchmark(flash_attn, (q, k, v), backward=False)

    # 测试 Backward
    q.grad, k.grad, v.grad = None, None, None
    torch_sdpa_bwd_time = benchmark(torch_sdpa, (q, k, v), backward=True)
    
    q.grad, k.grad, v.grad = None, None, None
    flash_attn_bwd_time = benchmark(flash_attn, (q, k, v), backward=True)

    # 输出结果
    print(f"  Torch SDPA: Forward {torch_sdpa_fwd_time*1e3:.3f} ms | Backward {torch_sdpa_bwd_time*1e3:.3f} ms")
    print(f"  FlashAttn:  Forward {flash_attn_fwd_time*1e3:.3f} ms | Backward {flash_attn_bwd_time*1e3:.3f} ms")
    print("-" * 50)

# 测试不同参数
test_cases = [
    (1, 32, 36480, 72),
    #(32, 8, 128, 64),   # 小 batch
    #(32, 8, 512, 64),   # 中等长度
    #(16, 16, 1024, 128), # 大 batch
    #(192, 16, 577, 64), # clip
    #(40,18,632,128), # dit_pytorch
    #(2,18,8840,128), # dit_pytorch
    #(2,24,4250,64), # sd3
    #(16, 32, 1024, 72), # dim 72
    #(1, 32, 36480, 72)

]

for case in test_cases:
    test_attention(*case)

