import torch
import time

configs = [
        {'tensor_size_a': (4096,1, 8192), 'dtype_a': torch.bfloat16},
        ]

for config in configs:
    a = torch.randn(config['tensor_size_a'], device='cuda', dtype=config['dtype_a'], requires_grad=True)
    a = a.contiguous()

    warmup = 10
    repeats = 50  # ~G~M~M次~U~L确~]计~W稳~Z

    for _ in range(warmup):
        y = torch.nn.functional.silu(a)
        y.backward(torch.ones_like(y))


    acc_forward = 0
    acc_backward = 0
    for _ in range(repeats):
        start = time.time()
        torch.cuda.synchronize()
        y = torch.nn.functional.silu(a)
        torch.cuda.synchronize()
        mid = time.time()
        y.backward(torch.ones_like(y))
        torch.cuda.synchronize()
        end = time.time()
        acc_forward += mid - start
        acc_backward += end - mid

    print(config, f"Averaged execution time for silu_forward: {acc_forward / repeats * 1000:.3f} ms")
    print(config, f"Averaged execution time for silu_backward: {acc_backward / repeats * 1000:.3f} ms")
