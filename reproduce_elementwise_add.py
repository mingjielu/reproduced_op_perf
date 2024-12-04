import time
import torch
torch.manual_seed(0)
configs = [
        {'tensor_size_a': (2,4096,1536), 'tensor_size_b': (2,4096,1536), 'dtype_a': torch.half, 'dtype_b': torch.half},
        {'tensor_size_a': (2,4096,1536), 'tensor_size_b': (2,1,1536), 'dtype_a': torch.half, 'dtype_b': torch.half},
        {'tensor_size_a': (2,154,1536), 'tensor_size_b': (2,1,1536),  'dtype_a': torch.half, 'dtype_b': torch.half},
        {'tensor_size_a': (4096,1,3072), 'tensor_size_b': (4096,1,3072),  'dtype_a': torch.half, 'dtype_b': torch.half},
        {'tensor_size_a': (3072,8192), 'tensor_size_b': (3072,8192),  'dtype_a': torch.float32, 'dtype_b': torch.half},
        ]

for config in configs:
    a = torch.randn(config['tensor_size_a'], device='cuda', dtype=config['dtype_a'])
    b = torch.randn(config['tensor_size_b'], device='cuda', dtype=config['dtype_b'])

    a = a.contiguous()
    b = b.contiguous()

    warmup = 10
    repeats = 50  # ~G~M~M次~U~L确~]计~W稳~Z

    for _ in range(warmup):
        c = a + b  # ~H~V使~T torch.add(a, b)

    torch.cuda.synchronize()
    start = time.time()

    for _ in range(repeats):
        c = a + b  # ~H~V使~T torch.add(a, b)
    print(c)

    torch.cuda.synchronize()
    end = time.time()

    print(config, f"Averaged execution time for aten::_add: {(end - start) / repeats * 1000:.3f} ms")
