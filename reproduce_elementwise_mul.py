import torch
import time

configs = [
        {'tensor_size_a': (32768,3072), 'tensor_size_b': (32768,1), 'dtype_a': torch.bfloat16, 'dtype_b': torch.bfloat16},
        {'tensor_size_a': (32768,3072), 'tensor_size_b': (32768,3072), 'dtype_a': torch.bfloat16, 'dtype_b': torch.bfloat16},
        {'tensor_size_a': (4096,1,3072), 'tensor_size_b': (1,),  'dtype_a': torch.bfloat16, 'dtype_b': torch.int64},
        ]

for config in configs:
    a = torch.randn(config['tensor_size_a'], device='cuda', dtype=config['dtype_a'])
    if config['dtype_b'] not in [torch.int32, torch.int64]:
        b = torch.randn(config['tensor_size_b'], device='cuda', dtype=config['dtype_b'])
    else:
        b = torch.randint(0, 1000, config['tensor_size_b'], device='cuda', dtype=config['dtype_b'])

    a = a.contiguous()
    b = b.contiguous()

    warmup = 10
    repeats = 50  # ~G~M~M次~U~L确~]计~W稳~Z

    for _ in range(warmup):
        c = a * b  # ~H~V使~T torch.add(a, b)

    torch.cuda.synchronize()
    start = time.time()

    for _ in range(repeats):
        c = a * b  # ~H~V使~T torch.add(a, b)

    torch.cuda.synchronize()
    end = time.time()

    print(config, f"Averaged execution time for aten::mul: {(end - start) / repeats * 1000:.3f} ms")
