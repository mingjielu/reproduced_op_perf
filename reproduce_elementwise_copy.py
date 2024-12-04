import torch
import time

configs = [
        {'tensor_size_a': (2,4096,1536), 'tensor_size_b': (2,4096,1536),'dtype_a':torch.half,'dtype_b':torch.float32},
        {'tensor_size_a': (2,4096,1536), 'tensor_size_b': (2,4096,1536),'dtype_a':torch.float32,'dtype_b':torch.float32},           
        {'tensor_size_a': (2,4096,1536), 'tensor_size_b': (2,4096,1536),'dtype_a':torch.float32,'dtype_b':torch.half},
        {'tensor_size_a': (2,154,1536), 'tensor_size_b': (2,154,1536),'dtype_a':torch.half,'dtype_b':torch.float32},
        {'tensor_size_a': (2,154,1536), 'tensor_size_b': (2,154,1536),'dtype_a':torch.float32,'dtype_b':torch.float32},
        {'tensor_size_a': (2,154,1536), 'tensor_size_b': (2,154,1536),'dtype_a':torch.float32,'dtype_b':torch.half},
        ]

for config in configs:
    a = torch.randn(config['tensor_size_a'], device='cuda', dtype=config['dtype_a'])
    b = torch.empty(config['tensor_size_b'], device='cuda', dtype=config['dtype_b'])

    a = a.contiguous()
    b = b.contiguous()

    warmup = 10
    repeats = 50

    for _ in range(warmup):
        b.copy_(a)

    torch.cuda.synchronize()
    start = time.time()

    for _ in range(repeats):
        b.copy_(a)

    torch.cuda.synchronize()
    end = time.time()

    print(config, f"Averaged execution time for aten::copy: {(end - start) / repeats * 1000:.3f} ms")

