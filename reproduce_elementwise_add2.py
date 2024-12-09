import torch
import time
 
configs = [
        {'tensor_size_a': (33792,5120), 'tensor_size_b': (33792,5120), 'dtype_a': torch.float32, 'dtype_b': torch.bfloat16, 'stride_a': (5120, 1), 'stride_b': (5120, 1)},
        {'tensor_size_a': (5120,16896), 'tensor_size_b': (5120,16896), 'dtype_a': torch.float32, 'dtype_b': torch.bfloat16, 'stride_a': (5120, 1), 'stride_b': (5120, 1)},
        {'tensor_size_a': (6144,5120), 'tensor_size_b': (6144,5120), 'dtype_a': torch.float32, 'dtype_b': torch.bfloat16, 'stride_a': (5120, 1), 'stride_b': (5120, 1)},
        ]
 
for config in configs:
    a = torch.randn(config['tensor_size_a'], device='cuda', dtype=config['dtype_a'])
    b = torch.randn(config['tensor_size_b'], device='cuda', dtype=config['dtype_b'])
    a = torch.as_strided(a, size=config['tensor_size_a'], stride=config['stride_a'])
    b = torch.as_strided(b, size=config['tensor_size_b'], stride=config['stride_b'])
 
    warmup = 10
    repeats = 50  # ~G~M~M次~U~L确~]计~W稳~Z
    #with torch.p
    for _ in range(warmup):
        #c = a + b  # ~H~V使~T torch.add(a, b)
        #a.add_(b)
        a += b
    torch.cuda.synchronize()
    start = time.time()
 
    for _ in range(repeats):
        #c = b + a  # ~H~V使~T torch.add(a, b)
        #$a.add_(b)
        a += b
    torch.cuda.synchronize()
    end = time.time()
    print(config, f"Averaged execution time for aten::_add: {(end - start) / repeats * 1000:.3f} ms")
