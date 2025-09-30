import torch
from torch.profiler import profile, record_function, ProfilerActivity


configs = [
        {'tensor_size_a': (1668,3584),  'dtype_a': torch.bfloat16, 'tensor_size_b': (3584, 18944), 'dtype_b': torch.bfloat16},
        {'tensor_size_a': (1668,18944),  'dtype_a': torch.bfloat16, 'tensor_size_b': (18944,3584), 'dtype_b': torch.bfloat16},
        #{'tensor_size_a': (512,1668),  'dtype_a': torch.bfloat16, 'tensor_size_b': (1668,3584), 'dtype_b': torch.bfloat16},
        ]

for config in configs:
    a = torch.randn(config['tensor_size_a'], device='cuda', dtype=config['dtype_a'], requires_grad = True)
    b = torch.randn(config['tensor_size_b'], device='cuda', dtype=config['dtype_b'], requires_grad = True)


    warmup = 10
    repeats = 56  #

    for _ in range(warmup):
        c = torch.mm(a, b)
        #c.sum().backward()


    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_stack=True) as prof:
        for _ in range(repeats):
            c = torch.mm(a , b)  # ~H~V使~T torch.add(a, b)
            #c.sum().backward()
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
