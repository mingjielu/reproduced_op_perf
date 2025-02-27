import time
import torch
from torch.profiler import profile, record_function, ProfilerActivity
def test_add():
    torch.manual_seed(0)
    configs = [
            {'tensor_size_a': (2,4096,1536), 'tensor_size_b': (2,4096,1536), 'dtype_a': torch.half, 'dtype_b': torch.half},
            {'tensor_size_a': (2,4096,1536), 'tensor_size_b': (2,1,1536), 'dtype_a': torch.half, 'dtype_b': torch.half},
            {'tensor_size_a': (2,154,1536), 'tensor_size_b': (2,1,1536),  'dtype_a': torch.half, 'dtype_b': torch.half},
            {'tensor_size_a': (4096,1,3072), 'tensor_size_b': (4096,1,3072),  'dtype_a': torch.half, 'dtype_b': torch.half},
            {'tensor_size_a': (3072,8192), 'tensor_size_b': (3072,8192),  'dtype_a': torch.float32, 'dtype_b': torch.half},
            ]
    #custom_add = torch.add
    #custom_add = torch.compile(torch.add)
    #torch.add=custom_add
    for config in configs:
        a = torch.randn(config['tensor_size_a'], device='cuda', dtype=config['dtype_a'])
        b = torch.randn(config['tensor_size_b'], device='cuda', dtype=config['dtype_b'])
    
        a = a.contiguous()
        b = b.contiguous()
    
        warmup = 10
        repeats = 50  # ~G~M~M次~U~L确~]计~W稳~Z
    
        for _ in range(warmup):
            #c = custom_add(a , b)  # ~H~V使~T torch.add(a, b)
            c = torch.add(a , b)  # ~H~V使~T torch.add(a, b)
    
        start = time.time()
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_stack=True) as prof:
        #if True:
            for _ in range(repeats):
                #c = custom_add(a , b)  # ~H~V使~T torch.add(a, b)
                c = torch.add(a , b)  # ~H~V使~T torch.add(a, b)
        torch.cuda.synchronize()
        end = time.time()
        #print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10,max_name_column_width=None))
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
        print(config, f"Averaged execution time for aten::_add: {(end - start) / repeats * 1000:.3f} ms")


if __name__ == "__main__":
    test_add()
