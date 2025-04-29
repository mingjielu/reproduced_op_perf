import torch
import time
from torch.profiler import profile, record_function, ProfilerActivity

points = torch.randn(4,6,88,16,44,3,1,device='cuda',dtype=torch.float32)
post_rots = torch.randn(4,6,1,1,1,3,3,device='cuda',dtype=torch.float32)

for i in range(5): #warmup
    output = post_rots.matmul(points)
torch.cuda.synchronize()
t1 = time.time()
output = post_rots.matmul(points)
torch.cuda.synchronize()
print('forward time(s/iter): ',time.time()-t1)

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_stack=True) as prof:
    output = post_rots.matmul(points)
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
