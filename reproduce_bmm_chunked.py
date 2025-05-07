import torch
import time

torch.manual_seed(0)

points = torch.randn(4, 6, 88, 16, 44, 3, 1, device='cuda', dtype=torch.float32)
post_rots = torch.randn(4, 6, 1, 1, 1, 3, 3, device='cuda', dtype=torch.float32)

# baseline
for i in range(5): # warmup
    out_ref = post_rots.matmul(points)
torch.cuda.synchronize()
t0 = time.time()
out_ref = post_rots.matmul(points)
torch.cuda.synchronize()
t1 = time.time()
print(f"[baseline matmul] time: {t1 - t0:.4f} s")

# chunked matmul
flat_points = points.reshape(-1, 3, 1)
flat_post_rots = post_rots.expand(4, 6, 88, 16, 44, 3, 3).reshape(-1, 3, 3)

chunk_size = 65535
num_chunks = (flat_points.shape[0] + chunk_size - 1) // chunk_size

for i in range(5): #warmup
    outputs = []
    for i in range(num_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, flat_points.shape[0])
        out = torch.bmm(flat_post_rots[start:end], flat_points[start:end])
        outputs.append(out)
    out_chunked = torch.cat(outputs, dim=0).reshape(4, 6, 88, 16, 44, 3, 1)

outputs = []
torch.cuda.synchronize()
t0 = time.time()
for i in range(num_chunks):
    start = i * chunk_size
    end = min((i + 1) * chunk_size, flat_points.shape[0])
    out = torch.bmm(flat_post_rots[start:end], flat_points[start:end])
    outputs.append(out)
out_chunked = torch.cat(outputs, dim=0).reshape(4, 6, 88, 16, 44, 3, 1)
torch.cuda.synchronize()
t1 = time.time()
print(f"[chunked matmul] time: {t1 - t0:.4f} s")

# correctness check
max_diff = (out_ref - out_chunked).abs().max().item()
print(f"[max abs diff] = {max_diff:.6f}")
