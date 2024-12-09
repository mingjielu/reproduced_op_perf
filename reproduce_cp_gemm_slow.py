import torch
import time
import transformer_engine as te
from torch.profiler import profile, record_function, ProfilerActivity
import os

class TELayerNormColumnParallelLinear(te.pytorch.LayerNormLinear):
    def __init__(
        self,
        input_size: int,
        output_size: int,
    ):
        super().__init__(
            in_features=input_size,
            out_features=output_size,
            eps=1e-05,
            sequence_parallel=False,
            fuse_wgrad_accumulation=False,
            tp_group=None,
            tp_size=1,
            get_rng_state_tracker=None,
            init_method=None,
            bias=True,
            return_bias=False,
            parallel_mode="column",
            return_layernorm_output=False,
            zero_centered_gamma=False
        )

    def forward(self, x):
        out = super().forward(x, is_first_microbatch=False)
        return out, None

def save_trace(prof):
    prof.export_chrome_trace('trace.json')
    print('trace saved')

def benchmark_ln_linear(batch_size=2, seq_len=4096, hidden_size=5120, warmup=10, iters=10):
    model = TELayerNormColumnParallelLinear(5120, 6144)
    model = model.cuda().bfloat16()
    x = torch.randn((seq_len, batch_size, hidden_size),
                    dtype=torch.bfloat16,
                    device='cuda',
                    requires_grad=True)

    for _ in range(warmup):
        out, _ = model(x)
        grad = torch.randn_like(out)
        out.backward(grad)
    torch.cuda.synchronize()

    # Profiling
    print("\nProfiling...")
    with profile(
        activities=[
            ProfilerActivity.CPU,
            ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(
            wait=2,
            warmup=2,
            active=2,
            repeat=1),
        record_shapes=True,
        with_stack=True,
        with_flops=True,
        on_trace_ready=save_trace
    ) as prof:
        for _ in range(iters):
            with record_function("forward"):
                out, _ = model(x)
            
            with record_function("backward"):
                grad = torch.randn_like(out)
                out.backward(grad)
            
            prof.step()

    # 常规计时
    print("\nTiming iterations...")
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(iters):
        out, _ = model(x)
    torch.cuda.synchronize()
    fwd_time = (time.time() - start) * 1000 / iters

    start = time.time()
    for _ in range(iters):
        out, _ = model(x)
        grad = torch.randn_like(out)
        out.backward(grad)
    torch.cuda.synchronize()
    fwd_bwd_time = (time.time() - start) * 1000 / iters

    bwd_time = fwd_bwd_time - fwd_time

    print("\nProfiler Results:")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    return fwd_time, bwd_time

fwd_time, bwd_time = benchmark_ln_linear()
print(f"\nAverage times:")
print(f"Forward time: {fwd_time:.3f} ms")
print(f"Backward time: {bwd_time:.3f} ms")
print(f"Total time: {fwd_time + bwd_time:.3f} ms")
