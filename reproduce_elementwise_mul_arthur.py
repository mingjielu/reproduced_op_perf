# encoding=utf-8
import torch
import time
import os
# os.environ['TRITON_INTERPRET'] = '1'
import triton
import triton.language as tl
from tqdm import tqdm
@triton.jit
def matmul_kernel(
        # Pointers to matrices
        a_ptr, b_ptr, c_ptr,
        # Matrix dimensions
        M, N, K,
        # The stride variables represent how much to increase the ptr by when moving by 1
        # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
        # by to get the element one row down (A has M rows).
        stride_am, stride_ak,  #
        stride_bm, stride_bn,  #
        stride_cm, stride_ck,
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr,  
        BLOCK_SIZE_K: tl.constexpr,  #
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (M, N) and C has shape (M, K)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    pid = tl.program_id(axis=0)
    grid_n = tl.cdiv(K, BLOCK_SIZE_K)
    pid_m = pid // grid_n
    pid_k = pid % grid_n
    offset_am = ((pid_m * BLOCK_SIZE_M) + tl.arange(0, BLOCK_SIZE_M)) % M
    offset_k = ((pid_k * BLOCK_SIZE_K) + tl.arange(0, BLOCK_SIZE_K)) % K
    a_ptrs = a_ptr + (offset_am[:,None] * stride_am + offset_k[None,:] * stride_ak)
    b_ptrs = b_ptr + (offset_am[:,None] * stride_bm + offset_k[None,:] * stride_bn)

    a = tl.load(a_ptrs, mask=(offset_am[:,None]<M) & (offset_k[None, :] < K), other=0.0, )
    b = tl.load(b_ptrs, mask=(offset_am[:,None]<M) & (offset_k[None, :] < K), other=0.0, )
    bias = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)
    res = tl.fma(a, b, bias)    
    c_ptrs = c_ptr + (offset_am[:,None] * stride_cm + offset_k[None,:] * stride_ck)
    
    tl.store(c_ptrs, res, mask=(offset_am[:,None]<M) & (offset_k[None, :] < K))
    
def matmul(a, b, config):
    # Check constraints.

    M, N = a.shape
    M, K = b.shape
    assert N==1 or K==N
    # Allocates output.
    c = torch.empty((M, K), device=a.device, dtype=torch.float16)
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(K, META['BLOCK_SIZE_K']), )
    matmul_kernel[grid](
        a, b, c,  #
        M, N, K,  #
        a.stride(0), a.stride(1),  #
        b.stride(0), b.stride(1),  #
        c.stride(0), c.stride(1),  #
        **config
    )
    return c

def get_full_tuning_space():
    configs = []
    block_m_range = [32, 64, 128, 256, 512, 1024]
    block_k_range = [32, 64, 128, 256]
    kpack_range = [1, 2]
    waves_per_eu_range = [0,2,4,8,16]
    matrix_instr_nonkdim_range = [16, 32]
    stages = [1,2]
    for block_m in block_m_range:
        for block_k in block_k_range:
            for kpack in kpack_range:
                for waves in waves_per_eu_range:
                    for kdim in matrix_instr_nonkdim_range:
                        for stage in stages:
                            configs.append({
                                'BLOCK_SIZE_M': block_m,
                                'BLOCK_SIZE_K': block_k,
                                'matrix_instr_nonkdim': kdim,
                                'kpack': kpack,
                                'waves_per_eu': waves,
                                'num_stages': stage
                                })
    return configs


def autotune():
    torch.manual_seed(0)
    configs = get_full_tuning_space()
    best_config = None
    best_time_ms = 1e1
    for config in tqdm(configs):
        a = torch.randn((32768, 3072), device="cuda", dtype=torch.float16)
        b = torch.randn((32768, 3072), device="cuda", dtype=torch.float16)
        warmup = 10
        repeats = 50  

        for _ in range(warmup):
            c = matmul(a,b,config)
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(repeats):
            c = matmul(a,b,config)
        torch.cuda.synchronize()
        end = time.time()
        kernel_dur_ms = (end - start) / repeats * 1000
        print(config, f"Averaged execution time for aten::mul: {(end - start) / repeats * 1000:.3f} ms")
        
        if kernel_dur_ms < best_time_ms:
            best_config = config
            best_time_ms = kernel_dur_ms
    print(f'{best_config=}')
    print(f'{best_time_ms=}')
    
    
def main():
    #autotune()
    torch.manual_seed(0)
    a = torch.randn((32768, 3072), device="cuda", dtype=torch.float16)
    b = torch.randn((32768, 3072), device="cuda", dtype=torch.float16)
    print(a.stride(), a.is_contiguous())
    print(b.stride(), b.is_contiguous())
    # config = {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_K': 256, 'matrix_instr_nonkdim': 16, 'kpack': 1, 'waves_per_eu': 2, 'num_stages': 2}
    config={'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_K': 256, 'matrix_instr_nonkdim': 16, 'kpack': 1, 'waves_per_eu': 8, 'num_stages': 1} # 0.18929481506347656
    
    warmup = 10
    repeats = 50  # ~G~M~M次~U~L确~]计~W稳~Z

    for _ in range(warmup):
        c = matmul(a,b,config)  # ~H~V使~T torch.add(a, b)

    torch.cuda.synchronize()
    start = time.time()

    for _ in range(repeats):
        c = matmul(a,b,config)  # ~H~V使~T torch.add(a, b)

    torch.cuda.synchronize()
    end = time.time()

    print(config, f"Averaged execution time for triton::mul: {(end - start) / repeats * 1000:.3f} ms")
    

if __name__=='__main__':
    main()
