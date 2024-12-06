# encoding=utf-8
import torch
import time
import os
import triton
import triton.language as tl

@triton.jit
def mul_kernel_1(
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
    
@triton.jit
def mul_kernel_2(
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
    offset_n = tl.arange(0,1)
    a_ptrs = a_ptr + (offset_am[:,None] * stride_am + offset_k[None,:] * stride_ak)
    b_ptrs = b_ptr + (offset_am[:,None] * stride_bm + offset_n[None,:] * stride_bn)

    a = tl.load(a_ptrs, mask=(offset_am[:,None]<M) & (offset_k[None, :] < K), other=0.0, )
    b = tl.load(b_ptrs, mask=(offset_am[:,None]<M) & (offset_n[None, :]<1), other=0.0, )
    bias = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)
    tmp=tl.broadcast_to(b, (BLOCK_SIZE_M, BLOCK_SIZE_K))
    res = tl.fma(a, tmp, bias)   
    c_ptrs = c_ptr + (offset_am[:,None] * stride_am + offset_k[None,:] * stride_ak)
    tl.store(c_ptrs, res, mask=(offset_am[:,None]<M) & (offset_k[None, :] < K))
    
@triton.jit
def mul_kernel_3(
        # Pointers to matrices
        a_ptr, b_ptr, c_ptr,
        # Matrix dimensions
        M, N, K,
        # The stride variables represent how much to increase the ptr by when moving by 1
        # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
        # by to get the element one row down (A has M rows).
        stride_am, stride_an ,stride_ak,  #
        stride_cm, stride_cn, stride_ck,
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
    offset_n = tl.arange(0,1)
    a_ptrs = a_ptr + (offset_am[:,None, None] * stride_am + offset_n[None, :, None] * stride_an + offset_k[None, None,:] * stride_ak)
    b_ptrs = b_ptr
    
    a = tl.load(a_ptrs, mask=(offset_am[:, None, None]<M) & offset_n[None, :, None]<1 & (offset_k[None, None, :] < K), other=0.0, )
    b = tl.load(b_ptrs).to(tl.bfloat16)
    tmp=tl.broadcast_to(b, (BLOCK_SIZE_M, 1, BLOCK_SIZE_K))
    res = a*b
    c_ptrs = c_ptr + (offset_am[:,None, None] * stride_cm + offset_n[None, :, None] * stride_cn + offset_k[None,None,:] * stride_ck)
    tl.store(c_ptrs, res, mask=(offset_am[:, None, None]<M) & (offset_n[None, :, None]<1) &(offset_k[None, None,:] < K))
    
    

def triton_mul(a, b):
    ashape = list(a.shape)
    bshape = list(b.shape)

    if ashape==[4096,1,3072] and bshape==[1]:
        c = torch.empty((ashape[0], ashape[1], ashape[2]), device=a.device, dtype=a.dtype)
        # config = {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_K': 32, 'matrix_instr_nonkdim': 32, 'kpack': 2, 'waves_per_eu': 0, 'num_stages': 1} 
        config = {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_K': 256, 'matrix_instr_nonkdim': 32, 'kpack': 2, 'waves_per_eu': 8, 'num_stages': 2} # 0.04675865173339844
        grid = lambda META: (triton.cdiv(ashape[0], META['BLOCK_SIZE_M']) * triton.cdiv(ashape[2], META['BLOCK_SIZE_K']), )
        mul_kernel_3[grid](
            a, b, c,  #
            ashape[0], ashape[1], ashape[2],  #
            a.stride(0), a.stride(1), a.stride(2) , #
            c.stride(0), c.stride(1), c.stride(2) ,#
            **config
        )
        return c
    elif ashape==[32768, 3072] and bshape==[32768, 1]:
        
        # config = {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_K': 256, 'matrix_instr_nonkdim': 16, 'kpack': 2, 'waves_per_eu': 16, 'num_stages': 1}
        config = {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_K': 256, 'matrix_instr_nonkdim': 16, 'kpack': 1, 'waves_per_eu': 8, 'num_stages': 2} # 0.1852560043334961
        c = torch.empty((ashape[0], ashape[1]), device=a.device, dtype=a.dtype)
        # 1D launch kernel where each block gets its own program.
        grid = lambda META: (triton.cdiv(ashape[0], META['BLOCK_SIZE_M']) * triton.cdiv(ashape[1], META['BLOCK_SIZE_K']), )
        mul_kernel_2[grid](
            a, b, c,  #
            ashape[0], bshape[1], ashape[1] , #
            a.stride(0), a.stride(1),  #
            b.stride(0), b.stride(1),  #
            c.stride(0), c.stride(1),  #
            **config
        )
        return c
    elif ashape==[32768, 3072] and bshape==[32768, 3072]:
        c = torch.empty((ashape[0], ashape[1]), device=a.device, dtype=a.dtype)
        # config={'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_K': 256, 'matrix_instr_nonkdim': 16, 'kpack': 1, 'waves_per_eu': 8, 'num_stages': 1}
        config = {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_K': 256, 'matrix_instr_nonkdim': 32, 'kpack': 2, 'waves_per_eu': 2, 'num_stages': 2} # 0.22263050079345703
        grid = lambda META: (triton.cdiv(ashape[0], META['BLOCK_SIZE_M']) * triton.cdiv(ashape[1], META['BLOCK_SIZE_K']), )
        mul_kernel_1[grid](
            a, b, c,  #
            ashape[0], bshape[1], ashape[1] ,  #
            a.stride(0), a.stride(1),  #
            b.stride(0), b.stride(1),  #
            c.stride(0), c.stride(1),  #
            **config
        )
        return c
    else:
        return a*b

def precision(a, b):
    torch_res = a * b
    triton_res = triton_mul(a,b)
    diff = torch_res.eq(triton_res)
    # assert diff.all()
    if not diff.all():
        print(torch_res)
        print(triton_res)


def main():
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

        precision(a, b)
        
        warmup = 10
        repeats = 50  # ~G~M~M次~U~L确~]计~W稳~Z

        for _ in range(warmup):
            c = triton_mul(a , b)  # ~H~V使~T torch.add(a, b)
            #c = torch.mul(a , b)  # ~H~V使~T torch.add(a, b)
            #c = a * b  # ~H~V使~T torch.add(a, b)

        torch.cuda.synchronize()
        start = time.time()

        for _ in range(repeats):
            c = triton_mul(a , b)  # ~H~V使~T torch.add(a, b)
            #c = torch.mul(a , b)  # ~H~V使~T torch.add(a, b)
            #c = a * b  # ~H~V使~T torch.add(a, b)

        torch.cuda.synchronize()
        end = time.time()

        print(config, f"Averaged execution time for triton::mul: {(end - start) / repeats * 1000:.3f} ms")

if __name__ == '__main__':
    main()