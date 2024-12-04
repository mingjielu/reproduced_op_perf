#include "hip/hip_runtime.h"
#include <iostream>

#include <hip/hip_fp16.h>

#include <hip/hip_runtime.h>

// CUDA kernel for element-wise addition of two FP16 arrays

__global__ void fp16_add(const half* A, const half* B, half* C, int64_t num_elements) {

    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_elements) {

        C[idx] = __hadd(A[idx], B[idx]);

    }

}


__global__ void fp16_add_vec(const half* A, const half* B, half* C, int64_t num_elements) {

    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned long long * a1 = (unsigned long long *)(A);

    unsigned long long * b1 = (unsigned long long *)(B);

    unsigned long long * c1 = (unsigned long long *)(C);

    unsigned long long a_val = a1[idx];

    unsigned long long b_val = b1[idx];

    unsigned long long c_val = 0;

    half* a2 = (half*)(&a_val);

    half* b2 = (half*)(&b_val);

    half* c2 = (half*)(&c_val);

    for(int i = 0; i < 4; ++i){

        //c2[i] = a2[i]  + b2[i];

        c2[i] = __hadd(a2[i], b2[i]);

    }

    c1[idx] = c_val;

}


void checkCudaError(hipError_t err) {

    if (err != hipSuccess) {

        std::cerr << "CUDA error: " << hipGetErrorString(err) << std::endl;

        exit(EXIT_FAILURE);

    }

}

int main() {

    //const int64_t num_elements = 1024 * 1024 * 256; // Example size
    const int64_t num_elements = 192 * 577 * 3072; // Example size

    const int64_t size = num_elements * sizeof(half);

    // Host pointers

    half *h_A = (half*)malloc(size);

    half *h_B = (half*)malloc(size);

    half *h_C = (half*)malloc(size);

    // Initialize the input arrays

    for (int64_t i = 0; i < num_elements; ++i) {

        h_A[i] = __float2half(i * 0.001f);

        h_B[i] = __float2half(i * 0.002f);

    }

    // Device pointers

    half *d_A, *d_B, *d_C;

    checkCudaError(hipMalloc((void**)&d_A, size));

    checkCudaError(hipMalloc((void**)&d_B, size));

    checkCudaError(hipMalloc((void**)&d_C, size));

    // Copy inputs to the device

    checkCudaError(hipMemcpy(d_A, h_A, size, hipMemcpyHostToDevice));

    checkCudaError(hipMemcpy(d_B, h_B, size, hipMemcpyHostToDevice));

    // Setup execution configuration

    const uint32_t threadsPerBlock = 128;

    const uint32_t blocksPerGrid = (num_elements + threadsPerBlock - 1) / threadsPerBlock;

    const uint32_t blocksPerGrid_vec = blocksPerGrid / 4;

    // Create CUDA events to measure time

    hipEvent_t start, stop;

    checkCudaError(hipEventCreate(&start));

    checkCudaError(hipEventCreate(&stop));

    // Record the start event

    for (int64_t i = 0; i < 5; ++i) {
    checkCudaError(hipEventRecord(start, 0));

    // Launch kernel

    fp16_add<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, num_elements);

    //fp16_add_vec<<<blocksPerGrid_vec, threadsPerBlock>>>(d_A, d_B, d_C, num_elements);

    // Record the stop event

    checkCudaError(hipEventRecord(stop, 0));

    // Wait for the stop event to complete

    checkCudaError(hipEventSynchronize(stop));

    // Calculate elapsed time

    float elapsedTime;

    checkCudaError(hipEventElapsedTime(&elapsedTime, start, stop));

    //fp16_add_vec<<<blocksPerGrid_vec, threadsPerBlock>>>(d_A, d_B, d_C, num_elements);

    std::cout << "Time to add two FP16 arrays: " << elapsedTime << " ms" << std::endl;

    checkCudaError(hipEventRecord(start, 0));

    fp16_add_vec<<<blocksPerGrid_vec, threadsPerBlock>>>(d_A, d_B, d_C, num_elements);

    checkCudaError(hipEventRecord(stop, 0));

    checkCudaError(hipEventSynchronize(stop));

    elapsedTime = 0;

    checkCudaError(hipEventElapsedTime(&elapsedTime, start, stop));

    std::cout << "Time to add_vec two FP16 arrays: " << elapsedTime << " ms" << std::endl;
    }

    // Copy result back to host

    checkCudaError(hipMemcpy(h_C, d_C, size, hipMemcpyDeviceToHost));

    // Cleanup

    free(h_A);

    free(h_B);

    free(h_C);

    checkCudaError(hipFree(d_A));

    checkCudaError(hipFree(d_B));

    checkCudaError(hipFree(d_C));

    checkCudaError(hipEventDestroy(start));

    checkCudaError(hipEventDestroy(stop));

    return 0;

} 
