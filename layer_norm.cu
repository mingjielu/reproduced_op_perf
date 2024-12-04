#include <iostream>

#include <cuda_fp16.h>

#include <cuda_runtime.h>

// CUDA kernel for element-wise addition of two FP16 arrays

__global__ void fp16_add(const half* A, const half* B, half* C, int num_elements) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_elements) {

        C[idx] = __hadd(A[idx], B[idx]);

    }

}


__global__ void fp16_add_vec(const half* A, const half* B, half* C, int num_elements) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

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


void checkCudaError(cudaError_t err) {

    if (err != cudaSuccess) {

        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;

        exit(EXIT_FAILURE);

    }

}

int main() {

    //const int num_elements = 1024 * 1024 * 128 ; // Example size
    const int num_elements = 32768 * 3072 ; // Example size

    const int size = num_elements * sizeof(half);

    // Host pointers

    half *h_A = (half*)malloc(size);

    half *h_B = (half*)malloc(size);

    half *h_C = (half*)malloc(size);

    // Initialize the input arrays

    for (int i = 0; i < num_elements; ++i) {

        h_A[i] = __float2half(i * 0.001f);

        h_B[i] = __float2half(i * 0.002f);

    }

    // Device pointers

    half *d_A, *d_B, *d_C;

    checkCudaError(cudaMalloc((void**)&d_A, size));

    checkCudaError(cudaMalloc((void**)&d_B, size));

    checkCudaError(cudaMalloc((void**)&d_C, size));

    // Copy inputs to the device

    checkCudaError(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));

    checkCudaError(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    // Setup execution configuration

    const int threadsPerBlock = 512;

    const int blocksPerGrid = (num_elements + threadsPerBlock - 1) / threadsPerBlock;

    const int blocksPerGrid_vec = blocksPerGrid / 4;

    // Create CUDA events to measure time

    cudaEvent_t start, stop;

    checkCudaError(cudaEventCreate(&start));

    checkCudaError(cudaEventCreate(&stop));

    // Record the start event
    for (int i = 0; i < 5; ++i) {

    checkCudaError(cudaEventRecord(start, 0));

    // Launch kernel

    fp16_add<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, num_elements);

    //fp16_add_vec<<<blocksPerGrid_vec, threadsPerBlock>>>(d_A, d_B, d_C, num_elements);

    // Record the stop event

    checkCudaError(cudaEventRecord(stop, 0));

    // Wait for the stop event to complete

    checkCudaError(cudaEventSynchronize(stop));

    // Calculate elapsed time

    float elapsedTime;

    checkCudaError(cudaEventElapsedTime(&elapsedTime, start, stop));

    std::cout << "Time to add two FP16 arrays: " << elapsedTime << " ms" << std::endl;
    }

    // Copy result back to host

    checkCudaError(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    // Cleanup

    free(h_A);

    free(h_B);

    free(h_C);

    checkCudaError(cudaFree(d_A));

    checkCudaError(cudaFree(d_B));

    checkCudaError(cudaFree(d_C));

    checkCudaError(cudaEventDestroy(start));

    checkCudaError(cudaEventDestroy(stop));

    return 0;

} 
