#include "../common/cuda_safe_call.hpp"

#include <stdio.h> // printf

// Vector addition: a = a + b
__global__ void add(int n, int *a, int *b)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] += b[i];
}

int main() 
{
    int device_id = 1;
    int n = 1000000; // number of elements
    int threads_per_block = 64;
    int blocks = (n+threads_per_block-1)/threads_per_block;
    size_t vector_size = n*sizeof(int); // size of n int

    // Set device
    CUDA_SAFE_CALL(cudaSetDevice(device_id));

    // Allocate host memory
    int *a, *b;
    a = (int*) malloc(vector_size);
    b = (int*) malloc(vector_size);
    
    // Allocate device memory
    int *d_a, *d_b;
    CUDA_SAFE_CALL(cudaMalloc(&d_a, vector_size));
    CUDA_SAFE_CALL(cudaMalloc(&d_b, vector_size));
    
    // Initialise host vectors
    for (size_t i = 0; i < n; i++)
    {
      a[i] = i;
      b[i] = 2*i;
    }
    
    // Copy host vectors to device
    CUDA_SAFE_CALL(cudaMemcpy(d_a, a, vector_size, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_b, b, vector_size, cudaMemcpyHostToDevice));
    
    // Perform vector addition
    add<<< blocks, threads_per_block>>>(n, d_a, d_b);
    
    CUDA_SAFE_CALL(cudaPeekAtLastError());
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    
    // Copy result back to host
    CUDA_SAFE_CALL(cudaMemcpy(a, d_a, vector_size, cudaMemcpyDeviceToHost));
    
    for(int i = 0; i < n; i++)
        if( a[i] != 3*i )
            printf("Error in result: a[%d] = %d (expected %d) \n", i, a[i], 3*i);
    
    // Check a few results
    printf("a[0]    = %d\n", a[0]);
    printf("a[n-1]  = %d\n", a[n-1]);
    
    // Free memory
    CUDA_SAFE_CALL(cudaFree(d_a));
    CUDA_SAFE_CALL(cudaFree(d_b));
    free(a);
    free(b);

    return 0;
}
