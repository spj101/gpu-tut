#include "../common/cuda_safe_call.hpp"

#include <stdio.h> // printf

__global__ void hello()
{
    printf( "Hello World from (block,thread): (%d,%d)\n", blockIdx.x, threadIdx.x );
}

int main() 
{
    int device_id = 1;
    
    int threads_per_block = 1;
    int blocks = 1;

    CUDA_SAFE_CALL(cudaSetDevice(device_id));

    hello<<< blocks, threads_per_block >>>();
    
    CUDA_SAFE_CALL(cudaPeekAtLastError());
    CUDA_SAFE_CALL(cudaDeviceSynchronize());

    return 0;
}
