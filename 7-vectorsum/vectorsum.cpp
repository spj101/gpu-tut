#include "../common/cuda_safe_call.hpp"

#include <thrust/execution_policy.h> // thrust::device
#include <thrust/host_vector.h> // thrust::host_vector
#include <thrust/device_vector.h> // thrust::device_vector
#include <thrust/reduce.h> // thrust::reduce
#include <cub/block/block_reduce.cuh> // cub::BlockReduce
#include <cuda/atomic> // cuda::atomic_ref, cuda::thread_scope_device, cuda::memory_order_relaxed
#include <cstdio>

//
// Code inspired by github.com/nvidia/cccl/README.md
//
constexpr int block_size = 256;

__global__ void reduce(int const* data, int* result, int N) 
{
  using BlockReduce = cub::BlockReduce<int, block_size>;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  int const index = threadIdx.x + blockIdx.x * blockDim.x;
  int sum = 0;
  if (index < N) {
    sum += data[index];
  }
  sum = BlockReduce(temp_storage).Sum(sum);

  if (threadIdx.x == 0) 
  {
    cuda::atomic_ref<int, cuda::thread_scope_device> atomic_result(*result);
    atomic_result.fetch_add(sum, cuda::memory_order_relaxed);
  }
}

int main() 
{

  // Allocate and initialize input data
  int const N = 1000;
  thrust::device_vector<int> data(N);
  thrust::fill(data.begin(), data.end(), 1);

  // Allocate output data
  thrust::device_vector<int> kernel_result(1);

  // Compute the sum reduction of `data` using a custom kernel
  int const num_blocks = (N + block_size - 1) / block_size;
  reduce<<<num_blocks, block_size>>>(thrust::raw_pointer_cast(data.data()),
                                     thrust::raw_pointer_cast(kernel_result.data()),
                                     N);

  CUDA_SAFE_CALL(cudaDeviceSynchronize());

  int const custom_result = kernel_result[0];

  // Compute the same sum reduction using Thrust
  int const thrust_result = thrust::reduce(thrust::device, data.begin(), data.end(), 0);

  // Ensure the two solutions are identical
  std::printf("Custom kernel sum: %d\n", kernel_result[0]);
  std::printf("Thrust reduce sum: %d\n", thrust_result);
  assert(kernel_result[0] == thrust_result);
  return 0;
}
