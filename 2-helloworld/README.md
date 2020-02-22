## Hello World

Demonstates:
* Kernel declaration: `__global__ void hello()`
* Kernel invocation: `hello<<< blocks, threads_per_block >>>()`
* Device synchronisation: `cudaDeviceSynchronize()`
* Built-in variables: `blockIdx` and `threadIdx`

CUDA function execution space specifiers:
* `__global__` - Declares a kernel. Executed on the device, callable from the host. Callable from the device for devices of compute capability 3.2 or higher (CUDA Dynamic Parallelism).
* `__host__` - Executed on the host, callable from the host only.
* `__device__` - Executed on the device, callable from the device only.

CUDA buit-in variables:
* `dim3 gridDim` -  Dimension of the grid in blocks
* `dim3 blockDim` - Dimension of the block in threads
* `dim3 blockIdx` - Block index within the grid
* `dim3 threadIdx` - Thread index within the block
* (`int warpSize` - Warp size in threads)

The `dim3` type variables can be accessed as:
* `threadIdx.x` - Access x-direction of `threadIdx`
* `threadIdx.y` - Access y-direction of `threadIdx`
* `threadIdx.z` - Access z-direction of `threadIdx`
