## QMC

(Note: this example uses the 3rd party `qmc` integrator code)

Demonstrates how to integrate a complex function in a way compatible with both CPU and GPU evaluation. 

Since the type `std::complex` is not currently supported by GPUs we use the type `thrust::complex` if compiling with GPU support, this is achieved using a `typedef` which depends on the presence of `__CUDACC__`.

Demonstrates:
* `__CUDACC__` - Macro defined if compiling with `nvcc` compiler.
* Can also detect if we are compiling for a particular architecture
```cpp
__host__ __device__ func()
{
#if __CUDA_ARCH__ >= 700
   // Device code path for compute capability 7.x
#elif __CUDA_ARCH__ >= 600
   // Device code path for compute capability 6.x
#elif __CUDA_ARCH__ >= 500
   // Device code path for compute capability 5.x
#elif __CUDA_ARCH__ >= 300
   // Device code path for compute capability 3.x
#elif !defined(__CUDA_ARCH__) 
   // Host code path
#endif
}
```

See also:
* [qmc Github](https://github.com/mppmu/qmc)

