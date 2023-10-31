## Vector Sum

Use a GPU to reduce a vector $a$:
```math
r = \sum_i a_i
```
using the thrust library and cub.

**The example is taken from `NVIDIA/cccl/README.md [24a6161e]`, can you spot the code error?**

Demonstrates:
* `thrust::reduce` - Similar to the C++ STL `std::reduce`.
* `cub::BlockReduce` - Used for on-device reduction
* `cuda::atomic_ref`. - Used for on-device synchronisation

The `cub` reduction works as follows:
1. Each block computes the sum of a subset of the array using `cub::BlockReduce`
2. The sum of each block is then reduced to a single value using an atomic add
