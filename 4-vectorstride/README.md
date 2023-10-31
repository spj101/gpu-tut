## Vector Stride

Use a GPU to perform vector addition:
```math
a = a + b
```
using the grid-stride loop pattern.

Demonstrates:
* Grid-stride loop pattern
```cpp
for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
```
