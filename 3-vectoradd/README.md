## Vector Add

Use a GPU to perform vector addition:
a = a + b

Demonstrates:
* Memory allocation: `cudaMalloc( void** devPtr, size_t size)`
* Freeing memory: `cudaFree ( void* devPtr )`
* Host/Device communication: `cudaMemcpy( void* dst, const void* src, size_t count, cudaMemcpyKind kind )`
