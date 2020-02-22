#include "../common/cuda_safe_call.hpp"

#include <thrust/host_vector.h> // thrust::host_vector
#include <thrust/device_vector.h> // thrust::device_vector
#include <thrust/transform.h> // thrust::transform
#include <thrust/functional.h> // thrust::placeholders

#include <iostream> // std::cout

int main() 
{
    int device_id = 1;
    int n = 1000000; // number of elements
    
    // Set device
    CUDA_SAFE_CALL(cudaSetDevice(device_id));

    // Allocate host vectors
    thrust::host_vector<int> a(n);
    thrust::host_vector<int> b(n);
    
    // Initialise host vectors
    for (size_t i = 0; i < n; i++)
    {
      a[i] = i;
      b[i] = 2*i;
    }
    
    // Allocate device memory and copy host vectors to device
    thrust::device_vector<int> d_a = a;
    thrust::device_vector<int> d_b = b;
    
    // Perform vector addition
    {
        using namespace thrust::placeholders; // thrust::placeholders::_1, thrust::placeholders::_2
        thrust::transform(d_a.begin(), d_a.end(), d_b.begin(), d_a.begin(), _1 + _2);
    }

    // Copy result back to host
    a = d_a;
    
    for(size_t i = 0; i < n; i++)
        if( a[i] != 3*i )
            std::cout << "Error in result: a[" << i << "] = " << a[i] << " (expected " << 3*i << ")" << std::endl;

    return 0;
}
