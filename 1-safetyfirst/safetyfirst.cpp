#include "../common/cuda_safe_call.hpp"

int main()
{
    int device_id = -1;

    cudaSetDevice(device_id); // Bad!
    // CUDA_SAFE_CALL(cudaSetDevice(device_id)); // Good

    return 0;
}
