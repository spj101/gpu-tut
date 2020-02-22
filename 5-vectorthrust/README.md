## Vector Thrust

Use a GPU to perform vector addition:
a = a + b
using the thrust library.

The thrust library is a C++ parallel programming library which resembles the C++ Standard Template Library (STL). Under the hood it uses a variety of parallel progamming systems (C++ threads, omp, tbb, cuda).

Demonstrates:
* `thrust::host_vector<T>` - The thrust host vector type, similar to std::vector<T>. Stored on host.
* `thrust::device_vector<T>` - The thrust device vector type, similar to std::vector<T>. Stored on device.
* `thrust::transform<class InputIterator1, class InputIterator2, class OutputIterator, class BinaryOperation>(InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, OutputIterator result, BinaryOperation binary_op)` - The thrust transform algorithm. Similar to the C++ STL `std::transform` algorithm. Applies `binary_op` to each pair of elements from two input sequences and stores the result in the corresponding position in an output sequence.
*  `a = d_a` - Host-device communicaion in thrust.
* C++ functors `struct add { int operator()(...) const {...}` - A class/struct that implements the `()` operator.

See also:
* [CUDA Toolkit Thrust](https://docs.nvidia.com/cuda/thrust/)
* [Thrust Github](https://github.com/thrust/thrust) (Note: sometimes out-of-sync with Nvidia version)
