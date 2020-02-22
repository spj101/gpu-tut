#include <iostream>
#include "../common/qmc.hpp"

#ifdef __CUDACC__
#include <thrust/complex.h>
typedef thrust::complex<double> complex_t;
#else
#include <complex>
typedef std::complex<double> complex_t;
#endif

struct my_functor_t {
    const unsigned long long int number_of_integration_variables = 3;
#ifdef __CUDACC__
    __host__ __device__
#endif
    complex_t operator()(double* x) const
    {
        return complex_t(1.,1.)*x[0]*x[1]*x[2];
    }
} my_functor;

int main() {
    
    integrators::Qmc<complex_t,double,3,integrators::transforms::Korobov<3>::type> integrator;
    integrators::result<complex_t> result = integrator.integrate(my_functor);
    
    std::cout << std::setprecision(16);
    std::cout << "integral = " << result.integral << std::endl;
    std::cout << "error    = " << result.error    << std::endl;

    return 0;
}
