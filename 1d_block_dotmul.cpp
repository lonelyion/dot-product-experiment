#include <iostream>
#include <vector>
#include "my_dot_product.h"

const size_t data_set_n = 2048, vector_n = 3;
//blocksPerGrid    //threadsPerBlock

__global__
void vector_scalar_product(const float *a, const float *b, double *c) {
    //__shared__ double cache[vector_n];
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int bid = blockIdx.x * blockDim.x;
    c[bid] = 0;
    for(int i = 0; i < vector_n; i++) {
        c[bid] += a[bid*vector_n+i] * b[bid*vector_n+i];
    }
    //printf("bidx: %d / bdim: %d / c[bid]: %lf\n", blockIdx.x, blockDim.x, c[bid]);
}

void vsp_cpu(const float *a, const float *b, double *c, size_t len, size_t arr_len) {
    //a[2048][10] => a[x][y] => a[10*x+y]
    for(int i = 0; i < len; i++) {
        for(int j = 0; j < arr_len; j++) {
            c[i] += a[i * arr_len + j] * b[i * arr_len + j];
        }
    }
}

int main() {
    const unsigned int data_bytes = sizeof(float)*data_set_n*vector_n;
    //const unsigned int result_bytes = sizeof(float)*data_set_n;

    // initialize array
    auto host_a = new float[data_set_n*vector_n];
    auto host_b = new float[data_set_n*vector_n];
    auto host_result = new double[data_set_n];
    auto host_result_from_gpu = new double[data_set_n];

    // random generate data
    initialize_data(host_a, data_set_n*vector_n);
    initialize_data(host_b, data_set_n*vector_n);
    memset(host_result, 0, data_set_n*sizeof(double));

    // calculate on CPU
    std::cout << "Calculating from CPU..." << std::endl;
    vsp_cpu(host_a, host_b, host_result, data_set_n, vector_n);
    std::cout << "CPU Calculation finished" << std::endl;
    // initialize GPU array
    float *dev_a, *dev_b;
    double *dev_result;
    cudaMalloc((void**)&dev_a, data_bytes);
    cudaMalloc((void**)&dev_b, data_bytes);
    cudaMalloc((void**)&dev_result, data_set_n*sizeof(double));
    cudaMemcpy(dev_a, host_a, data_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, host_b, data_bytes, cudaMemcpyHostToDevice);
    printf("Calculating from GPU...<<<%zu, %zu>>>\n", data_set_n, vector_n);
    vector_scalar_product<<<data_set_n, 1>>>(dev_a, dev_b, dev_result);
    cudaDeviceSynchronize();
    cudaMemcpy(host_result_from_gpu, dev_result, data_set_n*sizeof(double), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    std::cout <<
              (check_result(host_result, host_result_from_gpu, data_set_n, host_a, host_b) ? "Test passed" : "Test failed")
              << std::endl;

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_result);
    delete[] host_a;
    delete[] host_b;
    delete[] host_result;
    delete[] host_result_from_gpu;
    return 0;
}
