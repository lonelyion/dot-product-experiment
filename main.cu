#include "my_dot_product.h"

const int threadsPerBlock = 256;

__global__
void dot_product_gpu(const float *a, const float *b, double *partial_c, const int N) {
    __shared__ float cache[threadsPerBlock];    //这个内存缓冲区将保存每个线程计算的加和值
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;   //根据grid和block计算线程索引
    unsigned int cacheIdx = threadIdx.x;
    float temp = 0;
    while(tid < N) {
        temp += a[tid] * b[tid];
        tid += blockDim.x * gridDim.x;
    }
    cache[cacheIdx] = temp;

    __syncthreads();    // 线程同步

    // 规约运算 要求threadsPerBlock必须是2的指数
    int i = blockDim.x / 2;
    while(i != 0) {
        if(cacheIdx < i) {
            cache[cacheIdx] += cache[cacheIdx + i];
        }
        __syncthreads();    // 线程同步
        i /= 2;
    }
    if(cacheIdx == 0) {
        partial_c[blockIdx.x] = cache[0];
    }
}

void dot_product_cpu_openmp(const float *a, const float *b, double &result, size_t N) {
    double c = 0.0;
#pragma omp parallel for reduction(+:c)
    for(int i = 0; i < N; i++) {
        c += a[i] * b[i];
    }
#pragma omp barrier
    result = c;
}

void dot_product_cpu(const float *a, const float *b, double &c, size_t N) {
    c = 0.0;
    for(int i = 0; i < N; i++) {
        c += a[i] * b[i];
    }
}

int calc_main(int N, int blocksPerGrid) {
    float *host_a, *host_b;// host_c, dev_c;
    float *dev_a, *dev_b; //*dev_partial_c;
    double *host_partial_c, *dev_partial_c, host_c, dev_c;
    const size_t bytes_len = N*sizeof(float);
    // allocate host array
    host_a = (float*)malloc(bytes_len);
    host_b = (float*)malloc(bytes_len);
    host_partial_c = (double*)malloc(blocksPerGrid*sizeof(double));
    // allocate device array
    cudaMalloc((void**)&dev_a, bytes_len);
    cudaMalloc((void**)&dev_b, bytes_len);
    cudaMalloc((void**)&dev_partial_c, blocksPerGrid*sizeof(double));
    // generate random data with N(1, 1)    // Normal Distribution
    initialize_data(host_a, N);
    initialize_data(host_b, N);

    // CPU calculation
    auto ts_1 = std::chrono::steady_clock::now();
    dot_product_cpu(host_a, host_b, host_c, N);
    auto ts_2 = std::chrono::steady_clock::now();
    // CPU with OpenMP
    dot_product_cpu_openmp(host_a, host_b, host_c, N);
    auto ts_3 = std::chrono::steady_clock::now();
    // copy same data to device(GPU)
    cudaMemcpy(dev_a, host_a, bytes_len, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, host_b, bytes_len, cudaMemcpyHostToDevice);
    // GPU calculation
    //printf("Calculating from GPU...<<<%d, %d>>>\n", blocksPerGrid, threadsPerBlock);
    auto ts_4 = std::chrono::steady_clock::now();
    dot_product_gpu<<<blocksPerGrid, threadsPerBlock>>>(dev_a, dev_b, dev_partial_c, N);
    cudaDeviceSynchronize();
    auto ts_5 = std::chrono::steady_clock::now();
    cudaMemcpy(host_partial_c, dev_partial_c, blocksPerGrid*sizeof(double), cudaMemcpyDeviceToHost);
    // finish up calculating c
    dev_c = 0.0;
    for(int i = 0; i < blocksPerGrid; i++) {
        dev_c += host_partial_c[i];
    }
    // result check
    //if(abs(dev_c - host_c) <= 1.0) {
    //    printf("The result %.6f match!\n", dev_c);
    //} else {
    //    printf("The result did not match: CPU returns %.6f and GPU returns %.6f\n", host_c, dev_c);
    //}
    double diff = abs(dev_c - host_c);
    bool match = (diff <= 1.0);
    //printf("N: %9d | tCPU: %7.2lf | tGPU: %5.2lf | Diff: %.6lf\n", N, time_d(ts_3, ts_4), time_d(ts_1, ts_2), diff);
    //printf("%9d,%7.2lf,%5.2lf,%.6lf\n", N, time_d(ts_3, ts_4), time_d(ts_1, ts_2), diff);
    printf("%d,%d,%d,%d,%.6lf\n", N, time_d(ts_1, ts_2), time_d(ts_2, ts_3), time_d(ts_4, ts_5), diff);

    // release memory
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_partial_c);
    free(host_a);
    free(host_b);
    free(host_partial_c);
    return 0;
}

int main() {
    freopen("output.csv", "w", stdout);
    int data_size[25] = {16, 0};
    // 2^5  ~  2^28
    //  32  ~  268435456
    for(int i = 1; i < 25; i++) {
        data_size[i] = data_size[i-1] * 2;
    }
    for(int i : data_size) {
        int N = i;
        int blocksPerGrid = std::min(32, (N+threadsPerBlock-1)/threadsPerBlock);
        calc_main(N, blocksPerGrid);
    }
    return 0;
}