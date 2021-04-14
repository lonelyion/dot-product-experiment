//
// Created by ion on 2021/4/7.
//

#ifndef _MY_DOT_PRODUCT_H
#define _MY_DOT_PRODUCT_H

#include <random>
#include <vector>
#include <iostream>
#include <chrono>
#include "omp.h"

extern const size_t vector_n;

template <typename T>
double dot_cpu(const std::vector<T> &a, const std::vector<T> &b, int n)
{
    double dTemp = 0;
    for (int i=0; i<n; ++i)
    {
        dTemp += a[i] * b[i];
    }
    return dTemp;
}

template <typename T>
void print_vector(T *a, size_t row, size_t len, std::vector<T> &out) {
    printf("(");
    for(int i = 0; i < len; i++) {
        out[i] = a[row*len + i];
        printf("%f", a[row*len + i]);
        if(i != len - 1) printf(",");
    }
    printf(")");
}

void initialize_data(float* ip,int size)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    //std::uniform_real_distribution<float> dis(0.0, 5.0);
    std::normal_distribution<float> dis(1.0, 0.2);
#pragma omp parallel for
    for(int i = 0; i < size; i++)
    {
        ip[i] = dis(gen);
    }
#pragma omp barrier
}

template <typename T>
bool check_result(T *dst_cpu, T *dst_gpu, const int size, float *a, float *b) {
    static const double epsilon = 1.0E-7;
    for(int i = 0; i < size; i++) {
        if(abs(dst_cpu[i] - dst_gpu[i]) > epsilon) {
            printf("Match failed at %d\n", i);
            std::vector<float> va(vector_n+1),vb(vector_n+1);
            print_vector(a, i, vector_n, va);
            printf(" + ");
            print_vector(b, i, vector_n, vb);
            printf("\nCPU: %f / GPU: %f / Validate: %f\n", dst_cpu[i], dst_gpu[i], dot_cpu(va, vb, vector_n));
            return false;
        }
    }
    return true;
}

int time_d(std::chrono::time_point<std::chrono::steady_clock> a, std::chrono::time_point<std::chrono::steady_clock> b) {
    //std::chrono::duration<double, std::milli> duration = (b - a);
    //return duration.count();
    return std::chrono::duration_cast<std::chrono::microseconds>(b - a).count();
}

#endif //VECTOR_SCALAR_PRODUCT_MY_DOT_PRODUCT_H
