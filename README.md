# Dot Product Experiment

用不同的硬件和方法计算不同长度的向量点积并计算其消耗的时间

c = [a1, a2, ..., an] * [b1, b2, ..., bn] = sum(a1×b1, a2×b2, ... an×bn)

环境需求：

+ CMake 3.17 or above
+ CUDA版本不是太低应该都行
    + 须在环境变量中设置`CUDA_PATH`为CUDA的安装目录，在`PATH`中添加nvcc所在的目录
+ 支持OpenMP的C++编译器
