# Dot Product Experiment

用不同的硬件和方法计算不同长度的向量点积并计算其消耗的时间

c = [a1, a2, ..., an] * [b1, b2, ..., bn] = sum(a1×b1, a2×b2, ... an×bn)

程序分别用以下三种方式计算点积：
+ CPU单核暴力运算
+ OpenMP多线程运算
+ CUDA GPU运算

## 编译环境：

+ CMake 3.17 or above（CLion自带的好像编译不出OpenMP，自己装的可以）
+ CUDA版本不是太低应该都行
    + 须在环境变量中设置`CUDA_PATH`为CUDA的安装目录，在`PATH`中添加nvcc所在的目录
+ 支持OpenMP的C++编译器

## 运行和输出

如果你不想自己编译，也可以前往[Releases](https://github.com/lonelyion/dot_product_experiment/releases)页面下载Windows平台的二进制文件，当然需要NVIDIA的显卡（但是不需要安装CUDA），否则不能运行。

当然有N卡还是不能运行的话，可以尝试把数据量改小一点再编译，可能显存不够也会出问题。

程序会在当前目录创建一个`output.csv`的文件作为输出，表格样例如下，时间的单位为微秒(μs)：

| 向量长度N | tCPU1/单核运算时间 | tCPU2/OMP运算时间 | tGPU/CUDA运算时间 | Diff/CPU和GPU结果的差值 |
|  ----  | ----  |  ----  | ----  | ----  |
|16|0|13|82|0|
|32|0|108|23|0.000001|
|...|...|...|...|...|
|268435456|687158|110460|19145|0.586042|
