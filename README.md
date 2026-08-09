GPU Accelerated Matrix Multiplication using CUDA

This project implements matrix multiplication using CUDA parallel computing and compares GPU performance with CPU execution.

Technologies Used
- CUDA
- C++

Features
- Parallel matrix multiplication using CUDA kernels
- Shared memory optimization using tiling technique
- Performance comparison between CPU and GPU
- Execution time measurement

Compilation and Execution
nvcc mult.cu -o mult
./mult

Sample Input
1024 1024 1024 1024

Output
GPU execution time is significantly lower than CPU execution time and results are verified for correctness.
