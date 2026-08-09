#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <time.h>
#include <iostream>

#define BLOCK_SIZE 16

// CPU matrix multiplication
__host__ void cpu_matrix_mult(float *A, float *B, float *C, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            float sum = 0.0;
            for (int k = 0; k < n; ++k) {
                sum += A[i * n + k] * B[k * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

// Fill matrices and convert to square
__host__ int fill(float **A, float **B, int ax, int ay, int bx, int by) {

    int size = ax > bx ? ax : bx;
    size = size > ay ? size : ay;
    size = size > by ? size : by;

    int temp = size / BLOCK_SIZE + (size % BLOCK_SIZE ? 1 : 0);
    size = temp * BLOCK_SIZE;

    size_t bytes = size * size * sizeof(float);

    *A = (float *)malloc(bytes);
    *B = (float *)malloc(bytes);

    memset(*A, 0, bytes);
    memset(*B, 0, bytes);

    for (int i = 0; i < ax; i++) {
        for (int j = 0; j < ay; j++) {
            (*A)[i * size + j] = sinf(i + j);
        }
    }

    for (int i = 0; i < bx; i++) {
        for (int j = 0; j < by; j++) {
            (*B)[i * size + j] = cosf(i + j);
        }
    }

    return size;
}

// CUDA kernel using tiling
__global__ void matrixMulKernel(float *A, float *B, float *C, int n) {

    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float value = 0;

    for (int t = 0; t < n / BLOCK_SIZE; t++) {

        As[threadIdx.y][threadIdx.x] = A[row * n + t * BLOCK_SIZE + threadIdx.x];
        Bs[threadIdx.y][threadIdx.x] = B[(t * BLOCK_SIZE + threadIdx.y) * n + col];

        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; k++) {
            value += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < n && col < n) {
        C[row * n + col] = value;
    }
}

int main() {

    int ax, ay, bx, by;

    printf("Enter m n n k:\n");
    scanf("%d %d %d %d", &ax, &ay, &bx, &by);

    float *A_host, *B_host, *C_host, *C_cpu;
    float *A_dev, *B_dev, *C_dev;

    int n = fill(&A_host, &B_host, ax, ay, bx, by);

    size_t bytes = n * n * sizeof(float);

    C_host = (float *)malloc(bytes);
    C_cpu = (float *)malloc(bytes);

    cudaMalloc(&A_dev, bytes);
    cudaMalloc(&B_dev, bytes);
    cudaMalloc(&C_dev, bytes);

    cudaMemcpy(A_dev, A_host, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(B_dev, B_host, bytes, cudaMemcpyHostToDevice);

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(n / BLOCK_SIZE, n / BLOCK_SIZE);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    matrixMulKernel<<<grid, block>>>(A_dev, B_dev, C_dev, n);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpu_time;
    cudaEventElapsedTime(&gpu_time, start, stop);

    cudaMemcpy(C_host, C_dev, bytes, cudaMemcpyDeviceToHost);

    clock_t begin = clock();
    cpu_matrix_mult(A_host, B_host, C_cpu, n);
    clock_t end = clock();

    double cpu_time = 1000.0 * (end - begin) / CLOCKS_PER_SEC;

    printf("GPU time = %f ms\n", gpu_time);
    printf("CPU time = %lf ms\n", cpu_time);
    printf("Speedup = %lf\n", cpu_time / gpu_time);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }

    bool correct = true;
    for (int i = 0; i < ax && correct; i++) {
        for (int j = 0; j < by && correct; j++) {
            if (fabs(C_host[i * n + j] - C_cpu[i * n + j]) > 0.001) {
                correct = false;
            }
        }
    }

    if (correct)
        printf("Results are correct\n");
    else
        printf("Mismatch found\n");

    free(A_host);
    free(B_host);
    free(C_host);
    free(C_cpu);

    cudaFree(A_dev);
    cudaFree(B_dev);
    cudaFree(C_dev);

    return 0;
}