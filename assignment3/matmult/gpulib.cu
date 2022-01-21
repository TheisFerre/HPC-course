#include <stdio.h>
#include <stdlib.h>
#include "cublas_v2.h"
#include <helper_cuda.h>
#include <omp.h>

#define BLOCK_SIZE 16
#define IDX2C(i,j,ld) (((j)*(ld))+(i) // https://docs.nvidia.com/cuda/cublas/index.html compatible way to index 2d matrices in C for cublas

extern "C" {
   void matmult_gpulib(int M, int N, int K, double *A_h, double *B_h, double *C_h); 
}

void matmult_gpulib(int M, int N, int K, double *A_h, double *B_h, double *C_h)
{
    double *A_d;
    double *B_d;
    double *C_d;

    double time, elapsed;
    double transfer_time, transfer_elabsed;

    int A_size = M * K * sizeof(double);
    int B_size = K * N * sizeof(double);
    int C_size = M * N * sizeof(double);

    cudaMalloc((void **)&A_d, A_size);
    cudaMalloc((void **)&B_d, B_size);
    cudaMalloc((void **)&C_d, C_size);

    // transfer data to cuda
    transfer_time = omp_get_wtime();
    cudaMemcpy(A_d, A_h, A_size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, B_size, cudaMemcpyHostToDevice);
    transfer_elabsed = omp_get_wtime() - transfer_time;

    // initiate threads (how do we size them?)
    // Initialize number of blocks and threads
    // int BLOCK_SIZE = 16;
    dim3 numOfThreadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    int xSize = ceil((double)(N + numOfThreadsPerBlock.x - 1) / (double)numOfThreadsPerBlock.x);
    int ySize = ceil((double)(M + numOfThreadsPerBlock.y - 1) / (double)numOfThreadsPerBlock.y);
    dim3 numOfBlocks(xSize, ySize);

    /* CALL TO: library routine */
    // A = M X K
    // B = K X N
    // C = M X N

    // Column Major: Leading Dimension (ld) = # of rows 
    // first dimension of A (M X K)
    int lda = K;
    // first dimension of B (K X N)
    int ldb = N;
    // first dimension of C (M X N)
    int ldc = N;

    const double alf = 1.0;
    const double bet = 0;
    const double *alpha = &alf;
    const double *beta = &bet;

    // create cublas handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    // call cblas library
    // cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0, A, lda, B, ldb, 0.0, C, ldc);
    // switch A and B. RowMajor --> ColumnMajor
    time = omp_get_wtime();
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, alpha, B_d, ldb, A_d, lda, beta, C_d, ldc);
    elapsed = omp_get_wtime() - time;

    // destroy handle
    cublasDestroy(handle);
    /*END OF CALL*/
    printf("%f\t%f\n", elapsed, transfer_elabsed);

    // checkCudaErrors(cudaDeviceSynchronize());
    cudaMemcpy(C_h, C_d, C_size, cudaMemcpyDeviceToHost);
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}