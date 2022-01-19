
#include <stdio.h>
#include <stdlib.h>
#include <helper_cuda.h>

#define min(x,y) (((x) < (y)) ? (x) : (y))


__global__ void gpu1_kernel(int M,int N,int K, double *A_d, double *B_d, double *C_d){

    // single thread to compute all data
    int m, n, k;

    for(m = 0; m < M; m++){
        for(k = 0; k < K; k++){
            for(n = 0; n < N; n++){
                C_d[m * N + n] += A_d[m * K + k] * B_d[k * N + n];
            }
        }
    }

}
extern "C" {
    void matmult_gpu1(int M, int N, int K, double *A_h, double *B_h, double *C_h){
        double *A_d;
        double *B_d;
        double *C_d;

        int A_size = M * K * sizeof(double);
        int B_size = K * N * sizeof(double);
        int C_size = M * N * sizeof(double);

        cudaMalloc((void **)&A_d, A_size);
        cudaMalloc((void **)&B_d, B_size);
        cudaMalloc((void **)&C_d, C_size);

        // transfer data to cuda
        cudaMemcpy(A_d, A_h, A_size, cudaMemcpyHostToDevice);
        cudaMemcpy(B_d, B_h, B_size, cudaMemcpyHostToDevice);
        cudaMemcpy(C_d, C_h, C_size, cudaMemcpyHostToDevice);

        // initiate threads (how do we size them?)
        // Initialize number of blocks and threads
        int BLOCK_SIZE = 1;

        dim3 numOfThreadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
        dim3 numOfBlocks(BLOCK_SIZE, BLOCK_SIZE);
        gpu1_kernel<<<numOfBlocks, numOfThreadsPerBlock>>>(M, N , K, A_d, B_d, C_d);

        cudaDeviceSynchronize();
        cudaMemcpy(C_h, C_d, C_size, cudaMemcpyDeviceToHost);
        cudaFree(A_d);
        cudaFree(B_d);
        cudaFree(C_d); 
    }
}

__global__ void gpu2_kernel(int M, int N, int K, double *A_d, double *B_d, double *C_d){

    int ROW = blockIdx.y * blockDim.y + threadIdx.y;
    int COL = blockIdx.x * blockDim.x + threadIdx.x;

    // A = M X K
    // B = K X N
    // C = M X N

    if (ROW < M && COL < N) {
        int i, j;
        double sum_val = 0;
        for (i = 0; i < K; i++){
            sum_val += A_d[ROW * K + i] * B_d[i * N + COL];
        }
        C_d[ROW * N + COL] = sum_val;
    }
    

}
extern "C" {
    void matmult_gpu2(int M, int N, int K, double *A_h, double *B_h, double *C_h){
        double *A_d;
        double *B_d;
        double *C_d;

        int A_size = M * K * sizeof(double);
        int B_size = K * N * sizeof(double);
        int C_size = M * N * sizeof(double);

        cudaMalloc((void **)&A_d, A_size);
        cudaMalloc((void **)&B_d, B_size);
        cudaMalloc((void **)&C_d, C_size);

        // transfer data to cuda
        cudaMemcpy(A_d, A_h, A_size, cudaMemcpyHostToDevice);
        cudaMemcpy(B_d, B_h, B_size, cudaMemcpyHostToDevice);

        // initiate threads (how do we size them?)
        // Initialize number of blocks and threads
        int BLOCK_SIZE = 16;
        dim3 numOfThreadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
        int xSize = ceil((double)(N + numOfThreadsPerBlock.x - 1) / (double)numOfThreadsPerBlock.x);
        int ySize = ceil((double)(M + numOfThreadsPerBlock.y - 1) / (double)numOfThreadsPerBlock.y);
        dim3 numOfBlocks(xSize, ySize);

        gpu2_kernel<<<numOfThreadsPerBlock, numOfBlocks>>>(M, N , K, A_d, B_d, C_d);

        cudaDeviceSynchronize();
        cudaMemcpy(C_h, C_d, C_size, cudaMemcpyDeviceToHost);
        cudaFree(A_d);
        cudaFree(B_d);
        cudaFree(C_d);
        
    }

}

__global__ void gpu3_kernel(int M, int N, int K, double *A_d, double *B_d, double *C_d){

    int ROW1 = (blockIdx.y * blockDim.y + threadIdx.y) * 2;
    int ROW2 = ROW1 + 1;
    int COL = blockIdx.x * blockDim.x + threadIdx.x;

    // A = M X K
    // B = K X N
    // C = M X N

    if (ROW2 < M && COL < N) {
        int i, j;
        double sum_val1 = 0;
        double sum_val2 = 0;
        for (i = 0; i < K; i++){
            sum_val1 += A_d[ROW1 * K + i] * B_d[i * N + COL];
            sum_val2 += A_d[ROW2 * K + i] * B_d[i * N + COL];
        }
        C_d[ROW1 * N + COL] = sum_val1;
        C_d[ROW2 * N + COL] = sum_val2;
    }

    else if (ROW1 < M && COL < N){
        int i, j;
        double sum_val1 = 0;
        for (i = 0; i < K; i++){
            sum_val1 += A_d[ROW1 * K + i] * B_d[i * N + COL];
        }
        C_d[ROW1 * N + COL] = sum_val1;

    }
    

}
extern "C" {
    void matmult_gpu3(int M, int N, int K, double *A_h, double *B_h, double *C_h){
        double *A_d;
        double *B_d;
        double *C_d;

        int A_size = M * K * sizeof(double);
        int B_size = K * N * sizeof(double);
        int C_size = M * N * sizeof(double);

        cudaMalloc((void **)&A_d, A_size);
        cudaMalloc((void **)&B_d, B_size);
        cudaMalloc((void **)&C_d, C_size);

        // transfer data to cuda
        cudaMemcpy(A_d, A_h, A_size, cudaMemcpyHostToDevice);
        cudaMemcpy(B_d, B_h, B_size, cudaMemcpyHostToDevice);

        // initiate threads (how do we size them?)
        // Initialize number of blocks and threads
        // M / BLOCK_SIZE has to be greater than or equal to 1
        int BLOCK_SIZE = 4;
        dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
        // int xSize = (N + dimBlock.x - 1) / dimBlock.x;
        // int ySize = (M + dimBlock.y - 1) / dimBlock.y;

        // which dimension to half
        // C: M X N
        int xSize = ceil((double)N / (double)dimBlock.x);
        int ySize = ceil((double)M / (double)dimBlock.y / 2.0);

        dim3 dimGrid(xSize, ySize);
        gpu3_kernel<<<dimGrid, dimBlock>>>(M, N , K, A_d, B_d, C_d);
        cudaDeviceSynchronize();

        cudaMemcpy(C_h, C_d, C_size, cudaMemcpyDeviceToHost);
        cudaFree(A_d);
        cudaFree(B_d);
        cudaFree(C_d);
        
    }

}

#define THREAD_COMPUTE 4 // number of c elements a thread should compute
__global__ void gpu4_kernel(int M, int N, int K, double *A_d, double *B_d, double *C_d){

    int ROW = (blockIdx.y * blockDim.y + threadIdx.y) * THREAD_COMPUTE;
    int COL = blockIdx.x * blockDim.x + threadIdx.x;

    // A = M X K
    // B = K X N
    // C = M X N

    // TODO: split into if-else blocks
    int t;
    for (t = 0; t < THREAD_COMPUTE; t++){
        int i, j;
        double sum_val = 0;
        if ((ROW + t) < M && COL < N){
            for (i = 0; i < K; i++){
                sum_val += A_d[(ROW + t) * K + i] * B_d[i * N + COL];
            }
            C_d[(ROW + t) * N + COL] = sum_val;
        }
    }
}

extern "C" {
    void matmult_gpu4(int M, int N, int K, double *A_h, double *B_h, double *C_h){
        double *A_d;
        double *B_d;
        double *C_d;

        int A_size = M * K * sizeof(double);
        int B_size = K * N * sizeof(double);
        int C_size = M * N * sizeof(double);

        cudaMalloc((void **)&A_d, A_size);
        cudaMalloc((void **)&B_d, B_size);
        cudaMalloc((void **)&C_d, C_size);

        // transfer data to cuda
        cudaMemcpy(A_d, A_h, A_size, cudaMemcpyHostToDevice);
        cudaMemcpy(B_d, B_h, B_size, cudaMemcpyHostToDevice);

        // initiate threads (how do we size them?)
        // Initialize number of blocks and threads
        // M / BLOCK_SIZE has to be greater than or equal to 1
        int BLOCK_SIZE = 1;
        dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
        // int xSize = (N + dimBlock.x - 1) / dimBlock.x;
        // int ySize = (M + dimBlock.y - 1) / dimBlock.y;

        // which dimension to half
        // C: M X N
        int xSize = ceil((double) N / (double) dimBlock.x);
        int ySize = ceil((double) M / (double) dimBlock.y / (double) THREAD_COMPUTE);

        dim3 dimGrid(xSize, ySize);
        gpu4_kernel<<<dimGrid, dimBlock>>>(M, N , K, A_d, B_d, C_d);
        cudaDeviceSynchronize();

        cudaMemcpy(C_h, C_d, C_size, cudaMemcpyDeviceToHost);
        cudaFree(A_d);
        cudaFree(B_d);
        cudaFree(C_d);
        
    }

}

// Thread block size
#define BLOCK_SIZE 4

// M(row, col) = *(M.elements + row * M.stride + col)
typedef struct {
    int width;
    int height;
    int stride; 
    double* elements;
} Matrix;

// Get a matrix element
__device__ float GetElement(const Matrix A, int row, int col)
{
    return A.elements[row * A.stride + col];
}

// Set a matrix element
__device__ void SetElement(Matrix A, int row, int col,
                           float value)
{
    A.elements[row * A.stride + col] = value;
}

// Get the BLOCK_SIZExBLOCK_SIZE sub-matrix Asub of A that is
// located col sub-matrices to the right and row sub-matrices down
// from the upper-left corner of A
 __device__ Matrix GetSubMatrix(Matrix A, int row, int col) 
{
    Matrix Asub;
    Asub.width    = BLOCK_SIZE;
    Asub.height   = BLOCK_SIZE;
    Asub.stride   = A.stride;
    Asub.elements = &A.elements[A.stride * BLOCK_SIZE * row
                                         + BLOCK_SIZE * col];
    return Asub;
}

__global__ void gpu5_kernel(Matrix A, Matrix B, Matrix C){
    // Block row and column
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    // Each thread block computes one sub-matrix Csub of C
    Matrix Csub = GetSubMatrix(C, blockRow, blockCol);

    // Each thread computes one element of Csub
    // by accumulating results into Cvalue
    float Cvalue = 0;

    // Thread row and column within Csub
    int row = threadIdx.y;
    int col = threadIdx.x;

    // Loop over all the sub-matrices of A and B that are
    // required to compute Csub
    // Multiply each pair of sub-matrices together
    // and accumulate the results
    for (int m = 0; m < (A.width / BLOCK_SIZE); ++m) {
        // Get sub-matrix Asub of A
        Matrix Asub = GetSubMatrix(A, blockRow, m);

        // Get sub-matrix Bsub of B
        Matrix Bsub = GetSubMatrix(B, m, blockCol);

        // Shared memory used to store Asub and Bsub respectively
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

        // Load Asub and Bsub from device memory to shared memory
        // Each thread loads one element of each sub-matrix
        As[row][col] = GetElement(Asub, row, col);
        Bs[row][col] = GetElement(Bsub, row, col);

        // Synchronize to make sure the sub-matrices are loaded
        // before starting the computation
        __syncthreads();
        // Multiply Asub and Bsub together
        for (int e = 0; e < BLOCK_SIZE; ++e) {
            Cvalue += As[row][e] * Bs[e][col];
        }

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // // Write Csub to device memory
    // // Each thread writes one element
    SetElement(Csub, row, col, Cvalue);
    
}

extern "C" {
    void matmult_gpu5(int M, int N, int K, double *A_h, double *B_h, double *C_h){

        // initialize matrix structs
        Matrix A, B, C;
        A.width = K; A.height = M; A.elements = A_h;
        B.width = N; B.height = K; B.elements = B_h;
        C.width = N; C.height = M; C.elements = C_h;

        // Load A and B to device memory
        Matrix d_A;
        d_A.width = d_A.stride = A.width; d_A.height = A.height;
        size_t size = A.width * A.height * sizeof(double);
        cudaMalloc(&d_A.elements, size);
        cudaMemcpy(d_A.elements, A.elements, size,
                cudaMemcpyHostToDevice);

        Matrix d_B;
        d_B.width = d_B.stride = B.width; d_B.height = B.height;
        size = B.width * B.height * sizeof(double);
        cudaMalloc(&d_B.elements, size);
        cudaMemcpy(d_B.elements, B.elements, size,
                cudaMemcpyHostToDevice);

        // Allocate C in device memory
        Matrix d_C;
        d_C.width = d_C.stride = C.width; d_C.height = C.height;
        size = C.width * C.height * sizeof(double);
        cudaMalloc(&d_C.elements, size);

        // Invoke kernel
        dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
        dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);
        gpu5_kernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);

        checkCudaErrors(cudaDeviceSynchronize());

        // Read C from device memory
        cudaMemcpy(C.elements, d_C.elements, size,
                cudaMemcpyDeviceToHost);

        // Free device memory
        cudaFree(d_A.elements);
        cudaFree(d_B.elements);
        cudaFree(d_C.elements);
    }
}

void matmult_gpu6(int M, int N, int K, double *A_h, double *B_h, double *C_h){

}

extern "C" {
    #include <cblas.h>
    void matmult_nat(int M, int N, int K, double *A, double *B, double *C) {

        int m, n, k;

        for(m = 0; m < M; m++){
            for(n = 0; n < N; n++){
                C[m * N + n] = 0;
            }
        }
        
        //Triple for loop for calculation (mkn)
        for(m = 0; m < M; m++){
            for(k = 0; k < K; k++){
                for(n = 0; n < N; n++){
                    C[m * N + n] += A[m * K + k] * B[k * N + n];
                }
            }
        }
    }

    void matmult_lib(int M, int N, int K, double *A, double *B, double *C){

        // first dimension of A (m X K)
        int lda = K;
        // first dimension of B (K X N)
        int ldb = N;
        // first dimension of C (M X N)
        int ldc = N;

        // call cblas library
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0, A, lda, B, ldb, 0.0, C, ldc);
    }

    void matmult_mkn(int M, int N, int K, double *A, double *B, double *C) {

        int m, n, k;

        for(m = 0; m < M; m++){
            for(n = 0; n < N; n++){
                C[m * N + n] = 0;
            }
        }
        
        //Triple for loop for calculation (mkn)
        for(m = 0; m < M; m++){
            for(k = 0; k < K; k++){
                for(n = 0; n < N; n++){
                    C[m * N + n] += A[m * K + k] * B[k * N + n];
                }
            }
        }
    }

    void matmult_mnk(int M, int N, int K, double *A, double *B, double *C) {

        int m, n, k;

        for(m = 0; m < M; m++){
            for(n = 0; n < N; n++){
                C[m * N + n] = 0;
            }
        }
        
        //Triple for loop for calculation (mnk)
        for(m = 0; m < M; m++){
            for(n = 0; n < N; n++){
                for(k = 0; k < K; k++){
                    C[m * N + n] += A[m * K + k] * B[k * N + n];
                }
            }
        }

    }

    void matmult_kmn(int M, int N, int K, double *A, double *B, double *C) {

        int m, n, k;

        for(m = 0; m < M; m++){
            for(n = 0; n < N; n++){
                C[m * N + n] = 0;
            }
        }
        
        //Triple for loop for calculation (kmn)
        for(k = 0; k < K; k++){
            for(m = 0; m < M; m++){
                for(n = 0; n < N; n++){
                    C[m * N + n] += A[m * K + k] * B[k * N + n];
                }
            }
        }

    }

    void matmult_knm(int M, int N, int K, double *A, double *B, double *C) {

        int m, n, k;

        for(m = 0; m < M; m++){
            for(n = 0; n < N; n++){
                C[m * N + n] = 0;
            }
        }
        
        //Triple for loop for calculation (knm)
        for(k = 0; k < K; k++){
            for(n = 0; n < N; n++){
                for(m = 0; m < M; m++){
                    C[m * N + n] += A[m * K + k] * B[k * N + n];
                }
            }
        }
    }

    void matmult_nkm(int M, int N, int K, double *A, double *B, double *C) {

        int m, n, k;

        for(m = 0; m < M; m++){
            for(n = 0; n < N; n++){
                C[m * N + n] = 0;
            }
        }
        
        //Triple for loop for calculation (nkm)
        for(n = 0; n < N; n++){
            for(k = 0; k < K; k++){
                for(m = 0; m < M; m++){
                    C[m * N + n] += A[m * K + k] * B[k * N + n];
                }
            }
        }

    }

    void matmult_nmk(int M, int N, int K, double *A, double *B, double *C) {

        int m, n, k;

        for(m = 0; m < M; m++){
            for(n = 0; n < N; n++){
                C[m * N + n] = 0;
            }
        }
        
        //Triple for loop for calculation (nmk)
        for(n = 0; n < N; n++){
            for(m = 0; m < M; m++){
                for(k = 0; k < K; k++){
                    C[m * N + n] += A[m * K + k] * B[k * N + n];
                }
            }
        }
    }

    void matmult_blk(int M, int N, int K, double *A, double *B, double *C, int bs){
        int kk,nn,n,k,m;
        
        for(m = 0; m < M; m++){
            for(n = 0; n < N; n++){
                C[m * N + n] = 0;
            }
        }

        int m0, k0, n0;

        for (m0 = 0; m0 < M; m0 += bs)
        { int minm0=min(m0 + bs,M);
            for (k0 = 0; k0 < K; k0 += bs)
            { int mink0=min(k0 + bs,K);
                for (n0 = 0; n0 < N; n0 += bs)
                {  int minn0=min(n0 + bs,N);
                    for (m = m0; m < minm0; m++)
                    {
                        for (k = k0; k < mink0; k++)
                        {
                            for (n = n0; n <minn0; n++)
                            {
                                C[m * N + n] += A[m * K + k] * B[k * N + n];
                            }
                        }
                    }
                }
            }
        }
    }

}