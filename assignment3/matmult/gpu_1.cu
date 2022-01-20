#include <omp.h>
#include <stdio.h>
#include <helper_cuda.h>
__global__ void gpu1_kernel(int M, int N, int K, double *A_d, double *B_d, double *C_d)
{

    // single thread to compute all data
    int m, n, k;
    
    for (n = 0; n < N; n++)
    {
        for (m = 0; m < M; m++)
        {
            double sum_val = 0;
            for (k = 0; k < K; k++)
            {
                 sum_val += A_d[m * K + k] * B_d[k * N + n];
            }
            C_d[m * N + n] = sum_val;
        }   
    }
}
extern "C"
{
    void matmult_gpu1(int M, int N, int K, double *A_h, double *B_h, double *C_h)
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
        cudaMemcpy(C_d, C_h, C_size, cudaMemcpyHostToDevice);
        transfer_elabsed = omp_get_wtime() - transfer_time;

        // Initialize number of blocks and threads
        dim3 THREADS_BLOCK(1, 1);
        dim3 GRIDSIZE(1, 1);

        time = omp_get_wtime();
 
        gpu1_kernel<<<GRIDSIZE, THREADS_BLOCK>>>(M, N, K, A_d, B_d, C_d);
        checkCudaErrors(cudaDeviceSynchronize());
        
        elapsed = omp_get_wtime() - time;

        transfer_time = omp_get_wtime();
        cudaMemcpy(C_h, C_d, C_size, cudaMemcpyDeviceToHost);
        transfer_elabsed += omp_get_wtime() - transfer_time;

        //printf("Kernel_time\t");
        //printf("Transfer_time\n");
        printf("%f\t%f\n", elapsed, transfer_elabsed);

        cudaFree(A_d);
        cudaFree(B_d);
        cudaFree(C_d);
    }
}