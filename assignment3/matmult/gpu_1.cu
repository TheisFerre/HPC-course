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