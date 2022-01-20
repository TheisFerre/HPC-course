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