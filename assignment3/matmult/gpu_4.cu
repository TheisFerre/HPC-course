
#define THREAD_COMPUTE 4 // number of c elements a thread should compute
__global__ void gpu4_kernel(int M, int N, int K, double *A_d, double *B_d, double *C_d)
{

    int ROW = (blockIdx.y * blockDim.y + threadIdx.y) * THREAD_COMPUTE;
    int COL = blockIdx.x * blockDim.x + threadIdx.x;

    // A = M X K
    // B = K X N
    // C = M X N

    // TODO: split into if-else blocks
    int t;
    for (t = 0; t < THREAD_COMPUTE; t++)
    {
        int i, j;
        double sum_val = 0;
        if ((ROW + t) < M && COL < N)
        {
            for (i = 0; i < K; i++)
            {
                sum_val += A_d[(ROW + t) * K + i] * B_d[i * N + COL];
            }
            C_d[(ROW + t) * N + COL] = sum_val;
        }
    }
}

extern "C"
{
    void matmult_gpu4(int M, int N, int K, double *A_h, double *B_h, double *C_h)
    {
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
        int xSize = ceil((double)N / (double)dimBlock.x);
        int ySize = ceil((double)M / (double)dimBlock.y / (double)THREAD_COMPUTE);

        dim3 dimGrid(xSize, ySize);
        gpu4_kernel<<<dimGrid, dimBlock>>>(M, N, K, A_d, B_d, C_d);
        cudaDeviceSynchronize();

        cudaMemcpy(C_h, C_d, C_size, cudaMemcpyDeviceToHost);
        cudaFree(A_d);
        cudaFree(B_d);
        cudaFree(C_d);
    }
}