
__global__ void gpu3_kernel(int M, int N, int K, double *A_d, double *B_d, double *C_d)
{

    int ROW1 = (blockIdx.y * blockDim.y + threadIdx.y) * 2;
    int ROW2 = ROW1 + 1;
    int COL = blockIdx.x * blockDim.x + threadIdx.x;

    // A = M X K
    // B = K X N
    // C = M X N

    if (ROW2 < M && COL < N)
    {
        int i, j;
        double sum_val1 = 0;
        double sum_val2 = 0;
        for (i = 0; i < K; i++)
        {
            sum_val1 += A_d[ROW1 * K + i] * B_d[i * N + COL];
            sum_val2 += A_d[ROW2 * K + i] * B_d[i * N + COL];
        }
        C_d[ROW1 * N + COL] = sum_val1;
        C_d[ROW2 * N + COL] = sum_val2;
    }

    else if (ROW1 < M && COL < N)
    {
        int i, j;
        double sum_val1 = 0;
        for (i = 0; i < K; i++)
        {
            sum_val1 += A_d[ROW1 * K + i] * B_d[i * N + COL];
        }
        C_d[ROW1 * N + COL] = sum_val1;
    }
}
extern "C"
{
    void matmult_gpu3(int M, int N, int K, double *A_h, double *B_h, double *C_h)
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
        int BLOCK_SIZE = 4;
        dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
        // int xSize = (N + dimBlock.x - 1) / dimBlock.x;
        // int ySize = (M + dimBlock.y - 1) / dimBlock.y;

        // which dimension to half
        // C: M X N
        int xSize = ceil((double)N / (double)dimBlock.x);
        int ySize = ceil((double)M / (double)dimBlock.y / 2.0);

        dim3 dimGrid(xSize, ySize);
        gpu3_kernel<<<dimGrid, dimBlock>>>(M, N, K, A_d, B_d, C_d);
        cudaDeviceSynchronize();

        cudaMemcpy(C_h, C_d, C_size, cudaMemcpyDeviceToHost);
        cudaFree(A_d);
        cudaFree(B_d);
        cudaFree(C_d);
    }
}