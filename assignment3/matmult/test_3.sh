# gpu5 compared to CPU DGEMM and the other gpu_# kernels

MATRIX_SIZES=[8 16 32 64 128 256 512 1024 2048 4096]

export MATMULT_RESULT=0
export MATMULT_COMPARE=0
for size in $MATRIX_SIZES
do
    ./matmult_f.nvcc gpu1 $size $size $size >> test_3.txt
    ./matmult_f.nvcc gpu2 $size $size $size >> test_3.txt
    ./matmult_f.nvcc gpu3 $size $size $size >> test_3.txt
    ./matmult_f.nvcc gpu4 $size $size $size >> test_3.txt
    ./matmult_f.nvcc gpu5 $size $size $size >> test_3.txt
    MKL_NUM_THREADS=16 numactl --cpunodebind=0 ./matmult_f.nvcc lib $size $size $size >> test_1.txt
done

