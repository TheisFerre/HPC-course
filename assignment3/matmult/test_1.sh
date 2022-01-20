# gpu1 and gpu2 kernels vs CPU DGEMM (CPU <-> GPU Bandwith)

MATRIX_SIZES=[8 16 32 64 128 256 512 1024 2048 4096]

export MATMULT_RESULT=0
export MATMULT_COMPARE=0
for size in $MATRIX_SIZES
do
    ./matmult_f.nvcc gpu1 $size $size $size >> test_1.txt
    ./matmult_f.nvcc gpu2 $size $size $size >> test_1.txt
    ./matmult_f.nvcc dgemm $size $size $size >> test_1.txt
done
