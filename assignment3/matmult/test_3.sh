#!/bin/bash
#BSUB -J test1
#BSUB -o test1_%J.out
#BSUB -e test1_%J.err
#BSUB -q hpcintrogpu
#BSUB -R "rusage[mem=8096]"
#BSUB -n 16 #IF USING ALL THREADS POSSIBLE
#BSUB -W 60
#BSUB -N
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "span[hosts=1]"

# gpu5 compared to CPU DGEMM and the other gpu_# kernels

MATRIX_SIZES="8 16 32 64 128 256 512 1024 2048 4096"

export MATMULT_RESULT=0
export MATMULT_COMPARE=0
export MKL_NUM_THREADS=16
export MFLOPS_MAX_IT=100
rm -f results/test_3.txt
for size in $MATRIX_SIZES
do
    ./matmult_f.nvcc gpulib $size $size $size >> results/test_3.txt
    numactl --cpunodebind=0 ./matmult_f.nvcc lib $size $size $size >> results/test_3.txt
done

