#!/bin/bash
#BSUB -J test3
#BSUB -o test3_%J.out
#BSUB -e test3_%J.err
#BSUB -q hpcintrogpu
#BSUB -R "rusage[mem=8096]"
#BSUB -n 16 #IF USING ALL THREADS POSSIBLE
#BSUB -W 60
#BSUB -N
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "span[hosts=1]"

# gpu5 compared to CPU DGEMM and the other gpu_# kernels

MATRIX_SIZES="64 128 256 512 1024 2048 4096"

export MATMULT_RESULT=0
export MATMULT_COMPARE=0
export MKL_NUM_THREADS=16
export MFLOPS_MAX_IT=100

for size in $MATRIX_SIZES
do
    rm -f results/test_3_$size.txt
    ./matmult_f.nvcc gpulib $size $size $size >> results/test_3_$size.txt
    ./matmult_f.nvcc lib $size $size $size >> results/test_3_$size.txt
done

