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

# gpu1 and gpu2 kernels vs CPU DGEMM (CPU <-> GPU Bandwith)

module load cuda/11.5.1

MATRIX_SIZES="8 16 32 64 128 256 512 1024 2048"

export MATMULT_RESULT=0
export MATMULT_COMPARE=0
export MFLOPS_MAX_IT=100
export MKL_NUM_THREADS=16

# using all threads
#MKL_NUM_THREADS=16 numactl --cpunodebind=0

# using a single thread
#MKL_NUM_THREADS=1 numactl --cpunodebind=0
for size in $MATRIX_SIZES
do
    rm -f results/test_1_mult_$size.txt

    ./matmult_f.nvcc gpu2 $size $size $size >> results/test_1_mult_$size.txt
    numactl --cpunodebind=0 ./matmult_f.nvcc lib $size $size $size >> results/test_1_mult_$size.txt
done
