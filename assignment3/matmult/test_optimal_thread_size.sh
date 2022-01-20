#!/bin/bash
#BSUB -J test1
#BSUB -o test1_%J.out
#BSUB -e test1_%J.err
#BSUB -q hpcintrogpu
#BSUB -R "rusage[mem=8096]"
#BSUB -W 60
#BSUB -N
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "span[hosts=1]"

# gpu1 and gpu2 kernels vs CPU DGEMM (CPU <-> GPU Bandwith)

module load gcc/10.3.0-binutils-2.36.1
module load cuda/11.5.1



size="2048" #16 32 64 128 256 512 1024 2048 4096"
THREAD_COMPUTE="1 2 3 4 5 6 7 8"

export MATMULT_RESULT=0
export MATMULT_COMPARE=0

rm -f results/test_compare_$size.txt

#MKL_NUM_THREADS=16 numactl --cpunodebind=0 
for T_COMPUTE in $THREAD_COMPUTE
do
    make clean
    make ARG=$T_COMPUTE

    ./matmult_f.nvcc gpu4 $size $size $size >> results/test_compare_$size.txt
done

