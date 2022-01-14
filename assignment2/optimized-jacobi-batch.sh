#!/bin/bash
#BSUB -J jacobi_batch
#BSUB -o jacobi_batch_%J.out
#BSUB -e jacobi_batch_%J.err
#BSUB -n 24
#BSUB -R "rusage[mem=8096]"
#BSUB -q hpcintro
#BSUB -R "select[model == XeonE5_2650v4]"
#BSUB -W 60
#BSUB -N
# uncomment the following line, if you want to assure that your job has
# a whole CPU for itself (shared L3 cache)
#BSUB -R "span[hosts=1]"

# Check cache
echo "checking CPU info"
CACHE_SIZE=$(lscpu) 
echo ${CACHE_SIZE}

CC=${1-"gcc"}

THREADS="1 2 3 4 5 6 7 8 9 10 11 12"

N_VAL=50
MAX_ITER=150
TOL=0.01
START_T=4

### OMP ENV VARS ###
#export OMP_NUM_THREADS=1
export OMP_RUNTIME=dynamic
#export OMP_PLACES=cores
#export OMP_PROC_BIND=spread
#export OMP_DISPLAY_ENV=verbose

rm -f ./jacobi-optimized-runtime.dat
for thread in $THREADS
do
    export OMP_NUM_THREADS=$thread
    printf "$thread " >> jacobi-optimized-runtime.dat
    ./poisson_j $N_VAL $MAX_ITER $TOL $START_T >> jacobi-optimized-runtime.dat
done


exit 0