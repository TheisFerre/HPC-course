#!/bin/bash
#BSUB -J jacobi_batch
#BSUB -o jacobi_batch_%J.out
#BSUB -e jacobi_batch_%J.err
#BSUB -n 4
#BSUB -R "rusage[mem=8096]"
#BSUB -q hpcintro
#BSUB -R "select[model == XeonE5_2650v4]"
#BSUB -W 60
#BSUB -N
# uncomment the following line, if you want to assure that your job has
# a whole CPU for itself (shared L3 cache)
#BSUB -R "span[hosts=1] affinity[socket(1)]"

# Check cache
echo "checking CPU info"
CACHE_SIZE=$(lscpu) 
echo ${CACHE_SIZE}

CC=${1-"gcc"}

THREADS="1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16"

N_VAL=500
MAX_ITER=150
TOL=0.01
START_T=4

### OMP ENV VARS ###
#export OMP_NUM_THREADS=1
export OMP_RUNTIME=dynamic,10
#export OMP_DISPLAY_ENV=verbose

rm -f ./jacobi-thread-runtime.dat
rm -f ./gauss-thread-runtime.dat
for thread in $THREADS
do
    export OMP_NUM_THREADS=$thread
    start=$(date +%s.%N)
    ./poisson_j $N_VAL $MAX_ITER $TOL $START_T
    end=$(date +%s.%N)
    runtime=$(echo "$end - $start" | bc)
    printf "$thread" >> jacobi-thread-runtime.dat
    printf " $runtime " >> jacobi-thread-runtime.dat
    printf "\n" >> jacobi-thread-runtime.dat

    start=$(date +%s.%N)
    ./poisson_gs $N_VAL $MAX_ITER $TOL $START_T
    end=$(date +%s.%N)
    runtime=$(echo "$end - $start" | bc)
    printf "$thread" >> gauss-thread-runtime.dat
    printf " $runtime " >> gauss-thread-runtime.dat
    printf "\n" >> gauss-thread-runtime.dat
done


exit 0