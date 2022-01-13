#!/bin/bash
#BSUB -J jacobi_batch
#BSUB -o jacobi_batch_%J.out
#BSUB -e jacobi_batch_%J.err
#BSUB -q hpcintro
#BSUB -n 1
#BSUB -R "rusage[mem=8096]"
#BSUB -W 20
#BSUB -N
# uncomment the following line, if you want to assure that your job has
# a whole CPU for itself (shared L3 cache)
#BSUB -R "span[hosts=1] affinity[socket(1)]"

# Check cache
echo "checking CPU info"
CACHE_SIZE=$(lscpu) 
echo ${CACHE_SIZE}

CC=${1-"gcc"}

N_VALUES="30 60 90 120 150 180 210 240 270 300"
MAX_ITER=1100
TOL=0.01
START_T=4

### OMP ENV VARS ###
#export OMP_NUM_THREADS=1
#export OMP_RUNTIME=dynamic,10
#export OMP_DISPLAY_ENV=verbose

rm -f ./poisson_jacobi_runtime.dat
for N in $N_VALUES
do
    start=$(date +%s.%N)
    ./poisson_j $N $MAX_ITER $TOL $START_T >> poisson_jacobi_runtime.dat
    end=$(date +%s.%N)
    runtime=$(echo "$end - $start" | bc)
    printf " $runtime " >> poisson_runtime.dat
    printf "$N" >> poisson_runtime.dat
    printf "\n" >> poisson_runtime.dat
    
    #echo "\n" >> poisson_runtime.dat
done


exit 0