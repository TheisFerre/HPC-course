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

N_VALUES="150"
MAX_ITER=10000
TOL=1
START_T=0

### OMP ENV VARS ###
export OMP_NUM_THREADS=12
export OMP_RUNTIME=dynamic,10
#export OMP_DISPLAY_ENV=verbose

rm -f ./poisson_jacobi_convergence.dat
rm -f ./poisson_gauss_convergence.dat
for N in $N_VALUES
do
    ./poisson_j $N $MAX_ITER $TOL $START_T >> poisson_jacobi_convergence.dat
    ./poisson_gs $N $MAX_ITER $TOL $START_T >> poisson_gauss_convergence.dat
done


exit 0