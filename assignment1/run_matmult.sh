#!/bin/bash
#BSUB -J mm_batch
#BSUB -o mm_batch_%J.out
#BSUB -e mm_batch_%J.err
#BSUB -q hpcintro
#BSUB -n 1
#BSUB -R "rusage[mem=8096]"
#BSUB -W 60
#BSUB -N
# uncomment the following line, if you want to assure that your job has
# a whole CPU for itself (shared L3 cache)
#BSUB -R "span[hosts=1] affinity[socket(1)]"

# LOAD OPENBLAS
#module load openblas

# Check cache
echo "checking CPU info"
CACHE_SIZE=$(lscpu)
echo ${CACHE_SIZE}

CC=${1-"gcc"}

NPARTS="10 15 20 35 50 75 100 150 200 300 500 750 1000 1400 1800 2000 2300 2600 3000"
PERM="mkn mnk kmn knm nkm nmk"
LOGEXT=$CC.dat

# set to "OPT" if running with optimizations
OPT=funroll

# enable(1)/disable(0) result checking
export MATMULT_COMPARE=0

for perm in $PERM
do
/bin/rm -f run_data/${perm}_${OPT}_matmult_c.$LOGEXT
for size in $NPARTS
do
    ./matmult_c.${CC} $perm $size $size $size >> run_data/${perm}_${OPT}_matmult_c.$LOGEXT
done
done


# time to say 'Good bye' ;-)
#
exit 0