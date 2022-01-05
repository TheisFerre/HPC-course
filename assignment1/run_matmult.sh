#!/bin/sh

CC=${1-"gcc"}

NPARTS = "10 1000 2000 3000 4000 5000 7500 10000 20000 40000 80000 200000 400000 800000 1200000 1600000 3000000"
PERM="nat lib mkn mnk kmn knm nkm nmk"
LOGEXT=$CC.dat

/bin/rm -f aos.$LOGEXT soa.$LOGEXT

for perm in $PERM
do
for size in $NPARTS
do
    ./matmult_c.${CC} $perm $size $size $size >> matmult_c.$LOGEXT
done
done


# time to say 'Good bye' ;-)
#
exit 0