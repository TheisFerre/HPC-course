#!/bin/sh

CC=${1-"gcc"}

NPARTS = "10 200 500 1000 2000 3000 4000 5000 7500 10000 20000"
PERM="nat lib mkn mnk kmn knm nkm nmk"
LOGEXT=$CC.dat

/bin/rm -f matmult_c.$LOGEXT

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