PERM="nat lib mkn mnk kmn knm nkm nmk"
SIZE=3
for perm in $PERM
do
    ./matmult_c.${CC} $perm $SIZE $SIZE $SIZE | grep -v CPU "#" >> matmult_c.$LOGEXT
done

echo "done testing matrix multiplications"
exit 0
