TARGET	= libmatmult.so
LIBSRCS	= matmult.c
LIBOBJS	= matmult.o matrix.o

OPT	= -g -O3
PIC	= -fPIC

CC	= gcc
CFLAGS= $(OPT) $(PIC) $(XOPTS)

SOFLAGS = -shared 
XLIBS	= 

$(TARGET): $(LIBOBJS)
	$(CC) -o $@ $(SOFLAGS) $(LIBOBJS) $(XLIBS)


clean:
	@/bin/rm -f core core.* $(LIBOBJS) 
# DO NOT DELETE

matmult.o: matrix.h
matrix.o: matrix.h
