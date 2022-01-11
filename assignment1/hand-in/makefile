TARGET	= libmatmult.so
LIBSRCS	= matmult.c
LIBOBJS	= matmult.o #matrix.o

OPT	= -g -Ofast -funroll-loops #basic -g #unroll -g -funroll-loops #fast -g -Ofast #fast-unroll -g -Ofast -funroll-loops
PIC	= -fPIC

ISA = -mavx2 #vectorization
CHIP = "-march=broadwell" #optimize for our chip

CC	= gcc
CFLAGS= $(OPT) $(PIC) $(XOPTS) $(ISA) $(CHIP)

SOFLAGS = -shared 
XLIBS	= 

$(TARGET): $(LIBOBJS)
	$(CC) -o $@ $(SOFLAGS) $(LIBOBJS) $(XLIBS)


clean:
	@/bin/rm -f core core.* $(LIBOBJS) 
# DO NOT DELETE
