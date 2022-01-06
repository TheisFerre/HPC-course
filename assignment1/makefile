TARGET	= libmatmult.so
LIBSRCS	= matmult.c
LIBOBJS	= matmult.o matrix.o

OPT	= -g -funroll-loops #-O3 -funroll-loops  #basic -g #Comparison1 -g -03 -funroll-loops #Comparison2 -g -03 CHIP = -march=broadwell #Comparison3
PIC	= -fPIC

ISA = -mavx2
CHIP = "-march=broadwell"

CC	= gcc
CFLAGS= $(OPT) $(PIC) $(XOPTS) $(ISA) $(CHIP)

SOFLAGS = -shared 
XLIBS	= 

$(TARGET): $(LIBOBJS)
	$(CC) -o $@ $(SOFLAGS) $(LIBOBJS) $(XLIBS)


clean:
	@/bin/rm -f core core.* $(LIBOBJS) 
# DO NOT DELETE

matmult.o: matrix.h
matrix.o: matrix.h
