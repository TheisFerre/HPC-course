#include <stdio.h>
#include <stdlib.h>
#include "matrix.h"

// allocate a double-prec m x n matrix
double ** 
dmalloc_2d(int m, int n) {
    if (m <= 0 || n <= 0) return NULL;
    double **A = malloc(m * sizeof(double *));
    if (A == NULL) return NULL;
    A[0] = malloc(m*n*sizeof(double));
    if (A[0] == NULL) {
        free(A); 
        return NULL; 
    }
    int i;
    for (i = 1; i < m; i++)
        A[i] = A[0] + i * n;
    return A;
}

void
free_2d(double **A) {
    free(A[0]);
    free(A);
}
