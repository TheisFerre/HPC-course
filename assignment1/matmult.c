#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <cblas.h>
#include "matrix.h"

void matmult_nat(int M, int N, int K, double **A, double **B, double **C) {

    int m, n, k;

    for(m = 0; m < M; m++){
        for(n = 0; n < N; n++){
            C[m][n] = 0;
        }
    }
    
    //Triple for loop for calculation (mkn)
    for(m = 0; m < M; m++){
        for(k = 0; k < K; k++){
            for(n = 0; n < N; n++){
                C[m][n] += A[m][k] * B[k][n];
            }
        }
    }
}

void matmult_lib(int M, int N, int K, double **A, double **B, double **C){

    // first dimension of A
    int lda = M;
    // first dimension of B
    int ldb = K;
    // first dimension of C
    int ldc = M;

    // call cblas library
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0, *A, lda, *B, ldb, 0.0, *C, ldc);
}

void matmult_mkn(int M, int N, int K, double **A, double **B, double **C) {

    int m, n, k;

    for(m = 0; m < M; m++){
        for(n = 0; n < N; n++){
            C[m][n] = 0;
        }
    }
    
    //Triple for loop for calculation (mkn)
    for(m = 0; m < M; m++){
        for(k = 0; k < K; k++){
            for(n = 0; n < N; n++){
                C[m][n] += A[m][k] * B[k][n];
            }
        }
    }
}

void matmult_mnk(int M, int N, int K, double **A, double **B, double **C) {

    int m, n, k;

    for(m = 0; m < M; m++){
        for(n = 0; n < N; n++){
            C[m][n] = 0;
        }
    }
    
    //Triple for loop for calculation (mnk)
    for(m = 0; m < M; m++){
        for(n = 0; n < N; n++){
            for(k = 0; k < K; k++){
                C[m][n] += A[m][k] * B[k][n];
            }
        }
    }

}

void matmult_kmn(int M, int N, int K, double **A, double **B, double **C) {

    int m, n, k;

    for(m = 0; m < M; m++){
        for(n = 0; n < N; n++){
            C[m][n] = 0;
        }
    }
    
    //Triple for loop for calculation (kmn)
    for(k = 0; k < K; k++){
        for(m = 0; m < M; m++){
            for(n = 0; n < N; n++){
                C[m][n] += A[m][k] * B[k][n];
            }
        }
    }

}

void matmult_knm(int M, int N, int K, double **A, double **B, double **C) {

    int m, n, k;

    for(m = 0; m < M; m++){
        for(n = 0; n < N; n++){
            C[m][n] = 0;
        }
    }
    
    //Triple for loop for calculation (knm)
    for(k = 0; k < K; k++){
        for(n = 0; n < N; n++){
            for(m = 0; m < M; m++){
                C[m][n] += A[m][k] * B[k][n];
            }
        }
    }
}

void matmult_nkm(int M, int N, int K, double **A, double **B, double **C) {

    int m, n, k;

    for(m = 0; m < M; m++){
        for(n = 0; n < N; n++){
            C[m][n] = 0;
        }
    }
    
    //Triple for loop for calculation (nkm)
    for(n = 0; n < N; n++){
         for(k = 0; k < K; k++){
            for(m = 0; m < M; m++){
                C[m][n] += A[m][k] * B[k][n];
            }
        }
    }

}

void matmult_nmk(int M, int N, int K, double **A, double **B, double **C) {

    int m, n, k;

    for(m = 0; m < M; m++){
        for(n = 0; n < N; n++){
            C[m][n] = 0;
        }
    }
    
    //Triple for loop for calculation (nmk)
    for(n = 0; n < N; n++){
        for(m = 0; m < M; m++){
            for(k = 0; k < K; k++){
                C[m][n] += A[m][k] * B[k][n];
            }
        }
    }
}

void matmult_blk(int M, int N, int K, double **A, double **B, double **C, int bs){
    int kk,nn,n,k,m;
    
    for(m = 0; m < M; m++){
        for(n = 0; n < N; n++){
            C[m][n] = 0;
        }
    }

/* Below implementation requires that the matrices are of NxN*/
    for(kk=0;kk<N;kk+=bs){
        for(nn=0;nn<K;nn+=bs){
            for(m=0;m<N;m++){
                for(k=nn;k<nn+bs;k++){
                    for(n=kk;n<kk+bs;n++){
                        C[m][n]+=A[m][k]*B[k][n];
                    }
                }
            }
        }
    }

}