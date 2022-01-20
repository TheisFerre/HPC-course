#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define min(x,y) (((x) < (y)) ? (x) : (y))

extern "C" {
    #include <cblas.h>
    void matmult_nat(int M, int N, int K, double *A, double *B, double *C) {

        int m, n, k;

        for(m = 0; m < M; m++){
            for(n = 0; n < N; n++){
                C[m * N + n] = 0;
            }
        }
        
        //Triple for loop for calculation (mkn)
        for(m = 0; m < M; m++){
            for(k = 0; k < K; k++){
                for(n = 0; n < N; n++){
                    C[m * N + n] += A[m * K + k] * B[k * N + n];
                }
            }
        }
    }

    void matmult_lib(int M, int N, int K, double *A, double *B, double *C){

        // first dimension of A (m X K)
        int lda = K;
        // first dimension of B (K X N)
        int ldb = N;
        // first dimension of C (M X N)
        int ldc = N;

        double time, elapsed;

        time = omp_get_wtime();
        // call cblas library
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0, A, lda, B, ldb, 0.0, C, ldc);
        elapsed = omp_get_wtime() - time;
    }

    void matmult_mkn(int M, int N, int K, double *A, double *B, double *C) {

        int m, n, k;

        for(m = 0; m < M; m++){
            for(n = 0; n < N; n++){
                C[m * N + n] = 0;
            }
        }
        
        //Triple for loop for calculation (mkn)
        for(m = 0; m < M; m++){
            for(k = 0; k < K; k++){
                for(n = 0; n < N; n++){
                    C[m * N + n] += A[m * K + k] * B[k * N + n];
                }
            }
        }
    }

    void matmult_mnk(int M, int N, int K, double *A, double *B, double *C) {

        int m, n, k;

        for(m = 0; m < M; m++){
            for(n = 0; n < N; n++){
                C[m * N + n] = 0;
            }
        }
        
        //Triple for loop for calculation (mnk)
        for(m = 0; m < M; m++){
            for(n = 0; n < N; n++){
                for(k = 0; k < K; k++){
                    C[m * N + n] += A[m * K + k] * B[k * N + n];
                }
            }
        }

    }

    void matmult_kmn(int M, int N, int K, double *A, double *B, double *C) {

        int m, n, k;

        for(m = 0; m < M; m++){
            for(n = 0; n < N; n++){
                C[m * N + n] = 0;
            }
        }
        
        //Triple for loop for calculation (kmn)
        for(k = 0; k < K; k++){
            for(m = 0; m < M; m++){
                for(n = 0; n < N; n++){
                    C[m * N + n] += A[m * K + k] * B[k * N + n];
                }
            }
        }

    }

    void matmult_knm(int M, int N, int K, double *A, double *B, double *C) {

        int m, n, k;

        for(m = 0; m < M; m++){
            for(n = 0; n < N; n++){
                C[m * N + n] = 0;
            }
        }
        
        //Triple for loop for calculation (knm)
        for(k = 0; k < K; k++){
            for(n = 0; n < N; n++){
                for(m = 0; m < M; m++){
                    C[m * N + n] += A[m * K + k] * B[k * N + n];
                }
            }
        }
    }

    void matmult_nkm(int M, int N, int K, double *A, double *B, double *C) {

        int m, n, k;

        for(m = 0; m < M; m++){
            for(n = 0; n < N; n++){
                C[m * N + n] = 0;
            }
        }
        
        //Triple for loop for calculation (nkm)
        for(n = 0; n < N; n++){
            for(k = 0; k < K; k++){
                for(m = 0; m < M; m++){
                    C[m * N + n] += A[m * K + k] * B[k * N + n];
                }
            }
        }

    }

    void matmult_nmk(int M, int N, int K, double *A, double *B, double *C) {

        int m, n, k;

        for(m = 0; m < M; m++){
            for(n = 0; n < N; n++){
                C[m * N + n] = 0;
            }
        }
        
        //Triple for loop for calculation (nmk)
        for(n = 0; n < N; n++){
            for(m = 0; m < M; m++){
                for(k = 0; k < K; k++){
                    C[m * N + n] += A[m * K + k] * B[k * N + n];
                }
            }
        }
    }

    void matmult_blk(int M, int N, int K, double *A, double *B, double *C, int bs){
        int kk,nn,n,k,m;
        
        for(m = 0; m < M; m++){
            for(n = 0; n < N; n++){
                C[m * N + n] = 0;
            }
        }

        int m0, k0, n0;

        for (m0 = 0; m0 < M; m0 += bs)
        { int minm0=min(m0 + bs,M);
            for (k0 = 0; k0 < K; k0 += bs)
            { int mink0=min(k0 + bs,K);
                for (n0 = 0; n0 < N; n0 += bs)
                {  int minn0=min(n0 + bs,N);
                    for (m = m0; m < minm0; m++)
                    {
                        for (k = k0; k < mink0; k++)
                        {
                            for (n = n0; n <minn0; n++)
                            {
                                C[m * N + n] += A[m * K + k] * B[k * N + n];
                            }
                        }
                    }
                }
            }
        }
    }

}