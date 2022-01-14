/* jacobi.c - Poisson problem in 3d
 * 
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "alloc3d.h"

#ifdef JACOBI_OPT
int jacobi(int N, int iter_max, double tolerance, double ***u_new, double***f) {
    //fill in your code here
    //printf("OPT\n");
    double delta;
    double delta_sq;
    delta = 2.0/(N + 1.0);
    delta_sq = delta * delta;

    //create another tmp array
    double 	***u_old;
    u_old = d_malloc_3d(N+2, N+2, N+2);
    
    double fbnorm = 1000000;
    int k = 0;
    int z, y, x;
    double tol = tolerance;
    double div = 1.0/6.0;
    //printf("Max iterations:\t%d\n", iter_max);
    //printf("Tolerance:\t%f\n", tolerance);
    // TOTAL FLOPS: (10 * N * N * N + FLOPS 2 * N * N * N) * iter_max
    while (fbnorm > tol && k < iter_max){
        // update u
        fbnorm=0;
        #pragma omp parallel private(z, y, x) shared(delta, f, u_old)
        {
        #pragma omp for collapse(2)
        for(z=0;z<N+2;z++){
            for(y=0;y<N+2;y++){
                for(x=0;x<N+2;x++){
                    // if (u_old[z][y][x] != u_new[z][y][x]) {
                    //     printf("Diff!\n");
                    // }
                    u_old[z][y][x] = u_new[z][y][x];
                }
            }
        }

        // LUPS 1 * N * N * N #LatticeUpdatesPrSecond
        #pragma omp for collapse(2) reduction(+:fbnorm)
        for(z=1;z<N+1;z++){
            for(y=1;y<N+1;y++){
                for(x=1;x<N+1;x++){
                    u_new[z][y][x] = div * (u_old[z-1][y][x] + \
                                            u_old[z+1][y][x] + \
                                            u_old[z][y-1][x] + \
                                            u_old[z][y+1][x] + \
                                            u_old[z][y][x-1] + \
                                            u_old[z][y][x+1] + \
                                            delta_sq * f[z][y][x]);
                    fbnorm+=(u_new[z][y][x]-u_old[z][y][x])*(u_new[z][y][x]-u_old[z][y][x]);
                }
            }
        }
        }
        fbnorm = sqrt(fbnorm);
        //printf("%f\n", fbnorm);

        k++;

        //printf("Iteration no. %i\t Norm: %f\n", k, d);
        
    }

    // for runtime plot
    //printf("%i", k);

    free(u_old);
    return 0;
}
#else
double frob_norm(int N, double ***u_new, double ***u_old){
    // check tolerance
    double d = 0;
    double diff = 0;
    int z, y, x;
    // FLOPS 2 * N * N * N
    for(z=1;z<N+1;z++){
        for(y=1;y<N+1;y++){
            for(x=1;x<N+1;x++){
                diff = u_new[z][y][x] - u_old[z][y][x];
                d += (diff) * (diff);
            }
        }
    }
    d = sqrt(d);
    //printf("%f\n", d);
    return d;
}

int
jacobi(int N, int iter_max, double tolerance, double ***u_new, double***f) {
    // fill in your code here
    //printf("NON-OPT\n");
    double delta;
    delta = 2.0/(N + 1.0);

    //create another tmp array
    double 	***u_old;
    u_old = d_malloc_3d(N+2, N+2, N+2);
    
    double d = INFINITY;
    int k = 0;
    int z, y, x;
    double tol = tolerance;
    //printf("Max iterations:\t%d\n", iter_max);
    //printf("Tolerance:\t%f\n", tolerance);
    // TOTAL FLOPS: (10 * N * N * N + FLOPS 2 * N * N * N) * iter_max
    while (d > tol && k < iter_max){
        // update u
        for(z=0;z<N+2;z++){
            for(y=0;y<N+2;y++){
                for(x=0;x<N+2;x++){
                    // if (u_old[z][y][x] != u_new[z][y][x]) {
                    //     printf("Diff!\n");
                    // }
                    u_old[z][y][x] = u_new[z][y][x];
                }
            }
        }

        // LUPS 1 * N * N * N #LatticeUpdatesPrSecond
        double div = 1.0/6.0;
        #pragma omp parallel for private(z, y, x) shared(delta, f, u_old)
        for(z=1;z<N+1;z++){
            for(y=1;y<N+1;y++){
                for(x=1;x<N+1;x++){
                    u_new[z][y][x] = div * (u_old[z-1][y][x] + \
                                            u_old[z+1][y][x] + \
                                            u_old[z][y-1][x] + \
                                            u_old[z][y+1][x] + \
                                            u_old[z][y][x-1] + \
                                            u_old[z][y][x+1] + \
                                            delta * delta * f[z][y][x]);
                }
            }
        }
        d = frob_norm(N, u_new, u_old);
        //printf("%f\n", d);

        k++;

        //printf("Iteration no. %i\t Norm: %f\n", k, d);
        
    }
    //long double flops;
    //flops = (N * N * N) * k / 1000000;
    //printf("%Lf", flops);

    // for runtime plot
    //printf("%i", k);

    // printing...
    // printf("\n\n");
    // for(int z=0;z<N+2;z++)
    //     for(int y=0;y<N+2;y++)
    //         for(int x=0;x<N+2;x++)
    //             printf("%.2f ",u_new[z][y][x]);
    free(u_old);
    return 0;
}
#endif


