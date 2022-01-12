/* jacobi.c - Poisson problem in 3d
 * 
 */
#include <math.h>
#include <stdio.h>
#include "alloc3d.h"

double frob_norm(int N, double ***u_new, double ***u_old){
    // check tolerance
    double d = 0;
    double diff = 0;
    for(int z=1;z<N+1;z++){
        for(int y=1;y<N+1;y++){
            for(int x=1;x<N+1;x++){
                diff = u_new[z][y][x] - u_old[z][y][x];
                d += (diff) * (diff);
            }
        }
    }
    d = sqrt(d);
    printf("%f\n", d);
    return d;
}

int
jacobi(int N, int iter_max, double tolerance, double ***u_new, double***f) {
    // fill in your code here
    double delta;
    delta = 2.0/(N + 1.0);

    //create another tmp array
    double 	***u_old;
    u_old = d_malloc_3d(N+2, N+2, N+2);
    for(int z=0;z<N+2;z++){
        for(int y=0;y<N+2;y++){
            for(int x=0;x<N+2;x++){
                u_old[z][y][x] = u_new[z][y][x];
            }
        }
    }
    double d = 100000;
    int k = 0;
    while (d > tolerance && k < iter_max){
        for(int z=1;z<N+1;z++){
            for(int y=1;y<N+1;y++){
                for(int x=1;x<N+1;x++){
                    u_new[z][y][x] = 1/6 * (u_old[z-1][y][x] + \
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

        // update u
        for(int z=1;z<N+1;z++){
            for(int y=1;y<N+1;y++){
                for(int x=1;x<N+1;x++){
                    // if (u_old[z][y][x] != u_new[z][y][x]) {
                    //     printf("Diff!\n");
                    // }
                    u_old[z][y][x] = u_new[z][y][x];
                }
            }
        }
        k += 1;
    }


    // printing...
    // printf("\n\n");
    // for(int z=0;z<N+2;z++)
    //     for(int y=0;y<N+2;y++)
    //         for(int x=0;x<N+2;x++)
    //             printf("%.2f ",u_new[z][y][x]);
    return 0;
}
