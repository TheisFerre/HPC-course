/* jacobi.c - Poisson problem in 3d
 * 
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "alloc3d.h"

__global__ void jacobi(int N, double ***u_new,double ***u_old, double***f) {
    double delta;
    double delta_sq;
    delta = 2.0/(N + 1.0);
    delta_sq = delta * delta;
    int z, y, x;
    double div = 1.0/6.0;

    x=blockIdx.x*blockDim.x+threadIdx.x;
    y=blockIdx.y*blockDim.y+threadIdx.y;
    z=blockIdx.z*blockDim.z+threadIdx.z;
    if (x>=N || y>=N || z>=N){return;}
    //perform Jacobi iterations
    u_new[z][y][x] = div * (u_old[z-1][y][x] + \
                            u_old[z+1][y][x] + \
                            u_old[z][y-1][x] + \
                            u_old[z][y+1][x] + \
                            u_old[z][y][x-1] + \
                            u_old[z][y][x+1] + \
                            delta_sq * f[z][y][x]);
}
