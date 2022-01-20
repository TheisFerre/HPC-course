/* jacobi.c - Poisson problem in 3d
 * 
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "alloc3d.h"

__global__ void jacobi_d0(int N, double ***u_new,double ***u_old,double, ***u_other_dev, double***f) {
    double delta;
    double delta_sq;
    delta = 2.0/(N + 1.0);
    delta_sq = delta * delta;
    int z, y, x;
    double div = 1.0/6.0;

    x=blockIdx.x*blockDim.x+threadIdx.x+1;
    y=blockIdx.y*blockDim.y+threadIdx.y+1;
    z=blockIdx.z*blockDim.z+threadIdx.z+1;
    if (x>=(N+1) || y>=(N+1) || z>=(N+1))return;

    //perform Jacobi iterations
    if (z==(N/2)){
    u_new[z][y][x] = div * (u_old[z-1][y][x] + \
                        u_other_dev[0][y][x] + \
                        u_old[z][y-1][x] + \
                        u_old[z][y+1][x] + \
                        u_old[z][y][x-1] + \
                        u_old[z][y][x+1] + \
                        delta_sq * f[z][y][x]);
    }
    else{
    u_new[z][y][x] = div * (u_old[z-1][y][x] + \
                            u_old[z+1][y][x] + \
                            u_old[z][y-1][x] + \
                            u_old[z][y+1][x] + \
                            u_old[z][y][x-1] + \
                            u_old[z][y][x+1] + \
                            delta_sq * f[z][y][x]);
    }
}

__global__ void jacobi_d1(int N, double ***u_new,double ***u_old,double, ***u_other_dev, double***f) {
    double delta;
    double delta_sq;
    delta = 2.0/(N + 1.0);
    delta_sq = delta * delta;
    int z, y, x;
    double div = 1.0/6.0;

    x=blockIdx.x*blockDim.x+threadIdx.x+1;
    y=blockIdx.y*blockDim.y+threadIdx.y+1;
    z=blockIdx.z*blockDim.z+threadIdx.z+1;
    if (x>=(N+1) || y>=(N+1) || z>=(N+1))return;

    //perform Jacobi iterations
    if (z==0){
    u_new[z][y][x] = div * (u_other_dev[N/2][y][x] + \
                        u_old[z+1][y][x] + \
                        u_old[z][y-1][x] + \
                        u_old[z][y+1][x] + \
                        u_old[z][y][x-1] + \
                        u_old[z][y][x+1] + \
                        delta_sq * f[z][y][x]);
    }
    else{
    u_new[z][y][x] = div * (u_old[z-1][y][x] + \
                            u_old[z+1][y][x] + \
                            u_old[z][y-1][x] + \
                            u_old[z][y+1][x] + \
                            u_old[z][y][x-1] + \
                            u_old[z][y][x+1] + \
                            delta_sq * f[z][y][x]);
    }
}

