/* jacobi.c - Poisson problem in 3d
 * 
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "alloc3d.h"
#include <helper_cuda.h>
__inline__ __device__ double warpReduceSum(double value) { 
    for (int i = 16; i > 0; i /= 2) 
        value += __shfl_down_sync(-1, value, i);  
    return value; 
} 

__inline__ __device__ double blockReduceSum(double value) { 
    __shared__ double smem[32]; // Max 32 warp sums 
 
    if (threadIdx.x < warpSize) 
        smem[threadIdx.x] = 0; 
    __syncthreads(); 
 
    value = warpReduceSum(value); 
 
    if (threadIdx.x % warpSize == 0) 
        smem[threadIdx.x / warpSize] = value; 
    __syncthreads(); 
 
    if (threadIdx.x < warpSize) 
        value = smem[threadIdx.x]; 
    return warpReduceSum(value); 
} 



__global__ void jacobi(int N, double ***u_new,double ***u_old, double***f,double *fbnorm) {
    double delta;
    double delta_sq;
    delta = 2.0/(N + 1.0);
    delta_sq = delta * delta;
    int z, y, x;
    double div = 1.0/6.0;
    double tmp;
    x=blockIdx.x*blockDim.x+threadIdx.x+1;
    y=blockIdx.y*blockDim.y+threadIdx.y+1;
    z=blockIdx.z*blockDim.z+threadIdx.z+1;
    if (x>=(N+1) || y>=(N+1) || z>=(N+1))return;

    //perform Jacobi iterations
    u_new[z][y][x] = div * (u_old[z-1][y][x] + \
                            u_old[z+1][y][x] + \
                            u_old[z][y-1][x] + \
                            u_old[z][y+1][x] + \
                            u_old[z][y][x-1] + \
                            u_old[z][y][x+1] + \
                            delta_sq * f[z][y][x]);

    
    // *fbnorm=1.0;
    tmp=(u_new[z][y][x]-u_old[z][y][x])*(u_new[z][y][x]-u_old[z][y][x]);
    tmp = blockReduceSum(tmp); 
    if (threadIdx.x == 1 && threadIdx.y==1 && threadIdx.z==1) 
        atomicAdd(fbnorm, tmp);
        *fbnorm=sqrt(*fbnorm);
    //sum reduction in cuda sum Til én variable fbnorm 1 adresse, data race    
}


