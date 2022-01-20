/* main.c - Poisson problem in 3D
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include "math.h"
#include "print.h"
#include "alloc3d_gpu.h"
#include "alloc3d.h"
#include "transfer3d_gpu.h"
#include "jacobi.h"
#include <helper_cuda.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#define N_DEFAULT 100

int
main(int argc, char *argv[]) {

    int     N = N_DEFAULT;
    int 	iter_max = 1000;
    double	tolerance;
    double	start_T;
    int		output_type = 0;
    char	*output_prefix = "poisson_res";
    char        *output_ext    = "";
    char	output_filename[FILENAME_MAX];
    double 	***u_h = NULL;
    double  ***u_old_d = NULL;
    double  ***u_new_d = NULL; 
    double  ***f_h = NULL;
    double  ***f_d = NULL;



    /* get the paramters from the command line */
    N = atoi(argv[1]);	// grid size
    iter_max  = atoi(argv[2]);  // max. no. of iterations
    tolerance = atof(argv[3]);  // tolerance
    start_T   = atof(argv[4]);  // start T for all inner grid points
    if (argc == 6) {
	output_type = atoi(argv[5]);  // ouput type
    }

    /////////////////////  ALLOCATE MEMORY /////////////////////
    // allocate host memory
    if ( (u_h = d_malloc_3d(N+2, N+2, N+2)) == NULL ) {
        perror("array u: allocation failed");
        exit(-1);
    }
    if ( (f_h = d_malloc_3d(N+2, N+2, N+2)) == NULL ) {
        perror("array f: allocation failed");
        exit(-1);
    }

    // allocate device memory
    if ( (u_old_d = d_malloc_3d_gpu(N+2, N+2, N+2)) == NULL ) {
        perror("array u: allocation failed");
        exit(-1);
    }
        if ( (u_new_d = d_malloc_3d_gpu(N+2, N+2, N+2)) == NULL ) {
        perror("array u: allocation failed");
        exit(-1);
    }

    if ( (f_d = d_malloc_3d_gpu(N+2, N+2, N+2)) == NULL ) {
        perror("array u: allocation failed");
        exit(-1);
    }

    /////////////////////  DATA INITIALISATION /////////////////////
    // /* Initialization of inner point in u*/
    for(int z=1;z<N+1;z++)
        for(int y=1;y<N+1;y++)
            for(int x=1;x<N+1;x++)
                u_h[z][y][x]=start_T;
    /* Initialization of boundary points in u ~ wall(x,y)*/
    for(int y=0;y<N+2;y++){
        for(int x=0;x<N+2;x++){
            u_h[0][y][x]=20;
        }
    }   
    for(int y=0;y<N+2;y++){
        for(int x=0;x<N+2;x++){
            u_h[N+1][y][x]=20;
        }
    }
    /* Wall (y,z) */
    for(int z=0;z<N+2;z++){
        for(int y=0;y<N+2;y++){
            u_h[z][y][0]=20;
        }
    }  
    for(int z=0;z<N+2;z++){
        for(int y=0;y<N+2;y++){
            u_h[z][y][N+1]=20;
        }
    }         
    /* Wall (x,z) */
    for(int z=0;z<N+2;z++){
        for(int x=0;x<N+2;x++){
            u_h[z][0][x]=0;
        }
    }       
    for(int z=0;z<N+2;z++){
        for(int x=0;x<N+2;x++){
            u_h[z][N+1][x]=20;
        }
    }     
    /* initialize f */
    double delta;
    delta = 2.0/(N + 1.0);
    for(int z=0;z<N+2;z++){
        for(int y=0;y<N+2;y++){
            for(int x=0;x<N+2;x++){
                if (-1 + delta * x <= -3.0/8.0 && -1.0 + delta * y <= -1.0/2.0 && -1 + delta * z >= -2.0/3.0 && -1.0 + delta * z <= 0) {
                    f_h[z][y][x] = 200;
                }
                else{
                    f_h[z][y][x] = 0;
                }
            }
        }
    }

    /////////////////////  COPY DATA FROM HOST TO DEVICE /////////////////////
    transfer_3d(u_new_d, u_h, N+2, N+2, N+2, cudaMemcpyHostToDevice);
    transfer_3d(u_old_d, u_h, N+2, N+2, N+2, cudaMemcpyHostToDevice);
    transfer_3d(f_d, f_h, N+2, N+2, N+2, cudaMemcpyHostToDevice);


    double*** temp = NULL;
    dim3 dimGrid(ceil(N/32.0),ceil(N/4.0),ceil(N/4.0));
    dim3 dimBlock(32,4,4);
    double *fbnorm_h, *fbnorm_d;
    double k=0;
    cudaMallocHost((void **) &fbnorm_h ,sizeof(double));
    cudaMalloc((void **) &fbnorm_d ,sizeof(double));
    *fbnorm_h=100;
    /////////////////////////////////  COMPUTE ///////////////////////////////
    while (*fbnorm_h > tolerance && k < iter_max){
        jacobi<<<dimGrid, dimBlock>>>(N,u_new_d,u_old_d,f_d,fbnorm_d);
        checkCudaErrors(cudaDeviceSynchronize());
        cudaMemcpy(fbnorm_h,fbnorm_d,sizeof(double),cudaMemcpyDeviceToHost);
        temp = u_old_d;
        u_old_d = u_new_d;
        u_new_d = temp;
        printf("norm is:%f",*fbnorm_h);
        k++;
    }



    /////////////////////  COPY DATA FROM DEVICE TO HOST /////////////////////
    transfer_3d(u_h, u_old_d, N+2, N+2, N+2, cudaMemcpyDeviceToHost);



    // dump  results if wanted 
    switch(output_type) {
	case 0:
	    // no output at all
	    break;
	case 3:
	    output_ext = ".bin";
	    sprintf(output_filename, "%s_%d%s", output_prefix, N+2, output_ext);
	    fprintf(stderr, "Write binary dump to %s: ", output_filename);
	    print_binary(output_filename, N+2, u_h);
	    break;
	case 4:
	    output_ext = ".vtk";
	    sprintf(output_filename, "%s_%d%s", output_prefix, N+2, output_ext);
	    fprintf(stderr, "Write VTK file to %s: ", output_filename);
	    print_vtk(output_filename, N+2, u_h);
	    break;
	default:
	    fprintf(stderr, "Non-supported output type!\n");
	    break;
    }


    ///////////////////////////     CLEAN UP    //////////////////////////////
    free(u_h);
    free(f_h);
    free_gpu(u_old_d);
    free_gpu(u_new_d);
    free_gpu(f_d);

    return(0);
}
