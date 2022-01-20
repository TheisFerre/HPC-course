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
    double  ***f_h = NULL;

    /* get the paramters from the command line */
    N = atoi(argv[1]);	// grid size
    iter_max  = atoi(argv[2]);  // max. no. of iterations
    //tolerance = atof(argv[3]);  // tolerance
    start_T   = atof(argv[3]);  // start T for all inner grid points
    if (argc == 5) {
	output_type = atoi(argv[4]);  // ouput type
    }

    long nElms = (N+2) * (N+2) * (N+2);

    /////////////////////  ALLOCATE HOST MEMORY /////////////////////
    // allocate host memory
    if ( (u_h = d_malloc_3d(N+2, N+2, N+2)) == NULL ) {
        perror("array u: allocation failed");
        exit(-1);
    }
    if ( (f_h = d_malloc_3d(N+2, N+2, N+2)) == NULL ) {
        perror("array f: allocation failed");
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

    /////////////////////  ALLOCATE DEVICE MEMORY /////////////////////
    cudaSetDevice(0);
    cudaDeviceEnablePeerAccess(1, 0); /* Device 0 can access device 1*/
    double  ***u_old_d0 = NULL;
    double  ***u_new_d0 = NULL; 
    double  ***f_d0 = NULL;
    if ( (u_old_d0 = d_malloc_3d_gpu((N+2)/2, (N+2), (N+2))) == NULL ) {
        perror("array u_old_d0: allocation failed");
        exit(-1);
    }
    if ( (u_new_d0 = d_malloc_3d_gpu((N+2)/2, (N+2), (N+2))) == NULL ) {
        perror("array u_new_d0: allocation failed");
        exit(-1);
    }

    if ( (f_d0 = d_malloc_3d_gpu((N+2)/2, (N+2), (N+2))) == NULL ) {
        perror("array f_d0: allocation failed");
        exit(-1);
    }
    cudaSetDevice(1);
    cudaDeviceEnablePeerAccess(0, 0); /* Device 1 can access device 0*/
    double  ***u_old_d1 = NULL;
    double  ***u_new_d1 = NULL; 
    double  ***f_d1 = NULL;
    if ( (u_old_d1 = d_malloc_3d_gpu((N+2)/2, (N+2), (N+2))) == NULL ) {
        perror("array u_old_d1: allocation failed");
        exit(-1);
    }
    if ( (u_new_d1 = d_malloc_3d_gpu((N+2)/2, (N+2), (N+2))) == NULL ) {
        perror("array u_new_d1: allocation failed");
        exit(-1);
    }

    if ( (f_d1 = d_malloc_3d_gpu((N+2)/2, (N+2), (N+2))) == NULL ) {
        perror("array f_d1: allocation failed");
        exit(-1);
    }

    /////////////////////  COPY DATA FROM HOST TO DEVICE /////////////////////
    transfer_3d_from_1d(u_old_d0, u_h[0][0], (N+2)/2, N+2, N+2, cudaMemcpyHostToDevice);
    transfer_3d_from_1d(u_old_d1, u_h[0][0]+(nElms/2), (N+2)/2, N+2, N+2, cudaMemcpyHostToDevice);
    transfer_3d(u_new_d0,u_old_d0,(N+2)/2,N+2,N+2,cudaMemcpyDeviceToDevice);
    transfer_3d(u_new_d1,u_old_d1,(N+2)/2,N+2,N+2,cudaMemcpyDeviceToDevice);

    transfer_3d_from_1d(f_d0, f_h[0][0], (N+2)/2, N+2, N+2, cudaMemcpyHostToDevice);
    transfer_3d_from_1d(f_d1, f_h[0][0]+(nElms/2), (N+2)/2, N+2, N+2, cudaMemcpyHostToDevice);


    double*** temp_d0 = NULL;
    double*** temp_d1 = NULL;
    dim3 dimGrid(ceil(N/8.0),ceil(N/8.0),ceil(N/16.0));
    dim3 dimBlock(8,8,8);
    /////////////////////////////////  COMPUTE ///////////////////////////////
    for (int k=0; k<iter_max;k++){
        cudaSetDevice(0);
        jacobi_d0<<<dimGrid, dimBlock>>>(N,u_new_d0,u_old_d0,u_old_d1,f_d0);
        cudaSetDevice(1);
        jacobi_d1<<<dimGrid, dimBlock>>>(N,u_new_d1,u_old_d1,u_old_d0,f_d1);
        checkCudaErrors(cudaDeviceSynchronize());
        cudaSetDevice(0);
        checkCudaErrors(cudaDeviceSynchronize());
        temp_d0 = u_old_d0; /* Swap pointers for d0*/
        u_old_d0 = u_new_d0;
        u_new_d0 = temp_d0;

        temp_d1 = u_old_d1; /* Swap pointers for d1*/
        u_old_d1 = u_new_d1;
        u_new_d1 = temp_d1;
    }



    /////////////////////  COPY DATA FROM DEVICE TO HOST /////////////////////
    transfer_3d_to_1d(u_h[0][0], u_old_d0, (N+2)/2, N+2, N+2, cudaMemcpyDeviceToHost);
    transfer_3d_to_1d(u_h[0][0]+(nElms/2),u_old_d1, (N+2)/2, N+2, N+2, cudaMemcpyDeviceToHost);


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
    free_gpu(u_old_d0);
    free_gpu(u_new_d0);
    free_gpu(f_d0);

    return(0);
}
