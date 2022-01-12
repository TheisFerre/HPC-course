/* main.c - Poisson problem in 3D
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include "alloc3d.h"
#include "print.h"

#ifdef _JACOBI
#include "jacobi.h"
#endif

#ifdef _GAUSS_SEIDEL
#include "gauss_seidel.h"
#endif

#define N_DEFAULT 100

int
main(int argc, char *argv[]) {

    int 	N = N_DEFAULT;
    int 	iter_max = 1000;
    double	tolerance;
    double	start_T;
    int		output_type = 0;
    char	*output_prefix = "poisson_res";
    char        *output_ext    = "";
    char	output_filename[FILENAME_MAX];
    double 	***u = NULL;
    double  ***f = NULL;



    /* get the paramters from the command line */
    N         = atoi(argv[1]);	// grid size
    iter_max  = atoi(argv[2]);  // max. no. of iterations
    tolerance = atof(argv[3]);  // tolerance
    start_T   = atof(argv[4]);  // start T for all inner grid points
    if (argc == 6) {
	output_type = atoi(argv[5]);  // ouput type
    }

    // allocate memory
    if ( (u = d_malloc_3d(N+2, N+2, N+2)) == NULL ) {
        perror("array u: allocation failed");
        exit(-1);
    }
    if ( (f = d_malloc_3d(N+2, N+2, N+2)) == NULL ) {
        perror("array f: allocation failed");
        exit(-1);
    }


    /* Initialization of inner point in u*/
    for(int z=1;z<N+1;z++)
        for(int y=1;y<N+1;y++)
            for(int x=1;x<N+1;x++)
                U[z][y][x]=start_T

    /* Initialization of boundary points in u ~ wall(x,y)*/
    for(int y=0;y<N+2;y++)
        for(int x=0;x<N+2;x++)
            U[0][y][x]=20   
    for(int y=0;y<N+2;y++)
        for(int x=0;x<N+2;x++)
            U[N+1][y][x]=20  
    /* Wall (y,z) */
    for(int z=0;z<N+2;z++)
        for(int y=0;y<N+2;y++)
            U[z][y][0]=20  
    for(int z=0;z<N+2;z++)
        for(int y=0;y<N+2;y++)
            U[z][y][N+1]=20  
    /* Wall (x,z) */
    for(int z=0;z<N+2;z++)
        for(int x=0;x<N+2;x++)
            U[z][0][x]=0
    for(int z=0;z<N+2;z++)
        for(int x=0;x<N+2;x++)
            U[z][N+1][x]=20
    /* initialize f */
    double delta;
    delta = 2/(N + 1);
    for(int z=0;z<N+2;z++)
        for(int y=0;y<N+2;y++)
            for(int x=0;x<N+2;x++)
                if (-1 + delta * x <= -2/8 && -1 + delta * y <= -1/2 && -1 + delta * z >= -2/3 && -1 + delta * z <= 0 && ){
                    f[z][y][x] = 200
                }
                else{
                    f[z][y][x] = 0
                } 





    /*
     *
     * fill in your code here 
     *
     *
     */

    // dump  results if wanted 
    switch(output_type) {
	case 0:
	    // no output at all
	    break;
	case 3:
	    output_ext = ".bin";
	    sprintf(output_filename, "%s_%d%s", output_prefix, N, output_ext);
	    fprintf(stderr, "Write binary dump to %s: ", output_filename);
	    print_binary(output_filename, N, u);
	    break;
	case 4:
	    output_ext = ".vtk";
	    sprintf(output_filename, "%s_%d%s", output_prefix, N, output_ext);
	    fprintf(stderr, "Write VTK file to %s: ", output_filename);
	    print_vtk(output_filename, N, u);
	    break;
	default:
	    fprintf(stderr, "Non-supported output type!\n");
	    break;
    }

    // de-allocate memory
    free(u);

    return(0);
}
