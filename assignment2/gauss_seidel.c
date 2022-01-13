/* gauss_seidel.c - Poisson problem in 3d
 *
 */ 
#include <math.h>

void
gauss_seidel(int N, int iter_max, double tolerance, double ***u, double***f) {


    double fbnorm=1000000;
    double C= 1.0/6.0;
    double delta = 2.0/(N + 1.0);
    double delta2=delta*delta;
    int k;
    double u_old;
    while(fbnorm>tolerance && k<iter_max)
    {fbnorm=0;
        #pragma omp for ordered(2) private(j,k)
        for(int z=1;z<N+1;z++)
        {
            for(int y=1;y<N+1;y++)
            {
                #pragma omp ordered depend(sink: z-1,y)\
                                    depend(sink: z,y-1)
                for(int x=1;x<N+1;x++)
                {
                    u_old = u[z][y][x];
                    u[z][y][x]=C*(u[z-1][y][x] + u[z+1][y][x]+ \
                                  u[z][y-1][x] + u[z][y+1][x]+ \
                                  u[z][y][x-1] + u[z][y][x+1] + delta2*f[z][y][x]);
                    fbnorm+=(u[z][y][x]-u_old)*(u[z][y][x]-u_old);
                }
                #pragma omp ordered depend(source)
            }   
        }
    fbnorm=sqrt(fbnorm);
    k++;
    }
}
