/* jacobi.h - Poisson problem 
 *
 * $Id: jacobi.h,v 1.1 2006/09/28 10:12:58 bd Exp bd $
 */

#ifndef _JACOBI_H
#define _JACOBI_H

__global__ void jacobi(int N, int iter_max, double tolerance, double ***u_new, double***f);

#endif
