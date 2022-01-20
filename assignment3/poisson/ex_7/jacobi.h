/* jacobi.h - Poisson problem 
 *
 * $Id: jacobi.h,v 1.1 2006/09/28 10:12:58 bd Exp bd $
 */

#ifndef _JACOBI_H
#define _JACOBI_H

__global__ void jacobi_d0(int N, double ***u_new,double ***u_old,double ***u_other_dev, double***f);
__global__ void jacobi_d1(int N, double ***u_new,double ***u_old,double ***u_other_dev double***f);

#endif
