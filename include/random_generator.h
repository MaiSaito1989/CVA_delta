
/*
* random_generator.h
*/

#ifndef __RANDOM_GENERATOR_H__
#define __RANDOM_GENERATOR_H__


double Uniform( void );

void rand_normal_ndim(double *z, int n);

void generate_brawnian_paths(double *W, int noc, double **R, int M, int N);

#endif
