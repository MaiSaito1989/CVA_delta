#include <mt19937.h>
#include<random_generator.h>

#include <cva_common.h>

double Uniform( void ){
	return genrand_real3() ;
}

void rand_normal_ndim(double *z, int n){
	int i;
	for	( i=0; i<n; i++) {
		z[i]=sqrt( -2.0*log(Uniform()) ) * sin( 2.0*PI*Uniform() );
	}
}


/*
 @param randoms Brownian paths.
 @param W (2*noc-1)*N*M brownian paths.
 @param noc the number of currencies.
 @param R (2*noc-1)X(2*nonc-1) correlation matrix.
 @param N the number of paths.
 @param M the number of partitions.
   */
void generate_brawnian_paths(double *W, int noc, double **R, int M, int N){
	int i;
	int a;
	int n;
	double *z;
	z = (double *)malloc(sizeof(double)*(2*noc-1));
	for(n=0; n<N*M; n++){
		rand_normal_ndim(z, 2*noc-1);
		for(i =0; i<2*noc-1; i++){
			W[i*(N*M)+n]=0.0;
			for(a=0; a<2*noc-1; a++){
				W[i*(N*M) + n]+=R[i][a]*z[a];
			}
		}
	}
	free(z);
}

