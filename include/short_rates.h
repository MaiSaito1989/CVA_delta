/*
* short_rates.h
*/
 
#include <cva_common.h>

#ifndef __SHORT_RATES_H__
#define __SHORT_RATES_H__

typedef double (*FUNC_R1)(const double t);

struct SHORT_RATES{
	double *x[NUM_OF_CURRENCY][NUM_OF_THREADS];
	//VECTOR_FIELD vf_x[NUM_OF_CURRENCY][2];
	VECTOR_FIELD1 vf_x[NUM_OF_CURRENCY*2];
	//VECTOR_FIELD *vf_x;
	FUNC_R1 sigma[NUM_OF_CURRENCY];
	FUNC_R1 a[NUM_OF_CURRENCY];
};

struct SHORT_RATE{
	//CVAのdiscount factorは１ヶ月ごとに計算することにする
	double *x[NUM_OF_CURRENCY];
	VECTOR_FIELD1 vf_x[NUM_OF_CURRENCY][2];
	FUNC_R1 sigma[NUM_OF_CURRENCY];
	FUNC_R1 a[NUM_OF_CURRENCY];
};

__device__ int r_vf_x00(double y, double *dy, void *parameters, double t);
__device__ int r_vf_x01(double y, double *dy, void *parameters, double t);
__device__ int r_vf_x10(double y, double *dy, void *parameters, double t);
__device__ int r_vf_x11(double y, double *dy, void *parameters, double t);
__device__ double r_sigma0(const double t);
__device__ double r_sigma1(const double t);
__device__ double r_a0(const double t);
__device__ double r_a1(const double t);


#endif

