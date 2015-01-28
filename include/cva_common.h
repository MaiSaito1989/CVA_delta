/*
* cva_common.h
*/
 
#ifndef __CVA_COMMON_H__
#define __CVA_COMMON_H__

#define NUM_OF_THREADS (5*1024)
#define NUM_OF_MARKET_DATA 41 
#define PI 3.1415926535
#define NUM_OF_CURRENCY 2
#define NUM_OF_SWAPS 2
#define RECOVERY_RATE 0.4

typedef int (*VECTOR_FIELD1)(double y, double *dy, void *parameters, double t);

struct TIME_STAMPS{
	int *T0;
	int *Ti[NUM_OF_CURRENCY][NUM_OF_SWAPS];
	double *sortedT;
	int length_T0;
	int length_Ti[NUM_OF_CURRENCY][NUM_OF_SWAPS];
	int length_sortedT;
};

#endif
