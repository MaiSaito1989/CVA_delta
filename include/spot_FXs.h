
/*
* spot_FXs.h
*/
 
#include <cva_common.h>

#ifndef __SPOT_FXS_H__
#define __SPOT_FXS_H__

struct SPOT_FXS{
	double *spot_fx[NUM_OF_CURRENCY-1][NUM_OF_THREADS];

	//derivative
	double *p_spot_fx[NUM_OF_CURRENCY-1][NUM_OF_THREADS];
};

struct SPOT_FX{
	double *spot_fx[NUM_OF_CURRENCY-1];
};

#endif

