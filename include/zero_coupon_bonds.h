/*
* zero_coupon_bonds.h
*/
 
#include <cva_common.h>

#ifndef __ZERO_COUPON_BONDS_H__
#define __ZERO_COUPON_BONDS_H__

struct ZCBS{
	double *P[NUM_OF_CURRENCY][NUM_OF_THREADS];

	//derivative(length_sortedT*NUM_OF_MARKET_DATA)
	double *p_P[NUM_OF_CURRENCY];
};

struct ZCB{
	double *P[NUM_OF_CURRENCY];
};

#endif

