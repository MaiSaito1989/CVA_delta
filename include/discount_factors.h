
/*
* discount_factors.h
*/
 
#include <cva_common.h>

#ifndef __DISCOUNT_FACTORS_H__
#define __DISCOUNT_FACTORS_H__

struct DISCOUNT_FACTORS{
	//CVAのdiscount factorは１ヶ月ごとに計算することにする
	double *D[NUM_OF_CURRENCY][NUM_OF_THREADS];
	
	//derivative
	double *p_D[NUM_OF_CURRENCY][NUM_OF_THREADS];
};

struct DISCOUNT_FACTOR{
	//CVAのdiscount factorは１ヶ月ごとに計算することにする
	double *D[NUM_OF_CURRENCY];
};

#endif

