
/*
* spwas.h
*/

#include <cva_common.h>
 
#ifndef __SWAPS_H__
#define __SWAPS_H__

#define NUM_OF_SWAPS 2

struct SWAPS{
	double *swap[NUM_OF_CURRENCY][NUM_OF_SWAPS][NUM_OF_THREADS];
	//Nominal
	double N[NUM_OF_CURRENCY][NUM_OF_SWAPS];
	//fixed rate
	double K[NUM_OF_CURRENCY][NUM_OF_SWAPS];

	//double pointer (length_sortedT*NUM_OF_MARKET_DATA)
	double *p_swap[NUM_OF_CURRENCY][NUM_OF_SWAPS][NUM_OF_THREADS];
};

struct SWAP{
	double *swap[NUM_OF_CURRENCY][NUM_OF_SWAPS];
	//Nominal
	double N[NUM_OF_CURRENCY][NUM_OF_SWAPS];
	//fixed rate
	double K[NUM_OF_CURRENCY][NUM_OF_SWAPS];
};

#endif
