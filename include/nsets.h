/*
* cva_common.h
*/
 
#ifndef __NSETS_H__
#define __NSETS_H__


struct NSETS{
	double *nset[NUM_OF_THREADS];

	//derivative
	double *p_nset[NUM_OF_THREADS];
};

struct NSET{
	double *nset;
};


#endif
