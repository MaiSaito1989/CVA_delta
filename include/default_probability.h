
/*
* spwas.h
*/

#include <cva_common.h>
 
#ifndef __DEFAULT_PROBABILITY_H__
#define __DEFAULT_PROBABILITY_H__

typedef double (*FUNC_LAMBDA)(const double t);

struct DEFAULT_PROBABILITY{
	FUNC_LAMBDA lambda;
};


#endif
