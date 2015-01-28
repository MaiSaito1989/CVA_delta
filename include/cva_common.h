/*
* cva_common.h
*/
 
#ifndef __CVA_COMMON_H__
#define __CVA_COMMON_H__

#define NUM_OF_THREADS (32)
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

struct SDE{
	int dim_y; //process Xの次元
	int dim_BM; //ブラウン運動の次元
	void *parameters;
	double *init_y;//初期値
	double T;
	int(**V)(const double y[], double dy[], void *parameters, double t);
	double(*payoff)(double y[], void *parameters);
};

//parameters of Hull white model
struct HW_PARAMS{
	double(*sigma)(const double t);
	double(*b)(const double t);
	double(*beta)(const double t);
	double t;
	double K;
	//other parameters...

} ;
/////////////////////////
struct BS_PARAMS{
	double r;
	double K;
	double sigma;
} ;

struct BSE_PARAMS{
	double(*r)(const double t,void *parameters);
	double K;
	double(*sigma)(const double t,void *parameters);
	double *rd;
	double *rf;
	int j;
} ;

#endif
