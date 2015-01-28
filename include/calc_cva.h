
/*
* calc_cva.h
*/
#include <cva_common.h>
#include <short_rates.h>
#include <default_probability.h>
#include <discount_factors.h>
#include <spot_FXs.h>
#include <swaps.h>
#include <nsets.h>
#include <zero_coupon_bonds.h>
 
#ifndef __CALC_CVA_H__
#define __CALC_CVA_H__

__global__ void Calc_CVA(
		double *randoms,
		double *CVA,
		SHORT_RATES *rs,
		DISCOUNT_FACTORS *discount_factors,
		ZCBS *zcbs,
		SWAPS *swaps,
		SPOT_FXS *spot_fxs,
		NSETS *nsets,
		DEFAULT_PROBABILITY *default_probability,
		TIME_STAMPS *time_stamps,
		int N,
		int M,
		double *outputs
		);

#endif
