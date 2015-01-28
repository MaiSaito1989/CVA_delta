#include <stdio.h>	
#include <stdlib.h>
#include <math.h>
#include <random_generator.h>
#include <helper_cuda.h>
#include <mt19937.h>

#include <cva_common.h>
#include <calc_cva.h>
#include <short_rates.h>
#include <default_probability.h>
#include <discount_factors.h>
#include <zero_coupon_bonds.h>
#include <swaps.h>
#include <spot_FXs.h>
#include <nsets.h>
#include <utilities.h>

#include <calc_cva_delta.h>

__device__ int r_vf_x00(double y, double *dy, void *parameters, double t) {
	dy[0] = -r_a0(t)*y ; 
	return 0;
}

__device__ int r_vf_x01(double y, double *dy, void *parameters, double t) {
	dy[0] = r_sigma0(t); 
	return 0;
}

__device__ int r_vf_x10(double y, double *dy, void *parameters, double t) {
	dy[0] = -r_a1(t)*y;
	return 0;
}

__device__ int r_vf_x11(double y, double *dy, void *parameters, double t) {	
	dy[0] = r_sigma1(t);
	return 0;
}

__device__ double r_sigma0(const double t) {
    return 0.1;  
}

__device__ double r_sigma1(const double t) {
	return 0.1;
}
__device__ double r_a0(const double t) {
	return 2.0;
}

__device__ double r_a1(const double t) {
	return 2.0;
}

//default intensity lambda
__device__ double dp_lambda(const double t) {
	return 0.00001;
}

__device__ VECTOR_FIELD1 d_r_vf_x00 = r_vf_x00;
__device__ VECTOR_FIELD1 d_r_vf_x01 = r_vf_x01;
__device__ VECTOR_FIELD1 d_r_vf_x10 = r_vf_x10;
__device__ VECTOR_FIELD1 d_r_vf_x11 = r_vf_x11;
__device__ FUNC_R1 d_r_sigma0 = r_sigma0;
__device__ FUNC_R1 d_r_sigma1 = r_sigma1;
__device__ FUNC_R1 d_r_a0 = r_a0;
__device__ FUNC_R1 d_r_a1 = r_a1;
__device__ FUNC_LAMBDA d_dp_lambda = dp_lambda;

__global__ void Calc_CVA_Delta(
		double *randoms,
		double *CVA_Delta,
		SHORT_RATES *r,
		DISCOUNT_FACTORS *discount_factor,
		ZCBS *zcb,
		SWAPS *swap,
		SPOT_FXS *spot_fx,
		NSETS *nset,
		DEFAULT_PROBABILITY *default_probability,
		TIME_STAMPS *time_stamps,
		int N,
		int M,
		int g_c,
		double *outputs
) 
{

	int i,j,k,l,m,n,mm;
	double sum;
	int threadId = threadIdx.x + threadIdx.y*blockDim.x + blockDim.x*blockDim.y*blockIdx.x;
	
	//domestic curency
	double *Wd = randoms + threadId *time_stamps->length_sortedT;
	//foreign currency
	double *Wf[NUM_OF_CURRENCY-1];
	for (k=0; k<NUM_OF_CURRENCY-1; k++) {
		Wf[k] = randoms + threadId*time_stamps->length_sortedT + N*time_stamps->length_sortedT*(k+1);
	}
	//FX domestic/foreign
	double *Ws[NUM_OF_CURRENCY-1];
	for (k=0; k<NUM_OF_CURRENCY-1; k++) {
		Ws[k] = randoms + threadId*time_stamps->length_sortedT + N*time_stamps->length_sortedT*(k+NUM_OF_CURRENCY);
	}
	double *W[2*NUM_OF_CURRENCY-1];
	W[0] = Wd;
	for (k=0; k<NUM_OF_CURRENCY-1; k++) {
		W[k+1] = Wf[k];
	}
	for (k=0; k<NUM_OF_CURRENCY-1; k++) {
		W[k+NUM_OF_CURRENCY] = Ws[k];
	}

	//Variables for calculation
	double t;
	double h;
	double T;
	double temp1;
	double temp2;
	double temp3;
	int int_temp;
	double shuffle_temp;
	double v1;
	double v2;
	double v3;
	double ts,te;

	/*
	calc process x
	*/
	for (k=0; k<NUM_OF_CURRENCY; k++) {
		//old 初期値、　newにどんどん足される
		temp1 = r->x[k][threadId][0];
		t=0.0;
		//n番目分割点
		for (n=0; n<time_stamps->length_sortedT-1; n++) {
			//step size
			h=time_stamps->sortedT[n+1]-time_stamps->sortedT[n];
			//1こう目
			temp2 = temp1;

			//３こう目
			r->vf_x[k*NUM_OF_CURRENCY + 1](temp1, &temp3, NULL, t);
			temp2 += temp3*sqrt(h) *W[k][n];
			//2こう目
			r->vf_x[k*NUM_OF_CURRENCY + 0](temp1, &temp3, NULL, t);
			temp2 += temp3*h;

			t+=h;

			//値の入れ替え
			shuffle_temp = temp2;
			temp2 = temp1;
			temp1 = shuffle_temp;
			r->x[k][threadId][n+1] = temp1;
		}
	}

	/*
	calc discount factor
	*/
	for (k=0; k<NUM_OF_CURRENCY; k++) {
		temp1=0.0;
		//Expected loss を計算する日
		for (m=0; m<time_stamps->length_sortedT-1; m++) {
			t=0.0;
			T=time_stamps->sortedT[m+1];

			//integrate the path of x
			//台形近似
			temp1 += (r->x[k][threadId][m]+ r->x[k][threadId][m+1])*(time_stamps->sortedT[m+1]-time_stamps->sortedT[m])/2.0;
			//exp(-integrate(x))
			temp2=exp(-temp1);
			
			//calculate exp(-integrate(phi))
			ts=t;
			te=T;
			v1 = r->sigma[k](T)*r->sigma[k](T)/(2.0*r->a[k](t)*r->a[k](t)*r->a[k](t))*
			(exp(-2.0*r->a[k](t)*T)*(exp(r->a[k](t)*te)-exp(r->a[k](t)*ts))*(exp(r->a[k](t)*te)+exp(r->a[k](t)*ts)-4.0*exp(r->a[k](t)*T)) + 2.0*r->a[k](t)*(te- ts));	

			ts=t;
			te=t;
			v2 = r->sigma[k](t)*r->sigma[k](t)/(2.0*r->a[k](t)*r->a[k](t)*r->a[k](t))*
			(exp(-2.0*r->a[k](t)*T)*(exp(r->a[k](t)*te)-exp(r->a[k](t)*ts))*(exp(r->a[k](t)*te)+exp(r->a[k](t)*ts)-4.0*exp(r->a[k](t)*T)) + 2.0*r->a[k](t)*(te- ts));	
		
			temp3=(zcb->P[k][threadId][m+1]/zcb->P[k][threadId][0])*exp(-(v1-v2)/2.0);
			
			discount_factor->D[k][threadId][m+1] = temp2*temp3;
			outputs[threadId*time_stamps->length_sortedT + k*time_stamps->length_sortedT*N + m] = zcb->P[k][threadId][m+1];
		}
	}

	/*
	calc zero coupon bond
	*/
	for (k=0; k<NUM_OF_CURRENCY; k++) {
		for (j=0; j<NUM_OF_SWAPS; j++) {
			//Ti
			for (i=0; i<time_stamps->length_Ti[k][j]; i++) {
				//T0
				for (l=0; l<time_stamps->length_T0; l++) {
					//temp1=A(t,T)
					t=time_stamps->sortedT[time_stamps->T0[l]];
					T=time_stamps->sortedT[time_stamps->Ti[k][j][i]];

					ts=0.0;
					te=T;
					v1 = r->sigma[k](t)*r->sigma[k](t)/(2.0*r->a[k](t)*r->a[k](t)*r->a[k](t))*
					(exp(-2.0*r->a[k](t)*T)*(exp(r->a[k](t)*te)-exp(r->a[k](t)*ts))*(exp(r->a[k](t)*te)+exp(r->a[k](t)*ts)-4.0*exp(r->a[k](t)*T)) + 2.0*r->a[k](t)*(te- ts));	
					
					ts=0.0;
					te=t;
					v2 = r->sigma[k](t)*r->sigma[k](t)/(2.0*r->a[k](t)*r->a[k](t)*r->a[k](t))*
					(exp(-2.0*r->a[k](t)*T)*(exp(r->a[k](t)*te)-exp(r->a[k](t)*ts))*(exp(r->a[k](t)*te)+exp(r->a[k](t)*ts)-4.0*exp(r->a[k](t)*T)) + 2.0*r->a[k](t)*(te- ts));	
				
					ts=t;
					te=T;
					v3 =r->sigma[k](t)*r->sigma[k](t)/(2.0*r->a[k](t)*r->a[k](t)*r->a[k](t))*
					(exp(-2.0*r->a[k](t)*T)*(exp(r->a[k](t)*te)-exp(r->a[k](t)*ts))*(exp(r->a[k](t)*te)+exp(r->a[k](t)*ts)-4.0*exp(r->a[k](t)*T)) + 2.0*r->a[k](t)*(te- ts));	
					temp1 = log(zcb->P[k][threadId][time_stamps->Ti[k][j][i]]/zcb->P[k][threadId][time_stamps->T0[l]])+(v3-v1+v2)/2.0;

					//temp2=B(t,T)
					temp2 = (1.0- exp(-r->a[k](t)*(T-t)))/r->a[k](t);

					//P
					zcb->P[k][threadId][time_stamps->T0[l]*time_stamps->length_sortedT+time_stamps->Ti[k][j][i]]=exp(temp1-temp2*r->x[k][threadId][time_stamps->T0[l]]);
				}
			}
		}
	}


	/*
	calc swaps
	*/
	for (k=0; k<NUM_OF_CURRENCY; k++) {
		for (j=0; j<NUM_OF_SWAPS; j++) {
			//T0
			for (l=0; l<time_stamps->length_T0; l++) {
				swap->swap[k][j][threadId][time_stamps->T0[l]]=0.0;
				//Ti
				for (i=0; i<time_stamps->length_Ti[k][j]-1; i++) {
					//temp1=FxC(T0,Ti,Ti+1). Fixed Coupon
					temp1= swap->K[k][j]*zcb->P[k][threadId][time_stamps->T0[l]*time_stamps->length_sortedT + time_stamps->Ti[k][j][i+1]]*(time_stamps->sortedT[time_stamps->Ti[k][j][i+1]] - time_stamps->sortedT[time_stamps->Ti[k][j][i]]);

					//temp2=FlC(T0,Ti,Ti+1). Floating Coupon
					temp2= 
					(zcb->P[k][threadId][time_stamps->T0[l]*time_stamps->length_sortedT + time_stamps->Ti[k][j][i]]
					/zcb->P[k][threadId][time_stamps->T0[l]*time_stamps->length_sortedT + time_stamps->Ti[k][j][i+1]] - 1.0);
					
					//temp3=Swap(T0,...). Swap
					swap->swap[k][j][threadId][time_stamps->T0[l]]+=temp1-temp2;
				}
				swap->swap[k][j][threadId][time_stamps->T0[l]]*=swap->N[k][j];
			}
		}
	}


	/*
	calc FX
	*/
	for (k=0; k<NUM_OF_CURRENCY-1; k++) {
		//T0
		for (l=0; l<time_stamps->length_T0; l++) {
			//temp3=exp(Ito(sigma))
			temp3=0.0;
			t=0.0;
			//Euler Maruyama
			for (n=0; n<time_stamps->T0[l]-1; n++) {
				h = time_stamps->sortedT[n+1]-time_stamps->sortedT[n];
				temp3+=r->sigma[k+1](t)*sqrt(h)*Ws[k][n];
				t+=h;
			}
			spot_fx->spot_fx[k][threadId][time_stamps->T0[l]]=
			spot_fx->spot_fx[k][threadId][0]*
			discount_factor->D[k+1][threadId][time_stamps->T0[l]]/
			discount_factor->D[0][threadId][time_stamps->T0[l]]*
			exp(temp3);
		}
	}

	/*
	calc NSet
	*/
	for (l=0; l<time_stamps->length_T0; l++) {
		nset->nset[threadId][time_stamps->T0[l]]=0.0;
		//domestic
		for (j=0; j<NUM_OF_SWAPS; j++) {
			nset->nset[threadId][time_stamps->T0[l]] += swap->swap[0][j][threadId][time_stamps->T0[l]];
		}
		//foreign
		for (k=1; k<NUM_OF_CURRENCY; k++) {
			for (j=0; j<NUM_OF_SWAPS; j++) {
				nset->nset[threadId][time_stamps->T0[l]] += swap->swap[k][j][threadId][time_stamps->T0[l]]*spot_fx->spot_fx[k-1][threadId][time_stamps->T0[l]];
			}
		}
	}


	/*
	differentiates swaps
	*/
	//first term
	//differentiate FxC by P(dFxC/dP)
	for (j=0; j<NUM_OF_SWAPS; j++) {
		for (mm=0; mm<time_stamps->length_T0; mm++) {
			for(m=0; m<NUM_OF_MARKET_DATA; m++){
				swap->p_swap[g_c][j][threadId][time_stamps->T0[mm]*NUM_OF_MARKET_DATA + m] =  0.0;
				for (l=0; l<time_stamps->length_Ti[g_c][j]-1; l++) {
					swap->p_swap[g_c][j][threadId][time_stamps->T0[mm]*(NUM_OF_MARKET_DATA) + m] += 
					(time_stamps->sortedT[time_stamps->Ti[g_c][j][l+1]] - time_stamps->sortedT[time_stamps->Ti[g_c][j][l]])*
					(-zcb->P[g_c][threadId][time_stamps->T0[mm]*time_stamps->length_sortedT + time_stamps->Ti[g_c][j][l]]/
					zcb->P[g_c][threadId][0 + time_stamps->T0[mm]]*
					zcb->p_P[g_c][time_stamps->T0[mm]*NUM_OF_MARKET_DATA + m]
					+
					zcb->P[g_c][threadId][time_stamps->T0[mm]*time_stamps->length_sortedT + time_stamps->Ti[g_c][j][l]]/
					zcb->P[g_c][threadId][0 + time_stamps->Ti[g_c][j][l]]*
					zcb->p_P[g_c][time_stamps->Ti[g_c][j][l]*NUM_OF_MARKET_DATA + m]);
				}
				swap->p_swap[g_c][j][threadId][time_stamps->T0[mm]*(NUM_OF_MARKET_DATA) + m] *=  swap->N[g_c][j]*swap->K[g_c][j];
			}
		}
	}
	
	
	//second term
	//differentiate FwdR by P
	for (j=0; j<NUM_OF_SWAPS; j++) {
		//expected loss
		for (mm=0; mm<time_stamps->length_T0; mm++) {
			for(m=0; m<NUM_OF_MARKET_DATA; m++){
				sum = 0.0;
				for (l=0; l<time_stamps->length_Ti[g_c][j]-1; l++) {
					sum +=
					1.0/
					zcb->P[g_c][threadId][time_stamps->T0[mm]*time_stamps->length_sortedT + time_stamps->Ti[g_c][j][l+1]]*
					(
						-zcb->P[g_c][threadId][time_stamps->T0[mm]*time_stamps->length_sortedT + time_stamps->Ti[g_c][j][0]]/
						zcb->P[g_c][threadId][0 + time_stamps->T0[mm]]*
						zcb->p_P[g_c][time_stamps->T0[mm]*NUM_OF_MARKET_DATA + m]
						+
						zcb->P[g_c][threadId][time_stamps->T0[mm]*time_stamps->length_sortedT + time_stamps->Ti[g_c][j][l]]/
						zcb->P[g_c][threadId][0 + time_stamps->Ti[g_c][j][l]]*
						zcb->p_P[g_c][time_stamps->Ti[g_c][j][l]*NUM_OF_MARKET_DATA + m]
					)
					+
					zcb->P[g_c][threadId][time_stamps->T0[mm]*time_stamps->length_sortedT + time_stamps->Ti[g_c][j][0]]/
					(
						zcb->P[g_c][threadId][0 + time_stamps->T0[mm]]*
						zcb->P[g_c][threadId][0 + time_stamps->T0[mm]]
					)
					*
					(
						-zcb->P[g_c][threadId][time_stamps->T0[mm]*time_stamps->length_sortedT + time_stamps->Ti[g_c][j][l+1]]/
						zcb->P[g_c][threadId][0 + time_stamps->T0[mm]]*
						zcb->p_P[g_c][time_stamps->T0[mm]*NUM_OF_MARKET_DATA + m]
						+
						zcb->P[g_c][threadId][time_stamps->T0[mm] + time_stamps->Ti[g_c][j][l+1]]/
						zcb->P[g_c][threadId][0 + time_stamps->Ti[g_c][j][l]]*
						zcb->p_P[g_c][time_stamps->Ti[g_c][j][l+1]*NUM_OF_MARKET_DATA + m]
					);
				}
				sum +=
				-1.0/
				(
					zcb->P[g_c][threadId][time_stamps->T0[mm]*time_stamps->length_sortedT + time_stamps->Ti[g_c][j][0]]*
					zcb->P[g_c][threadId][time_stamps->T0[mm]*time_stamps->length_sortedT + time_stamps->Ti[g_c][j][0]]
				)
				*
				(
					-zcb->P[g_c][threadId][time_stamps->T0[mm]*time_stamps->length_sortedT + time_stamps->Ti[g_c][j][0]]/
					zcb->P[g_c][threadId][0 + time_stamps->T0[mm]]*
					zcb->p_P[g_c][time_stamps->T0[mm]*NUM_OF_MARKET_DATA + m]
					+
					zcb->P[g_c][threadId][time_stamps->T0[mm]*time_stamps->length_sortedT + time_stamps->Ti[g_c][j][0]]/
					zcb->P[g_c][threadId][0 + time_stamps->Ti[g_c][j][0]]*
					zcb->p_P[g_c][time_stamps->Ti[g_c][j][0]*NUM_OF_MARKET_DATA + m]
				);
				sum *= swap->N[g_c][j];

				swap->p_swap[g_c][j][threadId][time_stamps->T0[mm]*NUM_OF_MARKET_DATA + m] += sum;
			}
		}
	}
	/*
	differentiate Fxs
	*/
	if(g_c == 0){
		for (mm=0; mm<time_stamps->length_T0; mm++) {
			for(m=0; m<NUM_OF_MARKET_DATA; m++){
				spot_fx->p_spot_fx[0][threadId][time_stamps->T0[mm]*NUM_OF_MARKET_DATA + m] = 
					-spot_fx->spot_fx[0][threadId][time_stamps->T0[mm]]/
					zcb->P[0][threadId][time_stamps->T0[mm]]*
					zcb->p_P[0][time_stamps->T0[mm]*NUM_OF_MARKET_DATA + m];
			}
		}
	}else{
		for (mm=0; mm<time_stamps->length_T0; mm++) {
			for(m=0; m<NUM_OF_MARKET_DATA; m++){
				spot_fx->p_spot_fx[g_c][threadId][time_stamps->T0[mm]*NUM_OF_MARKET_DATA + m] = 
					spot_fx->spot_fx[g_c][threadId][time_stamps->T0[mm]]/
					zcb->P[g_c][threadId][time_stamps->T0[mm]]*
					zcb->p_P[g_c][time_stamps->T0[mm]*NUM_OF_MARKET_DATA + m];
			}
		}
	}
	/*
	differentiate discount factor
	*/
	if(g_c == 0){
		for (mm=0; mm<time_stamps->length_T0; mm++) {
			for(m=0; m<NUM_OF_MARKET_DATA; m++){
				discount_factor->p_D[0][threadId][time_stamps->T0[mm]*NUM_OF_MARKET_DATA + m] =
				discount_factor->D[0][threadId][time_stamps->T0[mm]]/
				zcb->P[0][threadId][time_stamps->T0[mm]]*
				zcb->p_P[0][time_stamps->T0[mm]*NUM_OF_MARKET_DATA + m];
			}
		}
	}


	/*
	differentiate Net Value
	*/
	//domestic
	if(g_c == 0){
		for (mm=0; mm<time_stamps->length_T0; mm++) {
			for(m=0; m<NUM_OF_MARKET_DATA; m++){
				nset->p_nset[threadId][time_stamps->T0[mm]*NUM_OF_MARKET_DATA + m] = 0.0;

				//differentiate domestic swaps
				for (j=0; j<NUM_OF_SWAPS; j++) {
					nset->p_nset[threadId][time_stamps->T0[mm]*NUM_OF_MARKET_DATA + m] +=
						swap->p_swap[0][j][threadId][time_stamps->T0[mm]*NUM_OF_MARKET_DATA + m];
				}

				//differentiate foreighn swaps
				for (k=1; k<NUM_OF_CURRENCY; k++) {
					for (j=0; j<NUM_OF_SWAPS; j++) {
						nset->p_nset[threadId][time_stamps->T0[mm]*NUM_OF_MARKET_DATA + m] +=
							swap->swap[k][j][threadId][time_stamps->T0[mm]]*
							spot_fx->p_spot_fx[k-1][threadId][time_stamps->T0[mm]*NUM_OF_MARKET_DATA + m];
					}
				}
			}
		}
	//foreign
	}else{
		for (mm=0; mm<time_stamps->length_T0; mm++) {
			for(m=0; m<NUM_OF_MARKET_DATA; m++){
				nset->p_nset[threadId][time_stamps->T0[mm] + m] = 0.0;

				//differentiate swaps in g_c
				for (j=0; j<NUM_OF_SWAPS; j++) {
					nset->p_nset[threadId][time_stamps->T0[mm] + m] +=
						swap->p_swap[g_c][j][threadId][time_stamps->T0[mm]*NUM_OF_MARKET_DATA + m]*
						spot_fx->spot_fx[g_c-1][threadId][time_stamps->T0[mm]]
						+
						swap->swap[g_c][j][threadId][time_stamps->T0[mm]]*
						spot_fx->p_spot_fx[g_c-1][threadId][time_stamps->T0[mm]*NUM_OF_MARKET_DATA + m];
				}
			}
		}
	}

						
	/*
	Calc CVA_greeksG
	*/
	//integrate(-lambda))
	temp1=0.0;
	temp2=0.0;
	//SP(Ti-1):Ti-1までに死んでいる確率
	temp3=0.0;
	int_temp=0;
	if(g_c == 0) {
		for (mm=0; mm<time_stamps->length_T0; mm++) {
			for (m=0; m<NUM_OF_MARKET_DATA; m++) {
				CVA_Delta[threadId*NUM_OF_MARKET_DATA + m] = 0.0;

				//PS
				//integrate(-lambda))
				for (n=int_temp; n<time_stamps->T0[mm]; n++) {
					//台形近似
					temp1 += (default_probability->lambda(time_stamps->sortedT[n+1])+default_probability->lambda(time_stamps->sortedT[n]))*(time_stamps->sortedT[n+1]-time_stamps->sortedT[n])/2.0;
				}
				int_temp=time_stamps->T0[mm];
				temp3=temp2;
				//SP(Ti)
				temp2=1.0-exp(-temp1);
				
				//expected loss*p_D
				CVA_Delta[threadId*NUM_OF_MARKET_DATA + m] += 
					discount_factor->p_D[0][threadId][time_stamps->T0[mm]*NUM_OF_MARKET_DATA + m]*
					nset->nset[threadId][time_stamps->T0[mm]]*(temp2-temp3);
				
				//(derivative of expected loss)*D
				if (nset->nset[threadId][time_stamps->T0[mm]]>0.0) {
					CVA_Delta[threadId*NUM_OF_MARKET_DATA + m] += 
						discount_factor->D[0][threadId][time_stamps->T0[mm]]*
						nset->p_nset[threadId][time_stamps->T0[mm]*NUM_OF_MARKET_DATA + m]*(temp2-temp3);
				}
				CVA_Delta[threadId*NUM_OF_MARKET_DATA + m] *= (1.0 - RECOVERY_RATE);
			}
		}
	}
}

int main(void) {
	int i,j,k,l,n,m;
	
	// Copy up each piece separately, including new “name” pointer value
	unsigned long init[4]={0x123, 0x234, 0x345, 0x456}, length=4;
	init_by_array(init, length);


#define TL0  24
#define TL21 20

	//delta 
	int Delta_Currency = 0;

	//input data(sample)
	double T0[TL0];
	for (i=0; i<TL0; i++) {
		T0[i] = (i+1)/12.0;
	}
	double T10[6];
	for (i=0; i<6; i++) {
		T10[i] = (i+1)/2.0;
	}
	double T11[6];
	for (i=0; i<6; i++) {
		T11[i] = (i+1);
	}
	double T20[8];
	for (i=0; i<8; i++) {
		T20[i] = (i+1);
	}
	double T21[TL21];
	for (i=0; i<TL21; i++) {
		T21[i] = (i+1)/2.0;
	}
	double ***Ti;
	Ti = (double ***)malloc(sizeof(double**)*NUM_OF_CURRENCY);
	Ti[0] = (double **)malloc(sizeof(double*)*NUM_OF_SWAPS);
	Ti[1] = (double **)malloc(sizeof(double*)*NUM_OF_SWAPS);
	Ti[0][0] = T10;
	Ti[0][1] = T11;
	Ti[1][0] = T20;
	Ti[1][1] = T21;

	/*****
	Time Stamps
	*****/
	//malloc in host
	TIME_STAMPS *time_stamps;
	time_stamps = (TIME_STAMPS *)malloc(sizeof(TIME_STAMPS));
	time_stamps->T0 = (int *)malloc(sizeof(int)*TL0);
	time_stamps->Ti[0][0] = (int *)malloc(sizeof(int)*6);
	time_stamps->Ti[0][1] = (int *)malloc(sizeof(int)*6);
	time_stamps->Ti[1][0] = (int *)malloc(sizeof(int)*8);
	time_stamps->Ti[1][1] = (int *)malloc(sizeof(int)*TL21);
	time_stamps->length_T0=TL0;
	time_stamps->length_Ti[0][0] = 6;
	time_stamps->length_Ti[0][1] = 6;
	time_stamps->length_Ti[1][0] = 8;
	time_stamps->length_Ti[1][1] = TL21;

	time_stamps_sort(time_stamps, T0, Ti, NUM_OF_CURRENCY, NUM_OF_SWAPS);

	printf("T0: ");
	for (l=0; l<time_stamps->length_T0; l++) {
		printf("%d ", time_stamps->T0[l]);
	}
	printf("\n");
	for (k=0; k<NUM_OF_CURRENCY; k++) {
		for (j=0; j<NUM_OF_SWAPS; j++) {
			printf("T%d%d: ", k, j);
			for (l=0; l<time_stamps->length_Ti[k][j]; l++) {
				printf("%d ", time_stamps->Ti[k][j][l]);
			}
			printf("\n");
		}
	}
	printf("\n");
	printf("sortedT[%d]:\n", time_stamps->length_sortedT);
	for (l=0; l<time_stamps->length_sortedT; l++) {
		printf("%d %lf\n", l, time_stamps->sortedT[l]);
	}
	printf("\n");

	//device malloc for time_stamps
	TIME_STAMPS *d_time_stamps;
	int *d_T0;
	int *d_Ti[NUM_OF_CURRENCY][NUM_OF_SWAPS];
	double *d_sortedT;
	checkCudaErrors(cudaMalloc((void**)&d_time_stamps, sizeof(TIME_STAMPS)));
	checkCudaErrors(cudaMalloc((void**)&d_T0, sizeof(int)*time_stamps->length_T0));
	for (k=0; k<NUM_OF_CURRENCY; k++) {
		for (j=0; j<NUM_OF_SWAPS; j++) {
			checkCudaErrors(cudaMalloc((void**)&d_Ti[k][j], sizeof(int)*time_stamps->length_Ti[k][j]));
		}
	}
	checkCudaErrors(cudaMalloc((void**)&d_sortedT, sizeof(double)*time_stamps->length_sortedT));


	//memory copy to device for time stamps
	checkCudaErrors(cudaMemcpy(d_T0, time_stamps->T0, sizeof(int)*time_stamps->length_T0, cudaMemcpyHostToDevice));
	for (k=0; k<NUM_OF_CURRENCY; k++) {
		for (j=0; j<NUM_OF_SWAPS; j++) {
			checkCudaErrors(cudaMemcpy(d_Ti[k][j], time_stamps->Ti[k][j], sizeof(int)*time_stamps->length_Ti[k][j], cudaMemcpyHostToDevice));
		}
	}
	checkCudaErrors(cudaMemcpy(d_sortedT, time_stamps->sortedT, sizeof(double)*time_stamps->length_sortedT, cudaMemcpyHostToDevice));
		
	
	//temporally swap Adresses for copying struct
	int *p_int_temp;
	double *p_double_temp;
	p_int_temp = time_stamps->T0;
	time_stamps->T0 = d_T0;
	d_T0 = p_int_temp;
	for (k=0; k<NUM_OF_CURRENCY; k++) {
		for (j=0; j<NUM_OF_SWAPS; j++) {
			p_int_temp = time_stamps->Ti[k][j];
			time_stamps->Ti[k][j] = d_Ti[k][j];
			d_Ti[k][j] = p_int_temp;
		}
	}
	p_double_temp = time_stamps->sortedT;
	time_stamps->sortedT = d_sortedT;
	d_sortedT = p_double_temp;

	//copy struct
	checkCudaErrors(cudaMemcpy(d_time_stamps, time_stamps, sizeof(TIME_STAMPS), cudaMemcpyHostToDevice));

	//reswap
	p_int_temp = time_stamps->T0;
	time_stamps->T0 = d_T0;
	d_T0 = p_int_temp;
	for (k=0; k<NUM_OF_CURRENCY; k++) {
		for (j=0; j<NUM_OF_SWAPS; j++) {
			p_int_temp = time_stamps->Ti[k][j];
			time_stamps->Ti[k][j] = d_Ti[k][j];
			d_Ti[k][j] = p_int_temp;
		}
	}
	p_double_temp = time_stamps->sortedT;
	time_stamps->sortedT = d_sortedT;
	d_sortedT = p_double_temp;


	/*****
	Short Rates
	*****/
	//malloc in host
	SHORT_RATES *rs;
	rs = (SHORT_RATES *)malloc(sizeof(SHORT_RATES));
	for (k=0; k<NUM_OF_CURRENCY; k++) {
		for (n=0; n<NUM_OF_THREADS; n++) {
			rs->x[k][n] = (double *)malloc(sizeof(double)*time_stamps->length_sortedT);			
		}
	}

	//device malloc for short rates
	SHORT_RATES *d_rs;
	double *d_x[NUM_OF_CURRENCY][NUM_OF_THREADS];
	checkCudaErrors(cudaMalloc((void**)&d_rs, sizeof(SHORT_RATES)));
	for (k=0; k<NUM_OF_CURRENCY; k++) {
		for (n=0; n<NUM_OF_THREADS; n++) {
			checkCudaErrors(cudaMalloc((void**)&d_x[k][n], sizeof(double)*time_stamps->length_sortedT));
		}
	}

	//set initial data
	checkCudaErrors(cudaMemcpyFromSymbol(&(rs->vf_x[0]), d_r_vf_x00, sizeof(VECTOR_FIELD1)));
	checkCudaErrors(cudaMemcpyFromSymbol(&(rs->vf_x[1]), d_r_vf_x01, sizeof(VECTOR_FIELD1)));
	checkCudaErrors(cudaMemcpyFromSymbol(&(rs->vf_x[2]), d_r_vf_x10, sizeof(VECTOR_FIELD1)));
	checkCudaErrors(cudaMemcpyFromSymbol(&(rs->vf_x[3]), d_r_vf_x11, sizeof(VECTOR_FIELD1)));
	checkCudaErrors(cudaMemcpyFromSymbol(&(rs->sigma[0]), d_r_sigma0, sizeof(FUNC_R1)));
	checkCudaErrors(cudaMemcpyFromSymbol(&(rs->sigma[1]), d_r_sigma1, sizeof(FUNC_R1)));
	checkCudaErrors(cudaMemcpyFromSymbol(&(rs->a[0]), d_r_a0, sizeof(FUNC_R1)));
	checkCudaErrors(cudaMemcpyFromSymbol(&(rs->a[1]), d_r_a1, sizeof(FUNC_R1)));
	for (k=0; k<NUM_OF_CURRENCY; k++) {
		for (n=0; n<NUM_OF_THREADS; n++) {
			for (j=0; j<time_stamps->length_sortedT; j++) {
				rs->x[k][n][j] = 0.0;
			}
		}
	}

	//temporally swap Adresses for copying struct
	for (k=0; k<NUM_OF_CURRENCY; k++) {
		for (n=0; n<NUM_OF_THREADS; n++) {
			p_double_temp = rs->x[k][n];
			rs->x[k][n]= d_x[k][n];
			d_x[k][n]= p_double_temp;
		}
	}

	//copy struct from host to device
	checkCudaErrors(cudaMemcpy(d_rs, rs, sizeof(SHORT_RATES), cudaMemcpyHostToDevice));
	for (k=0; k<NUM_OF_CURRENCY; k++) {
		for (n=0; n<NUM_OF_THREADS; n++) {
			checkCudaErrors(cudaMemcpy(rs->x[k][n], d_x[k][n], sizeof(double)*time_stamps->length_sortedT, cudaMemcpyHostToDevice));
		}
	}

	//reswap
	for (k=0; k<NUM_OF_CURRENCY; k++) {
		for (n=0; n<NUM_OF_THREADS; n++) {
			p_double_temp = rs->x[k][n];
			rs->x[k][n]= d_x[k][n];
			d_x[k][n]= p_double_temp;
		}
	}

	/*****
	Discount Factors
	*****/
	DISCOUNT_FACTORS *Ds;
	Ds = (DISCOUNT_FACTORS *)malloc(sizeof(DISCOUNT_FACTORS));
	for (k=0; k<NUM_OF_CURRENCY; k++) {
		for (n=0; n<NUM_OF_THREADS; n++) {
			Ds->D[k][n] = (double *)malloc(sizeof(double)*time_stamps->length_sortedT);
			Ds->p_D[k][n] = (double *)malloc(sizeof(double)*time_stamps->length_sortedT*NUM_OF_MARKET_DATA);
		}
	}

	//device malloc for discount factors
	DISCOUNT_FACTORS *d_Ds;
	double *d_D[NUM_OF_CURRENCY][NUM_OF_THREADS];
	double *d_p_D[NUM_OF_CURRENCY][NUM_OF_THREADS];
	checkCudaErrors(cudaMalloc((void**)&d_Ds, sizeof(DISCOUNT_FACTORS)));
	for (k=0; k<NUM_OF_CURRENCY; k++) {
		for (n=0; n<NUM_OF_THREADS; n++) {
			checkCudaErrors(cudaMalloc((void**)&d_D[k][n], (sizeof(double)*time_stamps->length_sortedT)));
			checkCudaErrors(cudaMalloc((void**)&d_p_D[k][n], (sizeof(double)*time_stamps->length_sortedT*NUM_OF_MARKET_DATA)));
		}
	}

	//set initial data
	for (k=0; k<NUM_OF_CURRENCY; k++) {
		for (n=0; n<NUM_OF_THREADS; n++) {
			for(i=0; i<time_stamps->length_sortedT; i++){
				Ds->D[k][n][i]=0.0;
			}
			for(i=0; i<time_stamps->length_sortedT*NUM_OF_MARKET_DATA; i++){
				Ds->p_D[k][n][i]=0.0;
			}
		}
	}

	//temporally swap Adresses for copying struct
	for (k=0; k<NUM_OF_CURRENCY; k++) {
		for (n=0; n<NUM_OF_THREADS; n++) {
			p_double_temp = Ds->D[k][n];
			Ds->D[k][n]= d_D[k][n];
			d_D[k][n]= p_double_temp;

			p_double_temp = Ds->p_D[k][n];
			Ds->p_D[k][n]= d_p_D[k][n];
			d_p_D[k][n]= p_double_temp;
		}
	}

	//copy struct from host to device
	checkCudaErrors(cudaMemcpy(d_Ds, Ds, sizeof(DISCOUNT_FACTORS), cudaMemcpyHostToDevice));
	for (k=0; k<NUM_OF_CURRENCY; k++) {
		for (n=0; n<NUM_OF_THREADS; n++) {
			checkCudaErrors(cudaMemcpy(Ds->D[k][n], d_D[k][n], sizeof(double)*time_stamps->length_sortedT, cudaMemcpyHostToDevice));
			checkCudaErrors(cudaMemcpy(Ds->p_D[k][n], d_p_D[k][n], sizeof(double)*time_stamps->length_sortedT*NUM_OF_MARKET_DATA, cudaMemcpyHostToDevice));
		}
	}

	//reswap
	for (k=0; k<NUM_OF_CURRENCY; k++) {
		for (n=0; n<NUM_OF_THREADS; n++) {
			p_double_temp = Ds->D[k][n];
			Ds->D[k][n]= d_D[k][n];
			d_D[k][n]= p_double_temp;

			p_double_temp = Ds->p_D[k][n];
			Ds->p_D[k][n]= d_p_D[k][n];
			d_p_D[k][n]= p_double_temp;
		}
	}

	/*****
	ZCB
	*****/
	ZCBS *zcbs;
	zcbs = (ZCBS *)malloc(sizeof(ZCBS));
	for (k=0; k<NUM_OF_CURRENCY; k++) {
		for (n=0; n<NUM_OF_THREADS; n++) {
			zcbs->P[k][n] = (double *)malloc(sizeof(double)*time_stamps->length_sortedT*time_stamps->length_sortedT);
		}
		zcbs->p_P[k] = (double *)malloc(sizeof(double)*time_stamps->length_sortedT*NUM_OF_MARKET_DATA);
	}

	//device malloc for ZCB
	ZCBS *d_zcbs;
	double *d_P[NUM_OF_CURRENCY][NUM_OF_THREADS];
	double *d_p_P[NUM_OF_CURRENCY];
	checkCudaErrors(cudaMalloc((void**)&d_zcbs, sizeof(ZCBS)));
	for (k=0; k<NUM_OF_CURRENCY; k++) {
		for (n=0; n<NUM_OF_THREADS; n++) {
			checkCudaErrors(cudaMalloc((void**)&d_P[k][n], (sizeof(double)*time_stamps->length_sortedT*time_stamps->length_sortedT)));
		}
		checkCudaErrors(cudaMalloc((void**)&d_p_P[k], (sizeof(double)*time_stamps->length_sortedT*NUM_OF_MARKET_DATA)));
	}
	//set initial data
	for(k=0; k<NUM_OF_CURRENCY; k++){
		for (n=0; n<NUM_OF_THREADS; n++) {
			for (l=0; l<time_stamps->length_sortedT*time_stamps->length_sortedT; l++) {
				zcbs->P[k][n][l]=0.0;
			}
		}
	}
	for (n=0; n<NUM_OF_THREADS; n++) {
		zcbs->P[0][n][0]=1.0;
		zcbs->P[1][n][0]=1.0;
		//T0
		//for (l=0; l<time_stamps->length_T0; l++) {
		for (l=0; l<time_stamps->length_sortedT; l++) {
			zcbs->P[0][n][l] = exp(-0.05 * time_stamps->sortedT[l]);	
		//	zcbs->P[0][n][time_stamps->T0[l]] = exp(-0.05 * time_stamps->sortedT[time_stamps->T0[l]]);	
			zcbs->P[1][n][l] = exp(-0.05 * time_stamps->sortedT[l]);	
		//	zcbs->P[1][n][time_stamps->T0[l]] =  exp(-0.05 * time_stamps->sortedT[time_stamps->T0[l]]);	
		}
		for (k=0; k<NUM_OF_CURRENCY; k++) {
			for (j=0; j<NUM_OF_SWAPS; j++) {
				//for (l=0; l<time_stamps->length_Ti[k][j]; l++) {
				for (l=0; l<time_stamps->length_sortedT; l++) {
					zcbs->P[k][n][l] = exp(-0.05 * time_stamps->sortedT[l]);	  
					//zcbs->P[k][n][time_stamps->Ti[k][j][l]] = exp(-0.05 * time_stamps->sortedT[time_stamps->Ti[k][j][l]]);	  
				}
			}
		}
	}
	for (j=0; j<NUM_OF_SWAPS; j++) {
		for (l=0; l<time_stamps->length_sortedT; l++) {
			for (m=0; m<NUM_OF_MARKET_DATA; m++) {
				if (l==m) {
					zcbs->p_P[Delta_Currency][time_stamps->Ti[Delta_Currency][j][l]*NUM_OF_MARKET_DATA + time_stamps->Ti[Delta_Currency][j][l]] = 1.0;
				} else {
					zcbs->p_P[Delta_Currency][time_stamps->Ti[Delta_Currency][j][l]*NUM_OF_MARKET_DATA + time_stamps->Ti[Delta_Currency][j][l]] = 0.0;
				}
			}
		}
	}

	//temporally swap Adresses for copying struct
	for (k=0; k<NUM_OF_CURRENCY; k++) {
		for (n=0; n<NUM_OF_THREADS; n++) {
			p_double_temp = zcbs->P[k][n];
			zcbs->P[k][n]= d_P[k][n];
			d_P[k][n]= p_double_temp;

		}
		p_double_temp = zcbs->p_P[k];
		zcbs->p_P[k]= d_p_P[k];
		d_p_P[k]= p_double_temp;
	}

	//copy struct from host to device
	checkCudaErrors(cudaMemcpy(d_zcbs, zcbs, sizeof(ZCBS), cudaMemcpyHostToDevice));
	for (k=0; k<NUM_OF_CURRENCY; k++) {
		for (n=0; n<NUM_OF_THREADS; n++) {
			checkCudaErrors(cudaMemcpy(zcbs->P[k][n], d_P[k][n], sizeof(double)*time_stamps->length_sortedT*time_stamps->length_sortedT, cudaMemcpyHostToDevice));
		}
		checkCudaErrors(cudaMemcpy(zcbs->p_P[k], d_p_P[k], sizeof(double)*time_stamps->length_sortedT*NUM_OF_MARKET_DATA, cudaMemcpyHostToDevice));
	}

	//reswap
	for (k=0; k<NUM_OF_CURRENCY; k++) {
		for (n=0; n<NUM_OF_THREADS; n++) {
			p_double_temp = zcbs->P[k][n];
			zcbs->P[k][n]= d_P[k][n];
			d_P[k][n]= p_double_temp;
		}
		for (n=0; n<time_stamps->length_sortedT*NUM_OF_MARKET_DATA; n++) {
			p_double_temp = zcbs->p_P[k];
			zcbs->p_P[k]= d_p_P[k];
			d_p_P[k]= p_double_temp;
		}
	}

	/*****
	FX
	*****/
	SPOT_FXS *spot_fxs;
	spot_fxs = (SPOT_FXS *)malloc(sizeof(SPOT_FXS));
	for (k = 0; k<NUM_OF_CURRENCY-1; k++){
		for(n=0; n<NUM_OF_THREADS; n++){
			spot_fxs->spot_fx[k][n]=(double*)malloc(sizeof(double)*time_stamps->length_sortedT);
			spot_fxs->p_spot_fx[k][n]=(double*)malloc(sizeof(double)*time_stamps->length_sortedT*NUM_OF_MARKET_DATA);
		}
	}
	//device malloc for FX
	SPOT_FXS *d_spot_fxs;
	double *d_spot_fx[NUM_OF_CURRENCY-1][NUM_OF_THREADS];
	double *d_p_spot_fx[NUM_OF_CURRENCY-1][NUM_OF_THREADS];
	checkCudaErrors(cudaMalloc((void**)&d_spot_fxs, sizeof(SPOT_FXS)));
	for (k=0; k<NUM_OF_CURRENCY-1; k++) {
		for (n=0; n<NUM_OF_THREADS; n++) {
			checkCudaErrors(cudaMalloc((void**)&d_spot_fx[k][n], (sizeof(double*)*time_stamps->length_sortedT)));
			checkCudaErrors(cudaMalloc((void**)&d_p_spot_fx[k][n], (sizeof(double*)*time_stamps->length_sortedT*NUM_OF_MARKET_DATA)));
		}
	}

	//set initial data
	for (k=0; k<NUM_OF_CURRENCY-1; k++) {
		for(n=0; n<NUM_OF_THREADS; n++){
			for(i=0; i<time_stamps->length_sortedT; i++){
				//domestic/foreign
				spot_fxs->spot_fx[k][n][i]=0.0;
			}
			for(i=0; i<time_stamps->length_sortedT*NUM_OF_MARKET_DATA; i++){
				spot_fxs->p_spot_fx[k][n][i]=0.0;
			}
		}
	}

	for(n=0; n<NUM_OF_THREADS; n++){
		//domestic/foreign
		spot_fxs->spot_fx[0][n][0]=120.0;
	}

	//temporally swap Adresses for copying struct
	for(k=0; k<NUM_OF_CURRENCY-1; k++){
		for(n=0; n<NUM_OF_THREADS; n++){
			p_double_temp = spot_fxs->spot_fx[k][n];
			spot_fxs->spot_fx[k][n]=d_spot_fx[k][n];
			d_spot_fx[k][n]=p_double_temp;

			p_double_temp = spot_fxs->p_spot_fx[k][n];
			spot_fxs->p_spot_fx[k][n]=d_p_spot_fx[k][n];
			d_p_spot_fx[k][n]=p_double_temp;
		}
	}

	//copy struct from host to device
	checkCudaErrors(cudaMemcpy(d_spot_fxs, spot_fxs, sizeof(SPOT_FXS), cudaMemcpyHostToDevice));
	for (k=0; k<NUM_OF_CURRENCY-1; k++) {
		for (n=0; n<NUM_OF_THREADS; n++) {
			checkCudaErrors(cudaMemcpy(spot_fxs->spot_fx[k][n], d_spot_fx[k][n], sizeof(double)*time_stamps->length_sortedT, cudaMemcpyHostToDevice));
			checkCudaErrors(cudaMemcpy(spot_fxs->p_spot_fx[k][n], d_p_spot_fx[k][n], sizeof(double)*time_stamps->length_sortedT*NUM_OF_MARKET_DATA, cudaMemcpyHostToDevice));
		}
	}

	//reswap
	for(k=0; k<NUM_OF_CURRENCY-1; k++){
		for(n=0; n<NUM_OF_THREADS; n++){
			p_double_temp = spot_fxs->spot_fx[k][n];
			spot_fxs->spot_fx[k][n]=d_spot_fx[k][n];
			d_spot_fx[k][n]=p_double_temp;

			p_double_temp = spot_fxs->p_spot_fx[k][n];
			spot_fxs->p_spot_fx[k][n]=d_p_spot_fx[k][n];
			d_p_spot_fx[k][n]=p_double_temp;
		}
	}
	
	/**
	SWAP
	*****/
	SWAPS *swaps;
	swaps = (SWAPS*)malloc(sizeof(SWAPS));
	for (k=0; k<NUM_OF_CURRENCY; k++) {
		for (j=0; j<NUM_OF_SWAPS; j++) {
			for(n=0; n<NUM_OF_THREADS; n++){
				swaps->swap[k][j][n] = (double *)malloc(sizeof(double)*time_stamps->length_sortedT);
				swaps->p_swap[k][j][n] = (double *)malloc(sizeof(double)*time_stamps->length_sortedT*NUM_OF_MARKET_DATA);
			}
		}
	}
	//device malloc for swaps
	SWAPS *d_swaps;
	double *d_swap[NUM_OF_CURRENCY][NUM_OF_SWAPS][NUM_OF_THREADS];
	double *d_p_swap[NUM_OF_CURRENCY][NUM_OF_SWAPS][NUM_OF_THREADS];
	checkCudaErrors(cudaMalloc((void**)&d_swaps, sizeof(SWAPS)));
	for (k=0; k<NUM_OF_CURRENCY; k++) {
		for (j=0; j<NUM_OF_SWAPS; j++) {
			for(n=0; n<NUM_OF_THREADS; n++){
				checkCudaErrors(cudaMalloc((void**)&d_swap[k][j][n], sizeof(double)*time_stamps->length_sortedT));
				checkCudaErrors(cudaMalloc((void**)&d_p_swap[k][j][n], sizeof(double)*time_stamps->length_sortedT*NUM_OF_MARKET_DATA));
			}
		}
	}

	for (k=0; k<NUM_OF_CURRENCY; k++) {
		for (j=0; j<NUM_OF_SWAPS; j++) {
			for(n=0; n<NUM_OF_THREADS; n++){
				for(i=0; i<time_stamps->length_sortedT; i++){
					swaps->swap[k][j][n][i] = 0.0;
				}
				for(i=0; i<time_stamps->length_sortedT*NUM_OF_MARKET_DATA; i++){
					swaps->p_swap[k][j][n][i] = 0.0;
				}
				checkCudaErrors(cudaMemcpy(d_swap[k][j][n], swaps->swap[k][j][n], sizeof(double)*time_stamps->length_sortedT, cudaMemcpyHostToDevice));
				checkCudaErrors(cudaMemcpy(d_p_swap[k][j][n], swaps->p_swap[k][j][n], sizeof(double)*time_stamps->length_sortedT*NUM_OF_MARKET_DATA, cudaMemcpyHostToDevice));
			}
		}
	}

	//set initial data
	swaps->N[0][0]=10000.0;
	swaps->N[0][1]=20000.0;
	swaps->N[1][0]=10000.0;
	swaps->N[1][1]=20000.0;
	swaps->K[0][0]=0.046;
	swaps->K[0][1]=0.046;
	swaps->K[1][0]=0.046;
	swaps->K[1][1]=0.046;
	//kokorahen ayashii
	//temporally swap Adresses for copying struct
	for (k=0; k<NUM_OF_CURRENCY; k++) {
		for (j=0; j<NUM_OF_SWAPS; j++) {
			for (n=0; n<NUM_OF_THREADS; n++) {
				p_double_temp = swaps->swap[k][j][n];
				swaps->swap[k][j][n]= d_swap[k][j][n];
				d_swap[k][j][n]= p_double_temp;

				p_double_temp = swaps->p_swap[k][j][n];
				swaps->p_swap[k][j][n]= d_p_swap[k][j][n];
				d_p_swap[k][j][n]= p_double_temp;
			}
		}
	}
	//copy struct from host to device
	checkCudaErrors(cudaMemcpy(d_swaps,swaps, sizeof(SWAPS), cudaMemcpyHostToDevice));
	//reswap
	for (k=0; k<NUM_OF_CURRENCY; k++) {
		for (j=0; j<NUM_OF_SWAPS; j++) {
			for (n=0; n<NUM_OF_THREADS; n++) {
				p_double_temp = swaps->swap[k][j][n];
				swaps->swap[k][j][n]= d_swap[k][j][n];
				d_swap[k][j][n]= p_double_temp;

				p_double_temp = swaps->p_swap[k][j][n];
				swaps->p_swap[k][j][n]= d_p_swap[k][j][n];
				d_p_swap[k][j][n]= p_double_temp;
			}
		}
	}


	/*****
	nset
	*****/
	NSETS *nsets;
	nsets = (NSETS*)malloc(sizeof(NSETS));
	for(n=0; n<NUM_OF_THREADS; n++){
		nsets->nset[n] = (double *)malloc(sizeof(double)*time_stamps->length_sortedT);
		nsets->p_nset[n] = (double *)malloc(sizeof(double)*time_stamps->length_sortedT*NUM_OF_MARKET_DATA);
	}
	//device malloc for nset
	NSETS *d_nsets;
	double *d_nset[NUM_OF_THREADS];
	double *d_p_nset[NUM_OF_THREADS];

	checkCudaErrors(cudaMalloc((void**)&d_nsets,sizeof(NSETS)));
	for(n=0; n<NUM_OF_THREADS; n++){
		checkCudaErrors(cudaMalloc((void**)&d_nset[n], (sizeof(double*)*time_stamps->length_sortedT)));
		checkCudaErrors(cudaMalloc((void**)&d_p_nset[n], (sizeof(double*)*time_stamps->length_sortedT*NUM_OF_MARKET_DATA)));
	}

	for(n=0; n<NUM_OF_THREADS; n++){
		for(i=0; i<time_stamps->length_sortedT; i++){
			nsets->nset[n][i] = 0.0;
		}
	   for(i=0; i<time_stamps->length_sortedT*NUM_OF_MARKET_DATA; i++){
			nsets->p_nset[n][i] = 0.0;
		}
		checkCudaErrors(cudaMemcpy(d_nset[n], nsets->nset[n], sizeof(double)*time_stamps->length_sortedT, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_p_nset[n], nsets->p_nset[n], sizeof(double)*time_stamps->length_sortedT*NUM_OF_MARKET_DATA, cudaMemcpyHostToDevice));
	}

	//temporally swap Adresses for copying struct
	for (n=0; n<NUM_OF_THREADS; n++){
		p_double_temp = nsets->nset[n];
		nsets->nset[n] = d_nset[n];
		d_nset[n] = p_double_temp;

		p_double_temp = nsets->p_nset[n];
		nsets->p_nset[n] = d_p_nset[n];
		d_p_nset[n] = p_double_temp;
	}
	//copy struc from host to device
	checkCudaErrors(cudaMemcpy(d_nsets, nsets, sizeof(NSETS), cudaMemcpyHostToDevice));
	//reswap
	for (n=0; n<NUM_OF_THREADS; n++){
		p_double_temp = nsets->nset[n];
		nsets->nset[n] = d_nset[n];
		d_nset[n] = p_double_temp;

		p_double_temp = nsets->p_nset[n];
		nsets->p_nset[n] = d_p_nset[n];
		d_p_nset[n] = p_double_temp;
	}
	
	/*******
	default prob
	*******/
	//malloc on host
	DEFAULT_PROBABILITY *dp;
	dp = (DEFAULT_PROBABILITY *)malloc(sizeof(DEFAULT_PROBABILITY));
	//device malloc for default prob
	DEFAULT_PROBABILITY *d_dp;
	checkCudaErrors(cudaMalloc((void**)&d_dp, sizeof(DEFAULT_PROBABILITY)));
	//set function pointer
	checkCudaErrors(cudaMemcpyFromSymbol(&dp->lambda, d_dp_lambda, sizeof(FUNC_LAMBDA)));
	//memory copy to device
	checkCudaErrors(cudaMemcpy(d_dp, dp, sizeof(DEFAULT_PROBABILITY), cudaMemcpyHostToDevice));

	
	/*****
	CVA Delta
	*****/
	double *CVA_Delta;
	CVA_Delta = (double *)malloc(sizeof(double)*NUM_OF_THREADS*NUM_OF_MARKET_DATA);
	//device malloc for CVA
	double *d_CVA_Delta;
	checkCudaErrors(cudaMalloc((void**)&d_CVA_Delta,(sizeof(double)*NUM_OF_THREADS*NUM_OF_MARKET_DATA)));
	for (n=0; n<NUM_OF_THREADS*NUM_OF_MARKET_DATA; n++) {
		CVA_Delta[n] = 0.0;
	}
	checkCudaErrors(cudaMemcpy(d_CVA_Delta, CVA_Delta, sizeof(double)*NUM_OF_THREADS*NUM_OF_MARKET_DATA, cudaMemcpyHostToDevice));

	/****
	Generate Brownian Paths
	****/
	//Create the correlation matrix R
	double **R;
	R = (double **)malloc(sizeof(double *)*(2*NUM_OF_CURRENCY-1));
	for(i=0; i<2*NUM_OF_CURRENCY-1; i++){
		R[i] = (double *)malloc(sizeof(double)*(2*NUM_OF_CURRENCY-1));
		for(j=0; j<2*NUM_OF_CURRENCY-1; j++){
			if(i == j){
				R[i][j]=1.0;
			}else{
			R[i][j]=0.0;
			}
		}
	}
	//malloc on host
	double *h_W;
	h_W = (double *)malloc(sizeof(double)*(2*NUM_OF_CURRENCY-1)* time_stamps->length_sortedT*NUM_OF_THREADS);
	//malloc on device
	double *d_W;
	checkCudaErrors(cudaMalloc((void**)&d_W, sizeof(double)*(2*NUM_OF_CURRENCY-1)*time_stamps->length_sortedT*NUM_OF_THREADS));
	//generate browninan paths
	generate_brawnian_paths(h_W, NUM_OF_CURRENCY, R, time_stamps->length_sortedT, NUM_OF_THREADS);
	
	//memory copy to device
	checkCudaErrors(cudaMemcpy(d_W, h_W, sizeof(double)*(2*NUM_OF_CURRENCY-1)*time_stamps->length_sortedT*NUM_OF_THREADS, cudaMemcpyHostToDevice));
	

	double *outputs;
	outputs = (double *)malloc(sizeof(double)*NUM_OF_CURRENCY*time_stamps->length_sortedT*NUM_OF_THREADS);
	for(i=0; i<NUM_OF_CURRENCY*time_stamps->length_sortedT*NUM_OF_THREADS; i++){
		outputs[i]=0.0;
	}

	double *d_outputs;
	checkCudaErrors(cudaMalloc((void**)&d_outputs, sizeof(double)*NUM_OF_CURRENCY*time_stamps->length_sortedT*NUM_OF_THREADS));

	int bd = 32;
	int gd = NUM_OF_THREADS/32;

	checkCudaErrors(cudaMemcpy(d_outputs, outputs, sizeof(double)*NUM_OF_CURRENCY*time_stamps->length_sortedT*NUM_OF_THREADS, cudaMemcpyHostToDevice));

	Calc_CVA_Delta<<<bd,gd>>>(
	d_W, 
	d_CVA_Delta,
	d_rs, 
	d_Ds, 
	d_zcbs, 
	d_swaps, 
	d_spot_fxs, 
	d_nsets, 
	d_dp, 
	d_time_stamps, 
	NUM_OF_THREADS, 
	time_stamps->length_sortedT,
	0,
	d_outputs);

	checkCudaErrors(cudaMemcpy(CVA_Delta, d_CVA_Delta, sizeof(double)*NUM_OF_THREADS*NUM_OF_MARKET_DATA, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(outputs, d_outputs, sizeof(double)*NUM_OF_CURRENCY*time_stamps->length_sortedT*NUM_OF_THREADS, cudaMemcpyDeviceToHost));
	
	for (k=0; k<NUM_OF_CURRENCY; k++) {
		for (n=0; n<NUM_OF_THREADS; n++) {
			checkCudaErrors(cudaMemcpy(rs->x[k][n], d_x[k][n], sizeof(double)*time_stamps->length_sortedT, cudaMemcpyDeviceToHost));
		}
	}
	
	for (k=0; k<NUM_OF_CURRENCY; k++) {
		for (n=0; n<NUM_OF_THREADS; n++) {
			checkCudaErrors(cudaMemcpy(Ds->D[k][n], d_D[k][n], sizeof(double)*time_stamps->length_sortedT, cudaMemcpyDeviceToHost));
			checkCudaErrors(cudaMemcpy(Ds->p_D[k][n], d_p_D[k][n], sizeof(double)*time_stamps->length_sortedT*NUM_OF_MARKET_DATA, cudaMemcpyDeviceToHost));
		}
	}
	for (k=0; k<NUM_OF_CURRENCY; k++) {
		for (n=0; n<NUM_OF_THREADS; n++) {
			checkCudaErrors(cudaMemcpy(zcbs->P[k][n], d_P[k][n], sizeof(double)*time_stamps->length_sortedT*time_stamps->length_sortedT, cudaMemcpyDeviceToHost));
		}
		checkCudaErrors(cudaMemcpy(zcbs->p_P[k], d_p_P[k], sizeof(double)*time_stamps->length_sortedT*NUM_OF_MARKET_DATA, cudaMemcpyDeviceToHost));
	}
	for (k=0; k<NUM_OF_CURRENCY; k++) {
		for (j=0; j<NUM_OF_SWAPS; j++) {
			for(n=0; n<NUM_OF_THREADS; n++){
				checkCudaErrors(cudaMemcpy(swaps->swap[k][j][n],d_swap[k][j][n], sizeof(double)*time_stamps->length_sortedT,cudaMemcpyDeviceToHost));
				checkCudaErrors(cudaMemcpy(swaps->p_swap[k][j][n],d_p_swap[k][j][n], sizeof(double)*time_stamps->length_sortedT*NUM_OF_MARKET_DATA,cudaMemcpyDeviceToHost));
			}
		}
	}

	for (k=0; k<NUM_OF_CURRENCY-1; k++) {
		for (n=0; n<NUM_OF_THREADS; n++) {
			checkCudaErrors(cudaMemcpy(spot_fxs->spot_fx[k][n], d_spot_fx[k][n], sizeof(double)*time_stamps->length_sortedT, cudaMemcpyDeviceToHost));
			checkCudaErrors(cudaMemcpy(spot_fxs->p_spot_fx[k][n], d_p_spot_fx[k][n], sizeof(double)*time_stamps->length_sortedT*NUM_OF_MARKET_DATA, cudaMemcpyDeviceToHost));
		}
	}

	for(n=0; n<NUM_OF_THREADS; n++){
		checkCudaErrors(cudaMemcpy(nsets->nset[n], d_nset[n], sizeof(double)*time_stamps->length_sortedT, cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(nsets->p_nset[n], d_p_nset[n], sizeof(double)*time_stamps->length_sortedT*NUM_OF_MARKET_DATA, cudaMemcpyDeviceToHost));
	}

	
	printf("rs->x\n");
	for (k=0; k<NUM_OF_CURRENCY; k++){
		printf("k:%d\n" , k);
		for (n=0; n<20; n++){
			for(j=0; j<time_stamps->length_sortedT; j++){
				printf("%5.2lf ", rs->x[k][n][j]);
			}
			printf("\n");
		}
	}

	printf("Ds->D\n");
	for (k=0; k<NUM_OF_CURRENCY; k++){
		printf("k:%d\n" , k);
		for (n=0; n<20; n++){
			for(j=0; j<time_stamps->length_sortedT; j++){
				printf("%5.2lf ", Ds->D[k][n][j]);
			}
			printf("\n");
		}
	}

	printf("Ds->p_D\n");
	for (k=0; k<NUM_OF_CURRENCY; k++){
		for (n=0; n<3; n++){
			printf("k:%d, n:%d\n" , k, n);
			for(j=0; j<time_stamps->length_sortedT; j++){
				for (l=0; l<NUM_OF_MARKET_DATA; l++) {
					printf("%5.2lf ", Ds->p_D[k][n][j*NUM_OF_MARKET_DATA + l]);
				}
				printf("\n");
			}
			printf("\n");
		}
	}

	printf("zcb->P\n");
	for (k=0; k<NUM_OF_CURRENCY; k++) {
		for (n=0; n<2; n++) {
			printf("k:%d n:%d\n", k, n);
			for(i=0; i<time_stamps->length_sortedT; i++){
				printf("%5d ",i);
			}
			printf("\n");
			for (j=0; j<time_stamps->length_sortedT; j++) {
				for (i=0; i<time_stamps->length_sortedT; i++) {
					printf("%5.2lf ", zcbs->P[k][n][j*time_stamps->length_sortedT + i]);
				}
				printf("\n");
			}
		}
	}
	printf("zcb->p_P\n");
	for (k=0; k<NUM_OF_CURRENCY; k++) {
		printf("k:%d \n", k);
		for(i=0; i<time_stamps->length_sortedT; i++){
			for (j=0; j<NUM_OF_MARKET_DATA; j++) {
				printf("%5.2lf ", zcbs->p_P[k][i*NUM_OF_MARKET_DATA + j]);
			}
			printf("\n");
		}
	}

	printf("swap->swap\n");
	for (k=0; k<NUM_OF_CURRENCY; k++){
		for (j=0; j<NUM_OF_SWAPS; j++) {
			printf("k:%d j:%d\n" , k, j);
			for (n=0; n<2; n++){
				for(i=0; i<time_stamps->length_sortedT; i++){
					printf("%5.2lf ", swaps->swap[k][n][j][i]);
				}
				printf("\n");
			}
		}
	}
	printf("swap->p_swap\n");
	for (k=0; k<NUM_OF_CURRENCY; k++){
		for (j=0; j<NUM_OF_SWAPS; j++) {
			printf("k:%d j:%d\n" , k, j);
			for (n=0; n<2; n++){
				for(i=0; i<time_stamps->length_sortedT; i++){
					for (l=0; l<NUM_OF_MARKET_DATA; l++) {
						printf("%5.2lf ", swaps->p_swap[k][n][j][i*NUM_OF_MARKET_DATA + l]);
					}
					printf("\n");
				}
			}
		}
	}

	printf("spot_fx->spot_fx\n");
	for (k=0; k<NUM_OF_CURRENCY-1; k++){
		printf("k:%d\n" , k);
		for (n=0; n<2; n++){
			for(j=0; j<time_stamps->length_sortedT; j++){
				printf("%5.2lf ", spot_fxs->spot_fx[k][n][j]);
			}
			printf("\n");
		}
	}

	printf("spot_fx->p_spot_fx\n");
	for (k=0; k<NUM_OF_CURRENCY-1; k++){
		for (n=0; n<2; n++){
			printf("k:%d,n:%d\n" , k, n);
			for(j=0; j<time_stamps->length_sortedT; j++){
				for (l=0; l<NUM_OF_MARKET_DATA; l++) {
					printf("%5.2lf ", spot_fxs->p_spot_fx[k][n][j*NUM_OF_MARKET_DATA + l]);
				}
				printf("\n");
			}
		}
	}

	printf("nsets->nset\n");
	for (n=0; n<20; n++){
		for(j=0; j<time_stamps->length_sortedT; j++){
			printf("%5.2lf ", nsets->nset[n][j]);
		}
		printf("\n");
	}
	printf("nsets->p_nset\n");
	for (n=0; n<2; n++){
		printf("n:%d\n", n);
		for(j=0; j<time_stamps->length_sortedT; j++){
			for(l=0; l<NUM_OF_MARKET_DATA; l++){
				printf("%5.2lf ", nsets->p_nset[n][j*NUM_OF_MARKET_DATA + l]);
			}
			printf("\n");
		}
	}

	printf("CVA_Delta\n");
	for (n=0; n<20; n++) {
		for (j=0; j<NUM_OF_MARKET_DATA; j++) {
			printf("%+5.4lf ", CVA_Delta[n*NUM_OF_MARKET_DATA + j]);
		}
		printf("\n");
	}

	printf("outputs\n");
	for(k=0; k<NUM_OF_CURRENCY;k++){
		for (n=0; n<20; n++) {
			for (j=0; j<time_stamps->length_sortedT; j++) {
				printf("%+5.4lf ", outputs[k*time_stamps->length_sortedT*NUM_OF_THREADS + n*time_stamps->length_sortedT + j]);
			}
			printf("\n");
		}
		printf("\n");
	}
	/*
	printf("Wd\n");
	for(i=0; i<20; i++){
		for(k=0; k<time_stamps->length_sortedT; k++){
			printf("%+5.4lf ", h_W[k+i*time_stamps->length_sortedT]);
		}
		printf("\n");	
	}
	printf("\n");
	printf("Wf\n");
	for(j=1; j<NUM_OF_CURRENCY; j++){
		for(i=0; i<20; i++){
			for(k=0; k<time_stamps->length_sortedT; k++){
				printf("%+5.4lf ", h_W[j*NUM_OF_THREADS*time_stamps->length_sortedT + k + i*time_stamps->length_sortedT]);
			}
			printf("\n");	
		}
	}
	printf("Ws\n");
	for(j=NUM_OF_CURRENCY; j<2*NUM_OF_CURRENCY-1; j++){
		for(i=0; i<20; i++){
			for(k=0; k<time_stamps->length_sortedT; k++){
				printf("%+5.4lf ", h_W[j*NUM_OF_THREADS*time_stamps->length_sortedT + k + i*time_stamps->length_sortedT]);
			}
			printf("\n");	
		}
	}
	*/

	for(i=0; i<NUM_OF_CURRENCY; i++){
		free(Ti[i]);
	}
	free(Ti);
	//cudafree
	for(k=0; k<NUM_OF_CURRENCY; k++){
		for(j=0; j<NUM_OF_SWAPS; j++){
			free(time_stamps->Ti[k][j]);
			cudaFree(d_Ti[k][j]);
		}
	}
	free(time_stamps->T0);
	free(time_stamps->sortedT);
	cudaFree(d_time_stamps);
	cudaFree(d_T0);
	cudaFree(d_sortedT);
	free(time_stamps);

	for (k=0; k<NUM_OF_CURRENCY; k++) {
		for (n=0; n<NUM_OF_THREADS; n++) {
			free(rs->x[k][n]);
			cudaFree(d_x[k][n]);
		}
	}
	cudaFree(d_rs);
	free(rs);
	
	for (k=0; k<NUM_OF_CURRENCY; k++) {
		for (n=0; n<NUM_OF_THREADS; n++) {
		free(Ds->D[k][n]);
		free(Ds->p_D[k][n]);
		cudaFree(d_D[k][n]);
		cudaFree(d_p_D[k][n]);
		}
	}
	cudaFree(d_Ds);
	free(Ds);
	
	for (k=0; k<NUM_OF_CURRENCY; k++) {
		for (n=0; n<NUM_OF_THREADS; n++) {
			free(zcbs->P[k][n]); 
			cudaFree(d_P[k][n]);
		}
		free(zcbs->p_P[k]); 
		cudaFree(d_p_P[k]);
	}
	free(zcbs);
	cudaFree(d_zcbs);

	for (k=0; k<NUM_OF_CURRENCY-1; k++) {
		for (n=0; n<NUM_OF_THREADS; n++) {
			free(spot_fxs->spot_fx[k][n]); 
			free(spot_fxs->p_spot_fx[k][n]);
			cudaFree(d_spot_fx[k][n]);
			cudaFree(d_p_spot_fx[k][n]);
		}
	}
	cudaFree(d_spot_fxs);
	free(spot_fxs);

	for (k=0; k<NUM_OF_CURRENCY; k++) {
		for (j=0; j<NUM_OF_SWAPS; j++) {
			for(n=0; n<NUM_OF_THREADS; n++){
				free(swaps->swap[k][j][n]);
				free(swaps->p_swap[k][j][n]);
				cudaFree(d_swap[k][j][n]);
				cudaFree(d_p_swap[k][j][n]);
			}
		}
	}
	cudaFree(d_swaps);
	free(swaps);

	for(n=0; n<NUM_OF_THREADS; n++){
		free(nsets->nset[n]);
		free(nsets->p_nset[n]);
		cudaFree(d_nset[n]);
		cudaFree(d_p_nset[n]);
	}
	cudaFree(d_nsets);
	free(nsets);

	free(dp);
	cudaFree(d_dp);

	free(CVA_Delta);
	cudaFree(d_CVA_Delta);

	for (i=0; i<2*NUM_OF_CURRENCY-1; i++) {
		free(R[i]);
	}
	free(R);
	free(h_W);
	cudaFree(d_W);

}



