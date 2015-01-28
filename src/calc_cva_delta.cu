#include <stdio.h>	
#include <stdlib.h>
#include <math.h>
#include <random_generator.h>
#include <helper_cuda.h>
#include <mt19937.h>

#include <cva_common.h>
#include <short_rates.h>
#include <default_probability.h>
#include <discount_factors.h>
#include <zero_coupon_bonds.h>
#include <swaps.h>
#include <spot_FXs.h>
#include <nsets.h>
#include <utilities.h>
#include <calc_cva_delta.h>

//__global__ void Calc_CVA_Delta(
//		double *randoms,
//		double *CVA_Delta,
//		SHORT_RATES *r,
//		DISCOUNT_FACTORS *discount_factor,
//		ZCBS *zcb,
//		SWAPS *swap,
//		SPOT_FXS *spot_fx,
//		NSETS *nset,
//		DEFAULT_PROBABILITY *default_probability,
//		TIME_STAMPS *time_stamps,
//		int N,
//		int M,
//		int g_c,
//		double *outputs
//) 
//{
//
//	int i,j,k,l,m,n,mm;
//	double sum;
//	int threadId = threadIdx.x + threadIdx.y*blockDim.x + blockDim.x*blockDim.y*blockIdx.x;
//	
//	//domestic curency
//	double *Wd = randoms + threadId *time_stamps->length_sortedT;
//	//foreign currency
//	double *Wf[NUM_OF_CURRENCY-1];
//	for (k=0; k<NUM_OF_CURRENCY-1; k++) {
//		Wf[k] = randoms + threadId*time_stamps->length_sortedT + N*time_stamps->length_sortedT*(k+1);
//	}
//	//FX domestic/foreign
//	double *Ws[NUM_OF_CURRENCY-1];
//	for (k=0; k<NUM_OF_CURRENCY-1; k++) {
//		Ws[k] = randoms + threadId*time_stamps->length_sortedT + N*time_stamps->length_sortedT*(k+NUM_OF_CURRENCY);
//	}
//	double *W[2*NUM_OF_CURRENCY-1];
//	W[0] = Wd;
//	for (k=0; k<NUM_OF_CURRENCY-1; k++) {
//		W[k+1] = Wf[k];
//	}
//	for (k=0; k<NUM_OF_CURRENCY-1; k++) {
//		W[k+NUM_OF_CURRENCY] = Ws[k];
//	}
//
//	//Variables for calculation
//	double t;
//	double h;
//	double T;
//	double temp1;
//	double temp2;
//	double temp3;
//	int int_temp;
//	double shuffle_temp;
//	double v1;
//	double v2;
//	double v3;
//	double ts,te;
//
//	/*
//	calc process x
//	*/
//	for (k=0; k<NUM_OF_CURRENCY; k++) {
//		//old 初期値、　newにどんどん足される
//		temp1 = r->x[k][threadId][0];
//		t=0.0;
//		//n番目分割点
//		for (n=0; n<time_stamps->length_sortedT-1; n++) {
//			//step size
//			h=time_stamps->sortedT[n+1]-time_stamps->sortedT[n];
//			//1こう目
//			temp2 = temp1;
//
//			//３こう目
//			r->vf_x[k*NUM_OF_CURRENCY + 1](temp1, &temp3, NULL, t);
//			temp2 += temp3*sqrt(h) *W[k][n];
//			//2こう目
//			r->vf_x[k*NUM_OF_CURRENCY + 0](temp1, &temp3, NULL, t);
//			temp2 += temp3*h;
//
//			t+=h;
//
//			//値の入れ替え
//			shuffle_temp = temp2;
//			temp2 = temp1;
//			temp1 = shuffle_temp;
//			r->x[k][threadId][n+1] = temp1;
//		}
//	}
//
//	/*
//	calc discount factor
//	*/
//	for (k=0; k<NUM_OF_CURRENCY; k++) {
//		temp1=0.0;
//		//Expected loss を計算する日
//		for (m=0; m<time_stamps->length_sortedT-1; m++) {
//			t=0.0;
//			T=time_stamps->sortedT[m+1];
//
//			//integrate the path of x
//			//台形近似
//			temp1 += (r->x[k][threadId][m]+ r->x[k][threadId][m+1])*(time_stamps->sortedT[m+1]-time_stamps->sortedT[m])/2.0;
//			//exp(-integrate(x))
//			temp2=exp(-temp1);
//			
//			//calculate exp(-integrate(phi))
//			ts=t;
//			te=T;
//			v1 = r->sigma[k](T)*r->sigma[k](T)/(2.0*r->a[k](t)*r->a[k](t)*r->a[k](t))*
//			(exp(-2.0*r->a[k](t)*T)*(exp(r->a[k](t)*te)-exp(r->a[k](t)*ts))*(exp(r->a[k](t)*te)+exp(r->a[k](t)*ts)-4.0*exp(r->a[k](t)*T)) + 2.0*r->a[k](t)*(te- ts));	
//
//			ts=t;
//			te=t;
//			v2 = r->sigma[k](t)*r->sigma[k](t)/(2.0*r->a[k](t)*r->a[k](t)*r->a[k](t))*
//			(exp(-2.0*r->a[k](t)*T)*(exp(r->a[k](t)*te)-exp(r->a[k](t)*ts))*(exp(r->a[k](t)*te)+exp(r->a[k](t)*ts)-4.0*exp(r->a[k](t)*T)) + 2.0*r->a[k](t)*(te- ts));	
//		
//			temp3=(zcb->P[k][threadId][m+1]/zcb->P[k][threadId][0])*exp(-(v1-v2)/2.0);
//			
//			discount_factor->D[k][threadId][m+1] = temp2*temp3;
//			outputs[threadId*time_stamps->length_sortedT + k*time_stamps->length_sortedT*N + m] = zcb->P[k][threadId][m+1];
//		}
//	}
//
//	/*
//	calc zero coupon bond
//	*/
//	for (k=0; k<NUM_OF_CURRENCY; k++) {
//		for (j=0; j<NUM_OF_SWAPS; j++) {
//			//Ti
//			for (i=0; i<time_stamps->length_Ti[k][j]; i++) {
//				//T0
//				for (l=0; l<time_stamps->length_T0; l++) {
//					//temp1=A(t,T)
//					t=time_stamps->sortedT[time_stamps->T0[l]];
//					T=time_stamps->sortedT[time_stamps->Ti[k][j][i]];
//
//					ts=0.0;
//					te=T;
//					v1 = r->sigma[k](t)*r->sigma[k](t)/(2.0*r->a[k](t)*r->a[k](t)*r->a[k](t))*
//					(exp(-2.0*r->a[k](t)*T)*(exp(r->a[k](t)*te)-exp(r->a[k](t)*ts))*(exp(r->a[k](t)*te)+exp(r->a[k](t)*ts)-4.0*exp(r->a[k](t)*T)) + 2.0*r->a[k](t)*(te- ts));	
//					
//					ts=0.0;
//					te=t;
//					v2 = r->sigma[k](t)*r->sigma[k](t)/(2.0*r->a[k](t)*r->a[k](t)*r->a[k](t))*
//					(exp(-2.0*r->a[k](t)*T)*(exp(r->a[k](t)*te)-exp(r->a[k](t)*ts))*(exp(r->a[k](t)*te)+exp(r->a[k](t)*ts)-4.0*exp(r->a[k](t)*T)) + 2.0*r->a[k](t)*(te- ts));	
//				
//					ts=t;
//					te=T;
//					v3 =r->sigma[k](t)*r->sigma[k](t)/(2.0*r->a[k](t)*r->a[k](t)*r->a[k](t))*
//					(exp(-2.0*r->a[k](t)*T)*(exp(r->a[k](t)*te)-exp(r->a[k](t)*ts))*(exp(r->a[k](t)*te)+exp(r->a[k](t)*ts)-4.0*exp(r->a[k](t)*T)) + 2.0*r->a[k](t)*(te- ts));	
//					temp1 = log(zcb->P[k][threadId][time_stamps->Ti[k][j][i]]/zcb->P[k][threadId][time_stamps->T0[l]])+(v3-v1+v2)/2.0;
//
//					//temp2=B(t,T)
//					temp2 = (1.0- exp(-r->a[k](t)*(T-t)))/r->a[k](t);
//
//					//P
//					zcb->P[k][threadId][time_stamps->T0[l]*time_stamps->length_sortedT+time_stamps->Ti[k][j][i]]=exp(temp1-temp2*r->x[k][threadId][time_stamps->T0[l]]);
//				}
//			}
//		}
//	}
//
//
//	/*
//	calc swaps
//	*/
//	for (k=0; k<NUM_OF_CURRENCY; k++) {
//		for (j=0; j<NUM_OF_SWAPS; j++) {
//			//T0
//			for (l=0; l<time_stamps->length_T0; l++) {
//				swap->swap[k][j][threadId][time_stamps->T0[l]]=0.0;
//				//Ti
//				for (i=0; i<time_stamps->length_Ti[k][j]-1; i++) {
//					//temp1=FxC(T0,Ti,Ti+1). Fixed Coupon
//					temp1= swap->K[k][j]*zcb->P[k][threadId][time_stamps->T0[l]*time_stamps->length_sortedT + time_stamps->Ti[k][j][i+1]]*(time_stamps->sortedT[time_stamps->Ti[k][j][i+1]] - time_stamps->sortedT[time_stamps->Ti[k][j][i]]);
//
//					//temp2=FlC(T0,Ti,Ti+1). Floating Coupon
//					temp2= 
//					(zcb->P[k][threadId][time_stamps->T0[l]*time_stamps->length_sortedT + time_stamps->Ti[k][j][i]]
//					/zcb->P[k][threadId][time_stamps->T0[l]*time_stamps->length_sortedT + time_stamps->Ti[k][j][i+1]] - 1.0);
//					
//					//temp3=Swap(T0,...). Swap
//					swap->swap[k][j][threadId][time_stamps->T0[l]]+=temp1-temp2;
//				}
//				swap->swap[k][j][threadId][time_stamps->T0[l]]*=swap->N[k][j];
//			}
//		}
//	}
//
//
//	/*
//	calc FX
//	*/
//	for (k=0; k<NUM_OF_CURRENCY-1; k++) {
//		//T0
//		for (l=0; l<time_stamps->length_T0; l++) {
//			//temp3=exp(Ito(sigma))
//			temp3=0.0;
//			t=0.0;
//			//Euler Maruyama
//			for (n=0; n<time_stamps->T0[l]-1; n++) {
//				h = time_stamps->sortedT[n+1]-time_stamps->sortedT[n];
//				temp3+=r->sigma[k+1](t)*sqrt(h)*Ws[k][n];
//				t+=h;
//			}
//			spot_fx->spot_fx[k][threadId][time_stamps->T0[l]]=
//			spot_fx->spot_fx[k][threadId][0]*
//			discount_factor->D[k+1][threadId][time_stamps->T0[l]]/
//			discount_factor->D[0][threadId][time_stamps->T0[l]]*
//			exp(temp3);
//		}
//	}
//
//	/*
//	calc NSet
//	*/
//	for (l=0; l<time_stamps->length_T0; l++) {
//		nset->nset[threadId][time_stamps->T0[l]]=0.0;
//		//domestic
//		for (j=0; j<NUM_OF_SWAPS; j++) {
//			nset->nset[threadId][time_stamps->T0[l]] += swap->swap[0][j][threadId][time_stamps->T0[l]];
//		}
//		//foreign
//		for (k=1; k<NUM_OF_CURRENCY; k++) {
//			for (j=0; j<NUM_OF_SWAPS; j++) {
//				nset->nset[threadId][time_stamps->T0[l]] += swap->swap[k][j][threadId][time_stamps->T0[l]]*spot_fx->spot_fx[k-1][threadId][time_stamps->T0[l]];
//			}
//		}
//	}
//
//
//	/*
//	differentiates swaps
//	*/
//	//first term
//	//differentiate FxC by P(dFxC/dP)
//	for (j=0; j<NUM_OF_SWAPS; j++) {
//		for (mm=0; mm<time_stamps->length_T0; mm++) {
//			for(m=0; m<NUM_OF_MARKET_DATA; m++){
//				swap->p_swap[g_c][j][threadId][mm*(NUM_OF_MARKET_DATA) + m] =  0.0;
//				for (l=0; l<time_stamps->length_Ti[g_c][j]; l++) {
//					swap->p_swap[g_c][j][threadId][time_stamps->T0[mm]*(NUM_OF_MARKET_DATA) + m] += 
//					(time_stamps->sortedT[time_stamps->Ti[g_c][j][l+1]] - time_stamps->sortedT[time_stamps->Ti[g_c][j][l]])*
//					(-zcb->P[g_c][threadId][time_stamps->T0[mm]*time_stamps->length_sortedT + time_stamps->Ti[g_c][j][l]]/
//					zcb->P[g_c][threadId][0 + time_stamps->T0[mm]]*
//					zcb->p_P[g_c][time_stamps->T0[mm]*NUM_OF_MARKET_DATA + m]
//					+
//					zcb->P[g_c][threadId][time_stamps->T0[mm]*time_stamps->length_sortedT + time_stamps->Ti[g_c][j][l]]/
//					zcb->P[g_c][threadId][0 + time_stamps->Ti[g_c][j][l]]*
//					zcb->p_P[g_c][time_stamps->Ti[g_c][j][l]*NUM_OF_MARKET_DATA + m]);
//				}
//				swap->p_swap[g_c][j][threadId][mm*(NUM_OF_MARKET_DATA) + m] *=  swap->N[g_c][j]*swap->K[g_c][j];
//			}
//		}
//	}
//	
//	
//	//second term
//	//differentiate FwdR by P
//	for (j=0; j<NUM_OF_SWAPS; j++) {
//		//expected loss
//		for (mm=0; mm<time_stamps->length_T0; mm++) {
//			for(m=0; m<NUM_OF_MARKET_DATA; m++){
//				sum = 0.0;
//				for (l=0; l<time_stamps->length_Ti[g_c][j]; l++) {
//					sum +=
//					1.0/
//					zcb->P[g_c][threadId][time_stamps->T0[mm]*time_stamps->length_sortedT + time_stamps->Ti[g_c][j][l+1]]*
//					(
//						-zcb->P[g_c][threadId][time_stamps->T0[mm]*time_stamps->length_sortedT + time_stamps->Ti[g_c][j][0]]/
//						zcb->P[g_c][threadId][0 + time_stamps->T0[mm]]*
//						zcb->p_P[g_c][time_stamps->T0[mm]*NUM_OF_MARKET_DATA + m]
//						+
//						zcb->P[g_c][threadId][time_stamps->T0[mm]*time_stamps->length_sortedT + time_stamps->Ti[g_c][j][l]]/
//						zcb->P[g_c][threadId][0 + time_stamps->Ti[g_c][j][l]]*
//						zcb->p_P[g_c][time_stamps->Ti[g_c][j][l]*NUM_OF_MARKET_DATA + m]
//					)
//					+
//					zcb->P[g_c][threadId][time_stamps->T0[mm]*time_stamps->length_sortedT + time_stamps->Ti[g_c][j][0]]/
//					(
//						zcb->P[g_c][threadId][0 + time_stamps->T0[mm]]*
//						zcb->P[g_c][threadId][0 + time_stamps->T0[mm]]
//					)
//					*
//					(
//						-zcb->P[g_c][threadId][time_stamps->T0[mm]*time_stamps->length_sortedT + time_stamps->Ti[g_c][j][l+1]]/
//						zcb->P[g_c][threadId][0 + time_stamps->T0[mm]]*
//						zcb->p_P[g_c][time_stamps->T0[mm]*NUM_OF_MARKET_DATA + m]
//						+
//						zcb->P[g_c][threadId][time_stamps->T0[mm] + time_stamps->Ti[g_c][j][l+1]]/
//						zcb->P[g_c][threadId][0 + time_stamps->Ti[g_c][j][l]]*
//						zcb->p_P[g_c][time_stamps->Ti[g_c][j][l+1]*NUM_OF_MARKET_DATA + m]
//					);
//				}
//				sum +=
//				-1.0/
//				(
//					zcb->P[g_c][threadId][time_stamps->T0[mm]*time_stamps->length_sortedT + time_stamps->Ti[g_c][j][0]]*
//					zcb->P[g_c][threadId][time_stamps->T0[mm]*time_stamps->length_sortedT + time_stamps->Ti[g_c][j][0]]
//				)
//				*
//				(
//					-zcb->P[g_c][threadId][time_stamps->T0[mm]*time_stamps->length_sortedT + time_stamps->Ti[g_c][j][0]]/
//					zcb->P[g_c][threadId][0 + time_stamps->T0[mm]]*
//					zcb->p_P[g_c][time_stamps->T0[mm]*NUM_OF_MARKET_DATA + m]
//					+
//					zcb->P[g_c][threadId][time_stamps->T0[mm]*time_stamps->length_sortedT + time_stamps->Ti[g_c][j][0]]/
//					zcb->P[g_c][threadId][0 + time_stamps->Ti[g_c][j][0]]*
//					zcb->p_P[g_c][time_stamps->Ti[g_c][j][0]*NUM_OF_MARKET_DATA + m]
//				);
//				sum *= swap->N[g_c][j];
//				swap->p_swap[g_c][j][threadId][time_stamps->T0[mm]*(NUM_OF_MARKET_DATA) + m] += sum;
//			}
//		}
//	}
//	/*
//	differentiate Fxs
//	*/
//	if(g_c == 0){
//		for (mm=0; mm<time_stamps->length_T0; mm++) {
//			for(m=0; m<NUM_OF_MARKET_DATA; m++){
//				spot_fx->p_spot_fx[0][threadId][time_stamps->T0[mm]*NUM_OF_MARKET_DATA + m] = 
//					-spot_fx->spot_fx[0][threadId][time_stamps->T0[mm]]/
//					zcb->P[0][threadId][time_stamps->T0[mm]]*
//					zcb->p_P[0][time_stamps->T0[mm]*NUM_OF_MARKET_DATA + m];
//			}
//		}
//	}else{
//		for (mm=0; mm<time_stamps->length_T0; mm++) {
//			for(m=0; m<NUM_OF_MARKET_DATA; m++){
//				spot_fx->p_spot_fx[g_c][threadId][time_stamps->T0[mm]*NUM_OF_MARKET_DATA + m] = 
//					spot_fx->spot_fx[g_c][threadId][time_stamps->T0[mm]]/
//					zcb->P[g_c][threadId][time_stamps->T0[mm]]*
//					zcb->p_P[g_c][time_stamps->T0[mm]*NUM_OF_MARKET_DATA + m];
//			}
//		}
//	}
//	/*
//	differentiate discount factor
//	*/
//	if(g_c == 0){
//		for (mm=0; mm<time_stamps->length_T0; mm++) {
//			for(m=0; m<NUM_OF_MARKET_DATA; m++){
//				discount_factor->p_D[0][threadId][time_stamps->T0[mm]*NUM_OF_MARKET_DATA + m] =
//				discount_factor->D[0][threadId][time_stamps->T0[mm]]/
//				zcb->P[0][threadId][time_stamps->T0[mm]]*
//				zcb->p_P[0][time_stamps->T0[mm]*NUM_OF_MARKET_DATA + m];
//			}
//		}
//	}
//
//
//	/*
//	differentiate Net Value
//	*/
//	if(g_c == 0){
//		for (mm=0; mm<time_stamps->length_T0; mm++) {
//			for(m=0; m<NUM_OF_MARKET_DATA; m++){
//				nset->p_nset[threadId][time_stamps->T0[mm] + m] = 0;
//
//				//differentiate domestic swaps
//				for (j=0; j<NUM_OF_SWAPS; j++) {
//					nset->p_nset[threadId][time_stamps->T0[mm] + m] +=
//						swap->p_swap[0][j][threadId][time_stamps->T0[mm]*NUM_OF_MARKET_DATA + m];
//				}
//
//				//differentiate foreighn swaps
//				for (k=1; k<NUM_OF_CURRENCY; k++) {
//					for (j=0; j<NUM_OF_SWAPS; j++) {
//						nset->p_nset[threadId][time_stamps->T0[mm] + m] +=
//							swap->swap[k][j][threadId][time_stamps->T0[mm]]*
//							spot_fx->p_spot_fx[k-1][threadId][time_stamps->T0[mm]*NUM_OF_MARKET_DATA + m];
//					}
//				}
//			}
//		}
//	}else{
//		for (mm=0; mm<time_stamps->length_T0; mm++) {
//			for(m=0; m<NUM_OF_MARKET_DATA; m++){
//				nset->p_nset[threadId][time_stamps->T0[mm] + m] = 0.0;
//
//				//differentiate swaps in g_c
//				for (j=0; j<NUM_OF_SWAPS; j++) {
//					nset->p_nset[threadId][time_stamps->T0[mm] + m] +=
//						swap->p_swap[g_c][j][threadId][time_stamps->T0[mm]*NUM_OF_MARKET_DATA + m]*
//						spot_fx->spot_fx[g_c-1][threadId][time_stamps->T0[mm]]
//						+
//						swap->swap[g_c][j][threadId][time_stamps->T0[mm]]*
//						spot_fx->p_spot_fx[g_c-1][threadId][time_stamps->T0[mm]*NUM_OF_MARKET_DATA + m];
//				}
//			}
//		}
//	}
//
//						
//	/*
//	Calc CVA_greeksG
//	*/
//	if(g_c == 0) {
//		for (mm=0; mm<time_stamps->length_T0; mm++) {
//			for (m=0; m<NUM_OF_MARKET_DATA; m++) {
//				CVA_Delta[threadId*NUM_OF_MARKET_DATA + m] = 0.0;
//
//				//PS
//				//integrate(-lambda))
//				for (n=int_temp; n<time_stamps->T0[mm]; n++) {
//					//台形近似
//					temp1 += (default_probability->lambda(time_stamps->sortedT[n+1])+default_probability->lambda(time_stamps->sortedT[n]))*(time_stamps->sortedT[n+1]-time_stamps->sortedT[n])/2.0;
//				}
//				int_temp=time_stamps->T0[mm];
//				temp3=temp2;
//				//SP(Ti)
//				temp2=1.0-exp(-temp1);
//				
//				//expected loss*p_D
//				CVA_Delta[threadId*NUM_OF_MARKET_DATA + m] += 
//					discount_factor->p_D[0][threadId][time_stamps->T0[mm]*NUM_OF_MARKET_DATA + m]*
//					nset->nset[threadId][time_stamps->T0[mm]]*(temp2-temp3);
//				
//				//(derivative of expected loss)*D
//				if (nset->nset[threadId][time_stamps->T0[mm]]>0.0) {
//					CVA_Delta[threadId*NUM_OF_MARKET_DATA + m] += 
//						discount_factor->D[0][threadId][time_stamps->T0[mm]]*
//						nset->p_nset[threadId][time_stamps->T0[mm]*NUM_OF_MARKET_DATA + m]*(temp2-temp3);
//				}
//				CVA_Delta[threadId*NUM_OF_MARKET_DATA + m] *= (1.0 - RECOVERY_RATE);
//			}
//		}
//	}
//}

