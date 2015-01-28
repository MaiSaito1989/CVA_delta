#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mt19937.h>
#include <helper_cuda.h>
#include <cva_common.h>
#include <Euler_Maruyama.h>


//ModelのV0の関数
//yは引数、dyは関数の計算結果
//V0(y)=dy
__device__ int HullWhiteVasicek_V0(const double y[], double dy[], void *parameters, double t) {
	HW_PARAMS *params = reinterpret_cast<struct HW_PARAMS *>(parameters);
	//yとparamsを使って、dyを計算するコード
	dy[0] =  params->b(t) + params->beta(t)* y[0];
	return 1;
}

//ModelのV1の関数
//yは引数、dyは関数の計算結果
//V1(y)=dy
__device__ int HullWhiteVasicek_V1(const double y[], double dy[], void *parameters, double t) {
	HW_PARAMS *params = reinterpret_cast<struct HW_PARAMS *>(parameters);
	//yとparamsを使って、dyを計算するコード
	dy[0] = params->sigma(t);
	return 1;
}

__device__ double HullWhiteVasicek_sigma(const double t)  {

	return t;
}
__device__ double HullWhiteVasicek_b(const double t)  {

	return t;
}
__device__ double HullWhiteVasicek_beta(const double t)  {

	return t;
}


__device__ double market_instantaneous_FR(double t, int country){
	return 0.0;
}

/////////////////////////////////////////


__device__ void Euler_Maruyama(SDE *sde, double *randems, double *results, double *Xold, double *Xnew,double *temp, int N, int M) {
	//////////次元をあげたいから、dim_yを使うべき？引数のとこに？？
	int i;
	int n;
	int k;
	int m;

	double *z;
	int j = threadIdx.x+threadIdx.y*blockDim.x + blockDim.x*blockDim.y*blockIdx.x ;
	m = j *sde->dim_y;

	double *shuffle_temp;
	double h = sde->T/M;

	results[j]=0.0;

	int L=M*sde->dim_BM;
	z = &randems[j*L] ;

	//old 初期値、　newにどんどん足される
	for (k=0; k<sde->dim_y; k++) {
		Xold[m+k] = sde->init_y[k];
	}
	//n番目分割点
	for (n=0; n<M; n++) {
		//1こう目
		for (k=0; k<sde->dim_y; k++) {
			Xnew[m+k] = Xold[m+k];
		}
		//３こう目
		for(i=1; i<=sde->dim_BM; i++){
			sde->V[i](&Xold[m], &temp[m], sde->parameters, n*h);
			for (k=0; k<sde->dim_y; k++) {
				Xnew[m+k] +=  temp[m+k]*sqrt(h) * z[n*sde->dim_BM + i-1];
			}
		}
		//2こう目
		sde->V[0](&Xold[m], &temp[m], sde->parameters, n*h);
		for (k=0; k<sde->dim_y; k++) {
			Xnew[m+k] += temp[m+k]*h;
		}
		//ポインタの入れ替え
		shuffle_temp = Xnew;
		Xnew = Xold;
		Xold = shuffle_temp;
	}
	results[j] += sde->payoff(&Xold[m], sde->parameters)/N;


}

//M:分割数
__device__ void Euler_Maruyama_BSE(SDE *sde, double *randems, double *results, double *Xold, double *Xnew,double *temp, int N, int M) {
	//////////次元をあげたいから、dim_yを使うべき？引数のとこに？？
	int i;
	int n;
	int k;
	int m;

	double *z;
	int j = threadIdx.x+threadIdx.y*blockDim.x + blockDim.x*blockDim.y*blockIdx.x ;
	m = j *sde->dim_y;
	BSE_PARAMS*params = reinterpret_cast<struct BSE_PARAMS *>(sde->parameters);
	params->j = j;
	double *shuffle_temp;
	double h = sde->T/M;

	results[j]=0.0;

	int L=M*sde->dim_BM;
	z = &randems[j*L] ;

	//old 初期値、　newにどんどん足される
	for (k=0; k<sde->dim_y; k++) {
		Xold[m+k] = sde->init_y[k];
	}
	//n番目分割点
	for (n=0; n<M; n++) {
		//1こう目
		for (k=0; k<sde->dim_y; k++) {
			Xnew[m+k] = Xold[m+k];
		}
		//３こう目
		for(i=1; i<=sde->dim_BM; i++){
			sde->V[i](&Xold[m], &temp[m], sde->parameters, n*h);
			for (k=0; k<sde->dim_y; k++) {
				Xnew[m+k] +=  temp[m+k]*sqrt(h) * z[n*sde->dim_BM + i-1];
			}
		}
		//2こう目
		sde->V[0](&Xold[m], &temp[m], sde->parameters, n*h);
		for (k=0; k<sde->dim_y; k++) {
			Xnew[m+k] += temp[m+k]*h;
		}
		//ポインタの入れ替え
		shuffle_temp = Xnew;
		Xnew = Xold;
		Xold = shuffle_temp;
	}
	results[j] += sde->payoff(&Xold[m], sde->parameters)/N;


}



__global__ void Euler_Maruyama(SDE *sde, double *randems, double *results, double *Xolds, double *Xnews,double *temps, int model, int N, int M) {
	if(model == 1) {
		Euler_Maruyama(sde, randems, results, Xolds, Xnews,temps, N, M);
	}else if (model == 2){
		Euler_Maruyama_BSE(sde, randems, results, Xolds, Xnews,temps, N, M);
	}

}

__device__ double Probability_Survival(const double t) {
	//Probability Survival
	return 0.0;
}



