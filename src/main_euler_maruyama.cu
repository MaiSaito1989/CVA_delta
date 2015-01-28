#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <MT.h>
#include <helper_cuda.h>
#include <cva_common.h>

__device__ int BlackScholes_V0(const double y[], double dy[], void *parameters, double t) {
	BS_PARAMS *params = reinterpret_cast<struct BS_PARAMS *>(parameters);
	//yとparamsを使って、dyを計算するコード
	dy[0] = params->r * y[0];

	return 1;
}

__device__ int BlackScholes_V1(const double y[], double dy[], void *parameters, double t) {
	BS_PARAMS *params = reinterpret_cast<struct BS_PARAMS *>(parameters);
	//yとparamsを使って、dyを計算するコード
	dy[0] = params->sigma*y[0];
	return 1;
}
__device__ int BlackScholesExtended_V0(const double y[], double dy[], void *parameters, double t) {
	BSE_PARAMS *params = reinterpret_cast<struct BSE_PARAMS *>(parameters);
	//yとparamsを使って、dyを計算するコード
	dy[0] = params->r(t,parameters) * y[0];

	return 1;
}
__device__ int BlackScholesExtended_V1(const double y[], double dy[], void *parameters, double t) {
	BS_PARAMS *params = reinterpret_cast<struct BS_PARAMS *>(parameters);
	//yとparamsを使って、dyを計算するコード
	dy[0] = params->sigma*y[0];
	return 1;
}
__device__ double payoff(double y[], void *parameters){
	HW_PARAMS *params =  reinterpret_cast<struct HW_PARAMS *>(parameters);
	if((y[0]-params->K)>=0){
		return y[0]-params->K;
	} else {
		return 0.0;
	}
}

int main(void) {
	int N = 128*1024;
	int M = 32;//64
	int j = 0;

	//Prameters
	BS_PARAMS *bs_params = (BS_PARAMS *)malloc(sizeof(BS_PARAMS));
	bs_params->sigma = 0.2;
	bs_params->r = 0.1;
	bs_params->K = 60.0;

	HW_PARAMS *hw_params = (HW_PARAMS *)malloc(sizeof(HW_PARAMS));
	hw_params->t = 0.0;
	hw_params->K = 60.0;

	//SDE
	SDE *sde =(SDE *)malloc(sizeof(SDE));
	sde->dim_y = 1;
	sde->dim_BM = 1;
	sde->init_y = (double *)malloc(sizeof(double)*sde->dim_y);
	sde->init_y[0] = 62.0;
	sde->T = 5.0/12.0;
	sde->parameters = hw_params;
	sde->V = (VECTOR_FIELD*)malloc(sizeof(VECTOR_FIELD)*(2));

	SDE *d_sde;
	int(**d_V)(const double y[], double dy[], void *parameters);
	double *d_init_y;
	//BS_PARAMS *d_bs_params;
	HW_PARAMS *d_hw_params;

	// Allocate storage for struct and name
	checkCudaErrors(cudaMalloc((void**)&d_sde,sizeof(SDE)));
	checkCudaErrors(cudaMalloc((void**)&d_init_y,sizeof(double)*sde->dim_y));
	checkCudaErrors(cudaMalloc((void**)&d_hw_params,sizeof(HW_PARAMS)));
	checkCudaErrors(cudaMalloc((void**)&d_V,sizeof(int(*)(const double y[],double dy[], void *params))*(2)));

	checkCudaErrors(cudaMemcpyFromSymbol( &(sde->V[0]), vector_field0, sizeof(VECTOR_FIELD)));
	checkCudaErrors(cudaMemcpyFromSymbol( &(sde->V[1]), vector_field1, sizeof(VECTOR_FIELD)));

	checkCudaErrors(cudaMemcpyFromSymbol( &(hw_params->sigma), hullwhitevasicek_sigma, sizeof(FUNC_OF_TIME)));
	checkCudaErrors(cudaMemcpyFromSymbol( &(hw_params->b), hullwhitevasicek_b, sizeof(FUNC_OF_TIME)));
	checkCudaErrors(cudaMemcpyFromSymbol( &(hw_params->beta), hullwhitevasicek_beta, sizeof(FUNC_OF_TIME)));

	checkCudaErrors(cudaMemcpyFromSymbol( &(sde->payoff), payoff, sizeof(PAYOFF_FUNC)));

	// Copy up each piece separately, including new “name” pointer value
	checkCudaErrors(cudaMemcpy(d_sde, sde, sizeof(SDE), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_init_y, sde->init_y, sizeof(double)*sde->dim_y, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(&(d_sde->init_y), &d_init_y, sizeof(double*), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_hw_params, hw_params, sizeof(HW_PARAMS), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(&(d_sde->parameters), &d_hw_params, sizeof(HW_PARAMS*), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_V, sde->V, sizeof(VECTOR_FIELD)*2, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(&(d_sde->V), &d_V, sizeof(VECTOR_FIELD), cudaMemcpyHostToDevice));
	unsigned long init[4]={0x123, 0x234, 0x345, 0x456}, length=4;
	init_by_array(init, length);

	double *results;
	results = (double*)malloc(sizeof(double)*N);
	double *d_results;
	checkCudaErrors(cudaMalloc((void**)&d_results,sizeof(double)*N));

	double *h_randams;
	double *d_randoms;
	h_randams = (double *)malloc(sizeof(double)*M*N*sde->dim_BM);
	checkCudaErrors(cudaMalloc((void**)&d_randoms, sizeof(double)*M*N*sde->dim_BM));

	generate_brawnian_paths(h_randams,  2, M,  N);
	rand_normal_ndim(h_randams, M*N*sde->dim_BM);
	checkCudaErrors(cudaMemcpy(d_randoms, h_randams, sizeof(double)*M*N*sde->dim_BM, cudaMemcpyHostToDevice));




	double *d_Xold;
	double *d_Xnew;
	double *d_temp;

	checkCudaErrors(cudaMalloc((void**)&d_Xold, sizeof(double)*sde->dim_y*N));
	checkCudaErrors(cudaMalloc((void**)&d_Xnew, sizeof(double)*sde->dim_y*N));
	checkCudaErrors(cudaMalloc((void**)&d_temp, sizeof(double)*sde->dim_y*N));


	int bd = 1024;
	int gd = N/1024;

	printf("a\n");
	//Euler_Maruyama<<<gd, bd>>>(d_sde,d_randoms, d_results,d_Xold, d_Xnew, d_temp,1,N,M);
	printf("i\n");
	//cudaDeviceSynchronize();
	//checkCudaErrors(cudaMemcpy(results, d_results, sizeof(double)*N, cudaMemcpyDeviceToHost)) ;
	double result=0.0;

	Process_x(d_randoms, d_results, d_Xolds, d_Xnews,d_temps, model, N, M, 1);
	checkCudaErrors(cudaMemcpy(results, d_results, sizeof(double)*N, cudaMemcpyDeviceToHost)) ;



	//Process_r_kernel(*results,country);
	//Process_r(*d_randoms, double *d_results, double *d_Xolds, double *d_Xnews,double *d_temps, int model, int N, int M,int country){



	for (j=0; j<N; j++) {
		result += results[j];
	}

	printf("%lf nnn", result);

	free(sde->V);
	free(sde->init_y);
	free(sde);
	free(hw_params);
	free(h_randams);

	cudaFree(d_results);
	cudaFree(d_randoms);

	cudaFree(d_init_y);
	cudaFree(d_V);
	cudaFree(d_hw_params);

	cudaFree(d_sde);
	cudaFree(d_Xold);
	cudaFree(d_Xnew);
	cudaFree(d_temp);

}
