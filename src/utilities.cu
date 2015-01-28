#include <utilities.h>
#include <stdio.h>


void time_stamps_sort(TIME_STAMPS *ts, double *T0, double ***Ti, int noc, int nos) {

	int i, j, k, l;
	double min;
	int *min_indice;
	int length_min_indice;
	int index_sortedT;
	int currency_index;
	int swap_index;
	int total_length=0;


	//malloc
	double **Ts;
	Ts = (double **)malloc(sizeof(double *)*(noc*nos+1));
	Ts[0] = T0;
	for (k=0; k<noc; k++) {
		for (j=0; j<nos; j++) {
			Ts[k*nos + j + 1] = Ti[k][j];
		}
	}
	int *length_Ts;
	length_Ts = (int *)malloc(sizeof(int)*(noc*nos+1));
	length_Ts[0] = ts->length_T0;
	total_length += ts->length_T0;
	for (k=0; k<noc; k++) {
		for (j=0; j<nos; j++) {
			length_Ts[k*nos + j + 1] = ts->length_Ti[k][j];
			total_length += ts->length_Ti[k][j];
		}
	}
	int *head_index_Ts;
	head_index_Ts = (int *)malloc(sizeof(int)*(noc*nos+1));
	for (k=0; k<noc*nos+1; k++) {
		head_index_Ts[k] = 0;
	}
	int *index_Ts;
	index_Ts = (int *)malloc(sizeof(int)*(noc*nos+1));
	for (k=0; k<noc*nos+1; k++) {
		index_Ts[k] = 0;
	}

	double *temp_sortedT;
	temp_sortedT = (double *)malloc(sizeof(double)*(total_length+1));
	temp_sortedT[0]=0.0;

	//sort
	min_indice = (int *)malloc(sizeof(int)*(noc*nos+1));
	index_sortedT=1;
	for(k=0; k<noc*nos+1; k++) {
		for (j=head_index_Ts[k]; j<length_Ts[k]; j++) {
			min = Ts[k][j];
			min_indice[0] = k;
			length_min_indice = 1;
			for (i=k+1; i<noc*nos+1; i++) {
				//update minimum value
				if (head_index_Ts[i] < length_Ts[i] && min > Ts[i][head_index_Ts[i]]) {
					min = Ts[i][head_index_Ts[i]];
					for (l=1; l<length_min_indice; l++) {
						head_index_Ts[min_indice[l]]-=1;
					}
					//update index
					head_index_Ts[i]+=1;
					min_indice[0]=i;
					length_min_indice=1;
					j--;
				} 
				if (head_index_Ts[i] < length_Ts[i] && min == Ts[i][head_index_Ts[i]]) {
					head_index_Ts[i]+=1;
					//Add an index at min value
					length_min_indice+=1;
					min_indice[length_min_indice-1]=i;
				}
			}
			temp_sortedT[index_sortedT] = min;
			for (l=0; l<length_min_indice; l++) {
				if (min_indice[l]== 0) {
					ts->T0[index_Ts[min_indice[l]]] = index_sortedT;
					index_Ts[min_indice[l]] += 1;
				} else {
					currency_index = (int)((min_indice[l]-1)/nos);
					swap_index = (min_indice[l]-1) % nos;
					ts->Ti[currency_index][swap_index][index_Ts[min_indice[l]]] = index_sortedT;
					index_Ts[min_indice[l]] += 1;
				}
			}

			index_sortedT++;
		}
	}

	ts->sortedT = (double *)malloc(sizeof(double)*index_sortedT);
	for (k=0; k<index_sortedT; k++) {
		ts->sortedT[k] = temp_sortedT[k];
	}
	ts->length_sortedT = index_sortedT;
	
	free(Ts);
	free(length_Ts);
	free(head_index_Ts);
	free(index_Ts);
	free(min_indice);
	free(temp_sortedT);

}

