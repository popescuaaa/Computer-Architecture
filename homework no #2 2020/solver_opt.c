/*
 * Tema 2 ASC
 * 2020 Spring
 */
#include "utils.h"
#include "matrix_helpers.h"

/*
 * Add your optimized implementation here
 */
double* my_solver(int N, double *A, double* B) 
{
	/* Computing the main parameters used for computation */ 
	double* At = transpose(N, A);
	double* A2 = multiply_opt(N, A, A);
	double* T1 = multiply_opt(N, B, At);
	double* T2 = multiply_opt(N, A2, B);

	double* R = matrix_add(N, T1, T2);
	
	free(At);
	free(A2);
	free(T1);
	free(T2);

	return R;	
}
