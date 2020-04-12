/*
 * Tema 2 ASC
 * 2020 Spring
 */
#include <stdlib.h>
#include "cblas.h"
#include "utils.h"
#include "matrix_helpers.h"

/* 
 * Add your BLAS implementation here
 */
double* my_solver(int N, double *A, double *B) {
	/* Computing the main parameters used for computation */

	/* First element of the computation using BLAS */

	/**
	 * The blass function dgemm_ is used to process the result of
	 * the following equation:
	 * C := alpha*op( A )*op( B ) + beta*C,
	 * 
	 **/ 
	double* B_At = (double *) malloc(N * N * sizeof(double));
	if (B_At == NULL)
		exit(BAD_ALLOC);

	double* A2 = (double *) malloc(N * N * sizeof(double));
	if (A2 == NULL)
		exit(BAD_ALLOC);

	double* A2_B = (double *) malloc(N * N * sizeof(double));
	if (A2_B == NULL)
		exit(BAD_ALLOC);

	double* R = (double *) malloc(N * N * sizeof(double));
	if (R == NULL)
		exit(BAD_ALLOC);

	char is_A_transpose = CblasTrans;
	char is_B_transpose = CblasNoTrans;

	double alpha = 1.0;
	double beta = 0.0;

	/*  B * At */
	clab_dgemm(
		&is_B_transpose, // weather first matrix is transpose or not 
		&is_A_transpose, // weather second matrix is transpose or not
		&N, // number of rows in A
		&N, // number of columns in B
		&N, // the common number of elements between the two matrxies eq: 2_3 * 3_4 => 3 
		&alpha, // alpha paramater from the equation
		B, // the first matrix
		&N, // the first dimension of A
		A, // the second matrix
		&N, // the first dimension of B
		&beta, // beta parameter from the equation
		B_At, // the result matrix
		&N // the number of columns in R
	);
	
	is_A_transpose = CblasNoTrans;
	is_B_transpose = CblasNoTrans;

	/* A2 */
	cblas_dgemm(
		&is_A_transpose, // weather first matrix is transpose or not 
		&is_B_transpose, // weather second matrix is transpose or not
		&N, // number of rows in A
		&N, // number of columns in B
		&N, // the common number of elements between the two matrxies eq: 2_3 * 3_4 => 3 
		&alpha, // alpha paramater from the equation
		A, // the first matrix
		&N, // the first dimension of A
		A, // the second matrix
		&N, // the first dimension of B
		&beta, // beta parameter from the equation
		A2, // the result matrix
		&N // the number of columns in R
	);

	is_A_transpose = CblasNoTrans;
	is_B_transpose = CblasNoTrans;

	/* A2 * B */
	cblas_dgemm(
		&is_A_transpose, // weather first matrix is transpose or not 
		&is_B_transpose, // weather second matrix is transpose or not
		&N, // number of rows in A
		&N, // number of columns in B
		&N, // the common number of elements between the two matrxies eq: 2_3 * 3_4 => 3 
		&alpha, // alpha paramater from the equation
		A2, // the first matrix
		&N, // the first dimension of A
		B, // the second matrix
		&N, // the first dimension of B
		&beta, // beta parameter from the equation
		A2_B, // the result matrix
		&N // the number of columns in R
	);
	
	R = matrix_add(N, B_At, A2_B);

	free(B_At);
	free(A2_B);
	free(A2);

	return R;
}
