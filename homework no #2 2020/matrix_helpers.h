#ifndef _MATRIX_HELPERS_
#define _MATRIX_HELPERS_

#include <stdlib.h>

#define BAD_ALLOC 12

/**
 * @param: N: int - the number of elements per column / line
 * @param: M: double* - the matrix stored as a vector N * N 
 * @return: The transpose of a matrix
 * 
 **/ 
double* transpose(int N, double *M) 
{   
    size_t li;
    size_t ci;
    double *T = (double*) malloc(N * N * sizeof(double));
    if (T == NULL)
        exit(BAD_ALLOC);

    for (li = 0; li < N; li++) {
        for (ci = 0; ci < N; ci++) {
            T[li * N + ci] = M[ci * N + li];
        }
    }
    
    return T;
}

/**
 * @param: N: int number of elements per line/colum in each matrix
 * @param: A: double* the first matrix stored as a vector
 * @param: B: double* the second matrix stored as a vector
 * @return: The matrix A and B addition result
 * 
 **/ 
double* matrix_add(int N, double* A, double* B)
{
    size_t li;
    size_t ci;
    double* R = (double *) malloc(N * N * sizeof(double));
    if (R == NULL)
        exit(BAD_ALLOC);
    
    for (li = 0; li < N; li++) {
        for (ci = 0; ci < N; ci++) {
            R[li * N + ci] = A[li * N + ci] + B[li * N + ci];   
        }
    }
    
    return R;
}

/**
 * @param: N: int number of elements per line/colum in each matrix
 * @param: A: double* the first matrix stored as a vector
 * @param: B: double* the second matrix stored as a vector
 * @return: The matrix A and B substraction result
 * 
 **/ 
double* matrix_add(int N, double* A, double* B)
{   
    size_t li;
    size_t ci;
    double* R = (double *) malloc(N * N * sizeof(double));
    if (R == NULL)
        exit(BAD_ALLOC);
    
    for (li = 0; li < N; li++) {
        for (ci = 0; ci < N; ci++) {
            R[li * N + ci] = A[li * N + ci] - B[li * N + ci];   
        }
    }
    
    return R;
}

/**
 * Neoptimal version of the matrix multiplication algorithm 
 * 
 * @param: N: int the number of elements per line/column in each matrix
 * @param: A: double* the first matrix stored as a vector
 * @param: B: double* the second matrix stored as a vector
 * @return: The matrix A and B multiplication result 
 * 
 **/ 
double* multiply_neopt(int N, double *A, double *B) 
{   
    size_t li;
    size_t ci;
    size_t hi;
    double *R = (double *) malloc(N * N * sizeof(double));
    if (R == NULL)
        exit(BAD_ALLOC);

    for (li = 0; li < N; li++) {
      for (ci = 0; ci < N; ci++) {
        for (hi = 0; hi < N; hi++) {
              R[li * N  + ci] = A[li * N + hi] * B[hi * N + ci];
        }
      }
    }
    
    return R;
}

/**
 * Optimal version of the matrix multiplication algorithm 
 * using Strassen technique.
 * 
 * @param: N: int number of elements per line/column
 * @param: A: double* the first matrix stored as a vector
 * @param: B: double* the second matrix stored as a vector
 * @return: The matrix A and B multiplication result
 * 
 **/ 
double* multiply_opt(int N, double* A, double* B)
{
    double* R = (double *) malloc(N * N * sizeof(double));
    if (R == NULL)
        exit(BAD_ALLOC);

    /* The submatrixes size computation */
    int n = N / 2;
    size_t li;
    size_t ci;

    /* The workflow matrixes for computation result */
    double* M1;
    double* M2;
    double* M3;
    double* M4;
    double* M5;
    double* M6;
    double* M7;
    
    /* Matrix A subdivisions */
    double* A11;
    double* A12;
    double* A21;
    double* A22;

    /* Matrix B subdivisions */
    double* B11;
    double* B12;
    double* B21;
    double* B22;

    /* Matrix C subdivisions */
    double* C11;
    double* C12;
    double* C21;
    double* C22;

    /* Allocate space for all auxiliary variables */
    A11 = (double *) malloc(n * n * sizeof(double));
    if (A11 == NULL)
        exit(BAD_ALLOC);
    A12 = (double *) malloc(n * n * sizeof(double));
    if (A12 == NULL)
        exit(BAD_ALLOC);
    A21 = (double *) malloc(n * n * sizeof(double));
    if (A21 == NULL)
        exit(BAD_ALLOC);
    A22 = (double *) malloc(n * n * sizeof(double));
    if (A22 == NULL)
        exit(BAD_ALLOC);

    B11 = (double *) malloc(n * n * sizeof(double));
    if (B11 == NULL)
        exit(BAD_ALLOC);
    B12 = (double *) malloc(n * n * sizeof(double));
    if (B12 == NULL)
        exit(BAD_ALLOC);
    B21 = (double *) malloc(n * n * sizeof(double));
    if (B21 == NULL)
        exit(BAD_ALLOC);
    B22 = (double *) malloc(n * n * sizeof(double));
    if (B22 == NULL)
        exit(BAD_ALLOC);

    /* Populate the submatrixes */
    /* Quadrant 1 */
    for (li = 0; li < n; li++) {
        for (ci = 0; ci < n; ci++) {
            A11[li * n + ci] = A[li * n + ci];
            B11[li * n + ci] = B[li * n + ci];   
        }
    }
    
    /* Quadrant 2 */
    for (li = 0; li < n; li++) {
        for (ci = n; ci < N; ci++) {
            A12[li * n + ci] = A[li * n + ci];
            B12[li * n + ci] = B[li * n + ci];   
        }
    }

    /* Quadrant 3 */
    for (li = n; li < N; li++) {
        for (ci = 0; ci < n; ci++) {
            A21[li * n + ci] = A[li * n + ci];
            B21[li * n + ci] = B[li * n + ci];   
        }
    }

    /* Quadrant 4 */
    for (li = n; li < N; li++) {
        for (ci = n; ci < N; ci++) {
            A22[li * n + ci] = A[li * n + ci];
            B22[li * n + ci] = B[li * n + ci];   
        }
    }

    /* Process intermediate matrixes */
    M1 = matrix_add(n, matrix_add(n, A11, A22), matrix_add(n, B11, B22));
    M2 = multiply_neopt(n, matrix_add(n, A21, A22), B11);
    M3 = multiply_neopt(n, A11, matrix_substract(n, B12, B22));
    M4 = multiply_neopt(n, A22, matrix_substract(n, B21, B11));
    M5 = multiply_neopt(n, matrix_add(n, A11, A12), B22);
    M6 = multiply_neopt(n, matrix_substract(n, A21, A11), matrix_add(n, B11, B12));
    M7 = multiply_neopt(n, matrix_substract(n, A12, A22), matrix_add(n, B21, B22));

    C11 = matrix_add(n, matrix_add(n, M1, M4), matrix_substract(n, M7, M5));
    C12 = matrix_add(n, M3, M5);
    C21 = matrix_add(n, M2, M4);
    C22 = matrix_add(n, matrix_substract(n, M1, M2), matrix_add(n, M3, M6));

    /* Compute the Result matrix; each quadrant */
    /* Quadrant 1 */
    int C11_index = 0;
    for (li = 0; li < n; li++) {
        for (ci = 0; ci < n; ci++) {
           R[li * n + ci] = C11[C11_index];
           C11_index++;
        }
    }
    
    /* Quadrant 2 */
    int C12_index = 0;
    for (li = 0; li < N; li++) {
        for (ci = n; ci < N; ci++) {
           R[li * n + ci] = C12[C12_index];
           C12_index++;
        }
    }

    /* Quadrant 3 */
    int C21_index = 0;
    for (li = n; li < N; li++) {
        for (ci = 0; ci < n; ci++) {
            R[li * n + ci] = C21[C21_index];
            C21_index++;
        }
    }

    /* Quadrant 4 */
    int C22_index = 0;
    for (li = n; li < N; li++) {
        for (ci = n; ci < N; ci++) {
            R[li * n + ci] = C22[C22_index];
            C22_index++;
        }
    }

    /* Eliberate memory */
    free(M1);
    free(M2);
    free(M3);
    free(M4);
    free(M5);
    free(M6);
    free(M7);

    free(A11);
    free(A12);
    free(A21);
    free(A22);

    free(B11);
    free(B12);
    free(B21);
    free(B22);

    free(C11);
    free(C12);
    free(C21);
    free(C22);

    return R;
}   

#endif // _MATRIX_HELPERS_