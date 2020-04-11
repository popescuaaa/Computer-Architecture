#ifndef _MATRIX_HELPERS_
#define _MATRIX_HELPERS_

#include <stdlib.h>

#define BAD_ALLOC 12

double* transpose(int N, double *M) {
    double *T = (double*) malloc(N * N * sizeof(double));
    if (T == NULL)
        exit(BAD_ALLOC);

    for (size_t li = 0; li < N; li++) {
        for (size_t ci = 0; ci < N; ci++) {
            T[li * N + ci] = M[ci * N + li];
        }
    }
    
    return T;
}   


#endif // _MATRIX_HELPERS_