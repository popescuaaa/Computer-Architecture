#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
 
#define BAD_ALLOC           12

void BMMultiply(int n, double** a, double** b, double** c)
{
    int bi=0;
    int bj=0;
    int bk=0;
    int i=0;
    int j=0;
    int k=0;
    // TODO: set block dimension blockSize
    int blockSize=100; 
 
    for(bi=0; bi<n; bi+=blockSize)
        for(bj=0; bj<n; bj+=blockSize)
            for(bk=0; bk<n; bk+=blockSize)
                for(i=0; i<blockSize; i++)
                    for(j=0; j<blockSize; j++)
                        for(k=0; k<blockSize; k++)
                            c[bi+i][bj+j] += a[bi+i][bk+k]*b[bk+k][bj+j];

}
 
int main(void)
{
    int n;
    double** a;
    double** b;
    double** c;
    int numreps = 10;
    int i=0;
    int j=0;
    struct timeval tv1, tv2;
    struct timezone tz;
    double elapsed;
    // TODO: set matrix dimension n
    n = 500;
    // allocate memory for the matrices
    
    // TODO: allocate matrices A, B & C
    ///////////////////// Matrix A //////////////////////////
    // TODO ...
    a = (double **) malloc(sizeof(double *) * n);
    if (a == NULL)
    {
        exit(BAD_ALLOC);
    }

    for (i = 0; i < n; i++)
    {
        a[i] = (double *) malloc(sizeof(double) * n);
        if (a[i] == NULL)
        {
            exit(BAD_ALLOC);
        }
    }

    ///////////////////// Matrix B ////////////////////////// 
    // TODO ...
    
    b = (double **) malloc(sizeof(double *) * n);
    if (b == NULL)
    {
        exit(BAD_ALLOC);
    }

    for (i = 0; i < n; i++)
    {
        b[i] = (double *) malloc(sizeof(double) * n);
        if (b[i] == NULL)
        {
            exit(BAD_ALLOC);
        }
    }

    ///////////////////// Matrix C //////////////////////////
    // TODO ...
    
    c = (double **) malloc(sizeof(double *) * n);
    if (c == NULL)
    {
        exit(BAD_ALLOC);
    }

    for (i = 0; i < n; i++)
    {
        c[i] = (double *) malloc(sizeof(double) * n);
        if (c[i] == NULL)
        {
            exit(BAD_ALLOC);
        }
    }

    // Initialize matrices A & B
    for(i=0; i<n; i++)
    {
        for(j=0; j<n; j++)
        {
            a[i][j] = 1;
            b[i][j] = 2;
        }
    }
 
    //multiply matrices
 
    printf("Multiply matrices %d times...\n", numreps);
    for (i=0; i<numreps; i++)
    {
        gettimeofday(&tv1, &tz);
        BMMultiply(n, a, b, c);
        gettimeofday(&tv2, &tz);
        elapsed += (double) (tv2.tv_sec-tv1.tv_sec) + (double) (tv2.tv_usec-tv1.tv_usec) * 1.e-6;
    }
    printf("Time = %lf \n",elapsed);
 
    //deallocate memory for matrices A, B & C
    // TODO ...
    
    for (i = 0; i < n; i++)
    {
       free(a[i]);
       free(b[i]);
       free(c[i]);
    }

    free(a);
    free(b);
    free(c);
    
    return 0;
}