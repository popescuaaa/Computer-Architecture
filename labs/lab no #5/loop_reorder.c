#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

#define limit 			1000
#define elem_limit 		100
#define BAD_ALLOC 		12

int main(void)
{
	double *a = (double *) malloc (sizeof(double) * limit * limit);
	if (a == NULL)
	{
		exit(BAD_ALLOC);
	}

	double *b  = (double *) malloc (sizeof(double) * limit * limit);
	if (b == NULL)
	{
		exit(BAD_ALLOC);
	}

	double *c = (double *) malloc (sizeof(double) * limit * limit);
	if (c == NULL)
	{
		exit(BAD_ALLOC);
	}
	

	int i, j, k;

	/* for performance evaluation purpose only */
	struct timeval start, end;
	time_t t;	
	
	srand((unsigned) time(&t));

	for (i = 0; i < limit; i++)
	{
		for (j = 0; j < limit; j++)
		{
			a[i * limit + j] = rand() % elem_limit;
			b[i * limit + j] = rand() % elem_limit;		
		}
	}
	
	
	gettimeofday(&start, NULL);

	for (k = 0; k < limit; k++)
	{
		for (i = 0; i < limit; i++)
		{

			for (j = 0; j < limit; j++)
			{
				c[i * limit + j] = a[i * limit + k] * b[k * limit + j];
			}
		}
	}

	gettimeofday(&end, NULL);

	float elapsed = ((end.tv_sec - start.tv_sec) * 1000000.0f
		+ end.tv_usec - start.tv_usec) / 1000000.0f;
	
	printf("Time for limit = %d elems per line in matrix \ 
		 is %f seconds.\n", limit, elapsed);

	return 0;
}
