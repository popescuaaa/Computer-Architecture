#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <sys/time.h>

#define limit 			1000
#define elem_limits		100


int main(void)
{
	double a[limit][limit];
	double b[limit][limit];
	double c[limit][limit];
	int i;
	int j;
	int k;
	time_t t;
	
	/* for performace evaluation purpose only */
	struct timeval start, end;

	srand((unsigned) time(&t));

	for (i = 0; i < limit; i++)
	{
		for (j = 0; j < limit; j++)
		{
			a[i][j] = rand() % elem_limits;
			b[i][j] = rand() % elem_limits;
		}
	}

	gettimeofday(&start, NULL);

	for (i = 0; i < limit; i++)
	{
		for (j = 0; j < limit; j++)
		{
			for (k = 0; k < limit; ++k)
			{
				c[i][j] += a[i][k] * b[k][j];
			}
		}
	}

	gettimeofday(&end, NULL);

	float elapsed = ((end.tv_sec - start.tv_sec) * 1000000.0f
		+ end.tv_usec - start.tv_usec) / 1000000.0f;

	printf("Time for limit = %d elems per line / column in matrix \
			is %f seconds.\n", limit, elapsed);

	return 0;
}