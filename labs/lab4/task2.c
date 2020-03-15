#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdint.h>     // provides int8_t, uint8_t, int16_t etc.
#include <stdlib.h>

#define BAD_ALLOC	12
#define RANDOM_RANGE	1000


struct particle
{
    int8_t v_x, v_y, v_z;
};

int main(int argc, char* argv[])
{ 
    long line;
    long column;

    if(argc < 2)
    {
        printf("apelati cu %s <n>\n", argv[0]);
        return -1;
    }

    long n = atol(argv[1]);

    // TODO
    // alocati dinamic o matrice de n x n elemente de tip struct particle
    // verificati daca operatia a reusit
    struct particle* mat = (struct particle*) malloc(n * n * sizeof(struct particle));

    if (mat == NULL) exit (BAD_ALLOC);


	// *liniile pare contin particule cu toate componentele vitezei pozitive
    //   -> folositi modulo 128 pentru a limita rezultatului lui rand()
    // *liniile impare contin particule cu toate componentele vitezi negative
    //   -> folositi modulo 129 pentru a limita rezultatului lui rand()
     
     for(line = 0; line < n; ++line)
    {
        for(column = 0; column < n; ++column)
        {
            if(line % 2 == 0)
            {
                mat[line * n + column].v_x = rand() % 128;
                mat[line * n + column].v_y = rand() % 128;
                mat[line * n + column].v_z = rand() % 128;
            }
            else
            {
                mat[line * n + column].v_x = -(rand() % 129);
                mat[line * n + column].v_y = -(rand() % 129);
                mat[line * n + column].v_z = -(rand() % 129);

            }
        }
    }

    // TODO
    // scalati vitezele tuturor particulelor cu 0.5
    //   -> folositi un cast la int8_t* pentru a parcurge vitezele fara
    //      a fi nevoie sa accesati individual componentele v_x, v_y, si v_z
    
    int8_t* p_speeds = (int8_t*) mat;
    for(long i = 0; i < 3 * n * n; ++i)
        p_speeds[i] = p_speeds[i] / 2;

    // compute max particle speed
    float max_speed = 0.0f;
    for(long i = 0; i < n * n; ++i)
    {
        float speed = sqrt(mat[i].v_x * mat[i].v_x +
                           mat[i].v_y * mat[i].v_y +
                           mat[i].v_z * mat[i].v_z);
        if(max_speed < speed) max_speed = speed;
    }

    // print result
    printf("viteza maxima este: %f\n", max_speed);

    free(mat);

    return 0;
}

