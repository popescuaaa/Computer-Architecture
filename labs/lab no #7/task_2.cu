#include <stdio.h>
#include <math.h>
#include "utils/utils.h"

// TODO 6: Write the code to add the two arrays element by element and 
// store the result in another array
__global__ void add_arrays(const float *a, const float *b, float *c, int N) {
    
}

int main(void) {
    cudaSetDevice(0);
    int N = 1 << 20;

    float *host_array_a = 0;
    float *host_array_b = 0;
    float *host_array_c = 0;

    float *device_array_a = 0;
    float *device_array_b = 0;
    float *device_array_c = 0;

    // TODO 1: Allocate the host's arrays

    // TODO 2: Allocate the device's arrays

    // TODO 3: Check for allocation errors

    // TODO 4: Fill array with values; use fill_array_float to fill
    // host_array_a and fill_array_random to fill host_array_b. Each
    // function has the signature (float *a, int n), where n = number of elements.

    // TODO 5: Copy the host's arrays to device

    // TODO 6: Execute the kernel, calculating first the grid size
    // and the amount of threads in each block from the grid
    // Hint: For this execise the block_size can have any value lower than the
    //      API's maximum value (it's recommended to be close to the maximum
    //      value).

    // TODO 7: Copy back the results and then uncomment the checking function

    /* check_task_2(host_array_a, host_array_b, host_array_c, N); */

    // TODO 8: Free the memory
   
    return 0;
}