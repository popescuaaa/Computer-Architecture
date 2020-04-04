#include <stdio.h>
#include "utils/utils.h"

// ~TODO 3~
// Modify the kernel below such as each element of the 
// array will be now equal to 0 if it is an even number
// or 1, if it is an odd number
__global__ void kernel_parity_id(int *a, int N) {
	unsigned int local_index = threadIdx.x + blockDim.x * blockIdx.x;
    if (local_index < N) {
        if (a[local_index] % 2 == 0) {
            a[local_index] = 0;
        } else {
            a[local_index] = 1;
        }
    }
   
}

// ~TODO 4~
// Modify the kernel below such as each element will
// be equal to the BLOCK ID this computation takes
// place.
__global__ void kernel_block_id(int *a, int N) {
    unsigned int local_index = threadIdx.x + blockDim.x * blockIdx.x;
    if (local_index < n) {
    	a[local_index] = blockIdx.x;
  	}
}

// ~TODO 5~
// Modify the kernel below such as each element will
// be equal to the THREAD ID this computation takes
// place.
__global__ void kernel_thread_id(int *a, int N) {
    unsigned int local_index = threadIdx.x + blockDim.x * blockIdx.x;
    if (local_index < n) {
    	a[local_index] = threadIdx.x;
  	}
}

int main(void) {
    int nDevices;
    const int num_elements = 1 << 20;
    const int num_bytes = num_elements * sizeof(float);
    unsigned int index;

    // Get the number of CUDA-capable GPU(s)
    cudaGetDeviceCount(&nDevices);

    // ~TODO 1~
    // For each device, show some details in the format below, 
    // then set as active device the first one (assuming there
    // is at least CUDA-capable device). Pay attention to the
    // type of the fields in the cudaDeviceProp structure.
    //
    // Device number: <i>
    //      Device name: <name>
    //      Total memory: <mem>
    //      Memory Clock Rate (KHz): <mcr>
    //      Memory Bus Width (bits): <mbw>
    // 
    // Hint: look for cudaGetDeviceProperties and cudaSetDevice in
    // the Cuda Toolkit Documentation. 
    cudaDeviceProp properties;

    for (int i = 0; i < nDevices; ++i) {
	cudaGetDeviceProperties(&properties, i);
	
	printf("\t Device properties: << %d >> \n\n", i);
	printf("\t\t\n Device name: %s \n", properties.name);
	printf("\t\t\n Memory Clock Rate: %d \n", properties.clockRate);
	printf("\t\t\n Memory Bus Width: %d \n", properties.memoryBusWidth);

    }

    // ~TODO 2~
    // With information from example_2.cu, allocate an array with
    // integers (where a[i] = i). Then, modify the three kernels
    // above and execute them using 4 blocks, each with 4 threads.
    // Hint: num_elements = block_size * block_no (see example_2)
    //
    // You can use the fill_array_int(int *a, int n) function (from utils)
    // to fill your array as many times you want.
    float *host_array = 0;
    float *device_array = 0;

    host_array = (float *) malloc(num_bytes);
    if (host_array == 0) {
        printf("BAD ALLOC\n");
        exit(12);
    }

    cudaMalloc((void **) &device_array, num_bytes);
    if (device_array == 0) {
        printf("BAD ALLOC\n");
        exit(12);
    }

    for (index = 0; index < num_elements; index++) {
        host_array[index] = index;
    }
    
    cudaMemcpy(device_array, host_array, num_bytes, cudaMemcpyHostToDevice);

    const size_t block_size = 256;
    size_t blocks_no = num_elements / block_size;
      
    if (num_elements % block_size) 
		++blocks_no;

    // ~TODO 3~
    // Execute kernel_parity_id kernel and then copy from 
    // the device to the host; call cudaDeviceSynchronize()
    // after a kernel execution for safety purposes.
    //
    // Uncomment the line below to check your results
    kernel_parity_id<<<blocks_no, block_size>>>(device_array, num_elements);

    cudaMemcpy(host_array, device_array, num_bytes, cudaMemcpyDeviceToHost);

    check_task_1(3, host_array);

    // ~TODO 4~
    // Execute kernel_block_id kernel and then copy from 
    // the device to the host;
    //
    // Uncomment the line below to check your results

    kernel_block_id<<<blocks_no, block_size>>>(device_array, num_elements);

    cudaMemcpy(host_array, device_array, num_bytes, cudaMemcpyDeviceToHost);

    check_task_1(4, host_array); 

    // ~TODO 5~
    // Execute kernel_thread_id kernel and then copy from 
    // the device to the host;
    //
    // Uncomment the line below to check your results
    
    kernel_thread_id<<<blocks_no, block_size>>>(device_array, num_elements);

    cudaMemcpy(host_array, device_array, num_bytes, cudaMemcpyDeviceToHost);

    check_task_1(5, host_array);

    // TODO 6: Free the memory
    free(host_array);
	cudaFree(device_array);
    return 0;
}
