#include <stdio.h>
#include "utils/utils.h"

#define NUM_ELEM    (16 * 1024 * 1024)

__global__ void kernel_compute(float* a, float* b, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    a[idx] = sin(b[idx]) * (1 - cos(b[idx]));
}

int compute_UnifiedMem(void) {
    float *host_a = 0;
    float *host_b = 0;

    // TODO-unified: Alocare memorie unificata

    fill_array_random(host_b, NUM_ELEM);

    kernel_compute<<<NUM_ELEM / 256, 256>>> (host_a, host_b, NUM_ELEM);
    
    // TODO-unified: Dealocare memorie unificata
    
    return 0;
}

int compute_NoUnifiedMem(void) {
    float *device_a = 0;
    float *device_b = 0;
    float *host_a = 0;
    float *host_b = 0;

    host_a = (float *) malloc(NUM_ELEM * sizeof(float));
    host_b = (float *) malloc(NUM_ELEM * sizeof(float));
    
    // TODO-plain: Alocare memorie (GPU/VRAM)
    if (host_a == 0 || host_b == 0 || device_a == 0 || device_b == 0) {
        printf("[HOST] Couldn't allocate memory\n");
    	return 1;
    }

    fill_array_random(host_b, NUM_ELEM);
    
    // TODO-plain: Copiere date host_b (CPU/RAM) => device_b (GPU/VRAM)

    kernel_compute<<<NUM_ELEM / 256, 256>>> (device_a, device_b, NUM_ELEM);
    
    // TODO-plain: Copiere device_b (GPU/VRAM) => date host_b (CPU/RAM)
    
    free(host_a);
    free(host_b);

    // TODO-plain: Dealocare memorie (GPU/VRAM)
    
    return 0;
}

int main(void) {
#ifdef NO_UNIFIED_MEMORY
    compute_NoUnifiedMem();
#else
    compute_UnifiedMem();
#endif
}
