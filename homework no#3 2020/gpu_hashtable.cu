/* 
 * @author: Popescu Andrei Gabriel 333CA
 * @category: CUDA GPU programming
 *   
 *  The following code contains a minimalist implementation of a HashTable
 *  that uses the linear probing insertion methodology.
 *   
 *
 */


#include <iostream>
#include <limits.h>
#include <stdlib.h>
#include <ctime>
#include <sstream>
#include <string>
#include <stdio.h>
#include <math.h>

#include "gpu_hashtable.hpp"

/*
 * CUDA Function for exposed HashTable API
 *
 * */
#define LOAD_FACTOR                     0.9f
#define FAIL                            false
#define SUCCESS                         true


/* INIT HASH
 */
 GpuHashTable::GpuHashTable(int size) {
    limitSize = size;
    currentSize = 0;

    cudaMalloc(&hashTableBuckets, limitSize * sizeof(HashTableEntry));
    cudaMalloc(&currentSize, sizeof(int));

    if (hashTableBuckets == 0 || currentSize == 0) {
        cerr << "[HOST] Couldn't allocate memory for GpuHashTable!\n";
    }

    cudaMemset(hashTableBuckets, 0, limitSize * sizeof(HashTableEntry));
    cudaMemset(currentSize, 0, sizeof(int));
}

/* DESTROY HASH
 */
GpuHashTable::~GpuHashTable() {
    cudaFree(hashTableBuckets);
    cudaFree(currentSize);
}

/*
 *  Device functions cannot be called from host functions
 *  so basically is a API exposure problem to make this
 *  a kernel function or even a classical host function
 *              
 *  32 bits Murmur3 Hash function called < avalanche >
 *
 *  https://www.sderosiaux.com/articles/2017/08/26/the-murmur3-hash-function--hashtables-bloom-filters-hyperloglog/
 **/

__device__ int getHash(int data, int limit) {
    
    data ^= data >> 16;
    data *= 0x85ebca6b;
    data ^= data >> 13;
    data *= 0xc2b2ae35;
    data ^= data >> 16;
    return data % limit;
    //return ((long)abs(data) * 334496971) %  1844674407370955155 % limit;

}

__global__ void kernelInsertEntry(int *keys, int *values, int *currentSize, int numKeys, HashTableEntry *hashTableBuckets, int limitSize) {

    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
  
    if (threadId >= numKeys)
        return;
    
    int currentKey = keys[threadId];
    int currentValue = values[threadId];
    int hash = getHash(currentKey, limitSize);
    
    while(true) {
        if (hashTableBuckets[hash].HashTableEntryKey == KEY_INVALID) {
            atomicAdd(currentSize, 1);
            atomicExch(&hashTableBuckets[hash].HashTableEntryKey, currentKey);
            atomicExch(&hashTableBuckets[hash].HashTableEntryValue, currentValue);
            return;
        } else {
            if (hashTableBuckets[hash].HashTableEntryKey == currentKey) {
                atomicExch(&hashTableBuckets[hash].HashTableEntryValue, currentValue);
                return;
            }
        }
        //int inplaceKey = atomicCAS(&hashTableBuckets[hash].HashTableEntryKey, KEY_INVALID, currentKey);

        // if (inplaceKey == currentKey || inplaceKey == KEY_INVALID) {
        //     if (inplaceKey == currentKey)
                
        //     return;
        // }

        hash = (hash + 1) % limitSize;
    }
}


/* INSERT BATCH
 */
 bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) {
    cout << "Insert batch is called...\n";

    for (int i = 0; i < numKeys; i++) {
        cout << " Insert-> key: " << keys[i] << " " << "value: " << values[i] << endl;
    }

    int currentSizeCPU;
    cudaMemcpy(&currentSizeCPU, currentSize, sizeof(int), cudaMemcpyDeviceToHost);
    int futureLoadFactor = (float) (currentSizeCPU + numKeys) / limitSize;
   
    if (futureLoadFactor >= LOAD_FACTOR) {
        reshape(limitSize + numKeys);
    }

    int *deviceKeys;
    int *deviceValues;

    cudaMalloc(&deviceKeys, numKeys * sizeof(int));
    cudaMalloc(&deviceValues, numKeys * sizeof(int));

    if (deviceValues == 0 || deviceKeys == 0) {
        cerr << "[HOST] Couldn't allocate memory for device keys or values arrays!\n";
        return FAIL;
    }

    cudaMemcpy(deviceKeys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceValues, values, numKeys * sizeof(int), cudaMemcpyHostToDevice);

    /* Use CUDA to computer the optimal blockSize and WORKER/block */
    int minGridSize;
    int threadBlockSize = 512;
    //cudaOccupancyMaxPotentialBlockSize(&minGridSize, &threadBlockSize, kernelInsertEntry, 0, 0);
    cout << threadBlockSize << endl;
    int gridSize = ceil(numKeys/ threadBlockSize) + 1 ;
    cout << gridSize << endl;

    kernelInsertEntry<<< gridSize, threadBlockSize >>>(
            deviceKeys,
            deviceValues,
            currentSize,
            numKeys,
            hashTableBuckets,
            limitSize
            );

    cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess)
        printf("kernel launch failed with error \"%s\".\n", cudaGetErrorString(cudaerr));

    cudaFree(deviceKeys);
    cudaFree(deviceValues);
   

	return SUCCESS;
}


__global__ void kernelGetEntry( int *keys, int *values, int numKeys, int limitSize, HashTableEntry *hashTableBuckets) {

    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadId >= numKeys)
        return;

    int currentKey = keys[threadId];
    int hash = getHash(currentKey, limitSize);
   
    while(true) {
        if (hashTableBuckets[hash].HashTableEntryKey == currentKey) {
            atomicExch(&values[threadId], hashTableBuckets[hash].HashTableEntryValue);
            return;
        }

        if (hashTableBuckets[hash].HashTableEntryKey == KEY_INVALID) {
            atomicExch(&values[threadId], 0);
            return;
        }

        hash = (hash + 1) % limitSize;
    }
}


/* GET BATCH
 */
 int* GpuHashTable::getBatch(int* keys, int numKeys) {
    int *deviceKeys;
    int *values;
    int *deviceValues;

    cudaMalloc(&deviceKeys, numKeys * sizeof(int));
    cudaMalloc(&deviceValues, numKeys * sizeof(int));

    if (deviceKeys == 0 ||  deviceValues == 0) {
        cerr << "[HOST] Couldn't allocate memory for device keys GPU arrays!\n";
        return NULL;
    }

    cudaMemcpy(deviceKeys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);

    int minGridSize;
    int threadBlockSize = 512;
    //cudaOccupancyMaxPotentialBlockSize(&minGridSize, &threadBlockSize, kernelInsertEntry, 0, 0);

    int gridSize = ceil(numKeys / threadBlockSize) + 1;

    kernelGetEntry<<< gridSize, threadBlockSize >>>(
            deviceKeys,
            deviceValues,
            numKeys,
            limitSize,
            hashTableBuckets
            );

    cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess)
        printf("kernel launch failed with error \"%s\".\n", cudaGetErrorString(cudaerr));
        
    values = (int *) calloc(numKeys, sizeof(int));
    if (values == NULL) {
        cerr << "[HOST] Couldn't allocate memory for thre return values array!\n";
        return NULL;
    }

    cudaMemcpy(values, deviceValues, numKeys * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(deviceValues);
    cudaFree(deviceKeys);

    for (int i = 0; i < numKeys; i++) {
        cout << " Got-> key: " << keys[i] << " " << "value: " << values[i] << endl;
    }


    return values;
}


__global__ void kernelCopyTable(
        HashTableEntry *hashTableBucketsOrig,
        int limitSizeOrig,
        HashTableEntry *hashTableBuckets,
        int limitSize) {

    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadId >= limitSizeOrig)
        return;

    if (hashTableBucketsOrig[threadId].HashTableEntryKey == KEY_INVALID)
        return;

    int currentKey = hashTableBucketsOrig[threadId].HashTableEntryKey;
    int currentValue = hashTableBucketsOrig[threadId].HashTableEntryValue;
    int hash = getHash(currentKey, limitSize);
    
    while (true) {
        int inplaceKey = atomicCAS(&hashTableBuckets[hash].HashTableEntryKey, KEY_INVALID, currentKey);
        if (inplaceKey == currentKey || inplaceKey == KEY_INVALID) {
            atomicExch(&hashTableBuckets[hash].HashTableEntryValue, currentValue);
            return;
        }
        
        hash = (hash + 1) % limitSize;
    }
}

/* RESHAPE HASH
 */
void GpuHashTable::reshape(int numBucketsReshape) {
    HashTableEntry *hashTableBucketsReshaped;
    int newLimitSize = numBucketsReshape;

    cudaMalloc(&hashTableBucketsReshaped, newLimitSize * sizeof(HashTableEntry));

    if (hashTableBucketsReshaped == 0)
        cerr << "[HOST] Couldn't allocate memory for GpuHashTable Reshape!\n";

    cudaMemset(hashTableBucketsReshaped, 0, newLimitSize * sizeof(HashTableEntry));

    int minGridSize;
    int threadBlockSize = 512;
    //cudaOccupancyMaxPotentialBlockSize(&minGridSize, &threadBlockSize, kernelInsertEntry, 0, 0);
    
    int gridSize = ceil(limitSize / threadBlockSize) + 1;

    kernelCopyTable<<< gridSize, threadBlockSize >>>(
            hashTableBuckets,
            limitSize,
            hashTableBucketsReshaped,
            newLimitSize
            );

    cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess)
        printf("kernel launch failed with error \"%s\".\n", cudaGetErrorString(cudaerr));
    
    cudaFree(hashTableBuckets);
    
    hashTableBuckets = hashTableBucketsReshaped;
    limitSize = newLimitSize;
}

/* GET LOAD FACTOR
 * num elements / hash total slots elements
 */
float GpuHashTable::loadFactor() {
    int currentSizeCPU;
    cudaMemcpy(&currentSizeCPU, currentSize, sizeof(int), cudaMemcpyDeviceToHost);

    if (currentSizeCPU != 0)
        return 0.f;
    else
        return (float) currentSizeCPU / limitSize;
}

/*********************************************************/

#define HASH_INIT GpuHashTable GpuHashTable(1);
#define HASH_RESERVE(size) GpuHashTable.reshape(size);

#define HASH_BATCH_INSERT(keys, values, numKeys) GpuHashTable.insertBatch(keys, values, numKeys)
#define HASH_BATCH_GET(keys, numKeys) GpuHashTable.getBatch(keys, numKeys)

#define HASH_LOAD_FACTOR GpuHashTable.loadFactor()

#include "test_map.cpp"
 