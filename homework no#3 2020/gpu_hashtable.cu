/* 
 * @author: Popescu Andrei Gabriel 333CA
 * @category: CUDA GPU programming
 *
 */


#include <iostream>
#include <limits.h>
#include <stdlib.h>
#include <ctime>
#include <sstream>
#include <string>

#include "gpu_hashtable.hpp"

/*
 * CUDA Function for exposed HashTable API
 *
 * */

/*
 * Constant values
 *
 **/
#define LOAD_FACTOR                     0.8f
#define DEFAULT_WORKERS_BLOCK           512
#define DEFAULT_STATUS                  -1
#define FAIL                            false
#define SUCCESS                         true

/*
 *  Device functions cannot be called from host functions
 *  so basically is a API exposure problem to make this
 *  a kernel function or even a classical host function
 *
 **/

__device__ int getHash(int data, int limit) {
    return (long)abs(data) % limit;
}

__global__ void kernelInsertEntry(
        int *keys,
        int *values,
        int numKeys,
        HashTableEntry *hashTableBuckets,
        int limitSize) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx > numKeys)
        return;

    int currentKey = keys[idx];
    int currentValue = values[idx];
    int hash = getHash(currentKey, limitSize);
    int status = DEFAULT_STATUS;

    /*
     * Searching from current position in hashTable to the end
     */
    for (int i = 0; i < limitSize - hash; i++) {
        status = atomicCAS(&hashTableBuckets[hash + i].HashTableEntryKey, KEY_INVALID, currentKey);

        if (status ==  DEFAULT_STATUS || status == currentKey) {
            /* Add new or replace */
            hashTableBuckets[hash + i].HashTableEntryKey = currentKey;
            hashTableBuckets[hash + i].HashTableEntryValue = currentValue;
            return;
        }
    }

}

__global__ void kernelGetEntry(
        int *keys,
        int *values,
        int numKeys,
        int limitSize,
        HashTableEntry *hashTableBuckets) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx > numKeys)
        return;

    int currentKey = keys[idx];
    int hash = getHash(currentKey, limitSize);

    for (int i = 0; i < limitSize - hash; i++) {
        if (hashTableBuckets[hash + i].HashTableEntryKey == currentKey) {
            /* Insert in the values vector */
            values[idx] = hashTableBuckets[hash + i].HashTableEntryValue;
            return;
        }
    }

}

__global__ void kernelCopyHashTable(
        HashTableEntry *hashTableBucketsOrig,
        int limitSizeOrig,
        HashTableEntry *hashTableBuckets) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx > limitSizeOrig)
        return;

    int status = DEFAULT_STATUS;
    int statusEmpty = -2;

    status = atomicCAS(&hashTableBuckets[idx].HashTableEntryKey, KEY_INVALID, statusEmpty);

    if (status == DEFAULT_STATUS) {
        hashTableBuckets[idx].HashTableEntryKey =
                hashTableBucketsOrig[idx].HashTableEntryKey;
        hashTableBuckets[idx].HashTableEntryValue =
                hashTableBucketsOrig[idx].HashTableEntryValue;
        return;
    }
}

/* INIT HASH
 */
GpuHashTable::GpuHashTable(int size) {
    limitSize = size;
    currentSize = 0;
    cout << "[HOST] Host is allocating right now...!\n";

    cudaMalloc(&hashTableBuckets, limitSize * sizeof(HashTableEntry));
    if (hashTableBuckets == 0) {
        cerr << "[HOST] Couldn't allocate memory for GpuHashTable!\n";
    }

    cout << "[HOST] Host has allocated right now...!\n";
    cudaMemset(hashTableBuckets, 0, limitSize * sizeof(HashTableEntry));
}

/* DESTROY HASH
 */
GpuHashTable::~GpuHashTable() {
    cudaFree(hashTableBuckets);
}

/* RESHAPE HASH
 */
void GpuHashTable::reshape(int numBucketsReshape) {
    HashTableEntry *hashTableBucketsReshaped;
    int newLimitSize = numBucketsReshape;

    cudaMallocManaged(&hashTableBucketsReshaped, newLimitSize * sizeof(HashTableEntry));

    if (hashTableBucketsReshaped == 0) {
        cerr << "[HOST] Couldn't allocate memory for GpuHashTable Reshape!\n";
    }

    cudaMemset(hashTableBucketsReshaped, 0, newLimitSize * sizeof(HashTableEntry));

    int blocks;
    if (limitSize % DEFAULT_WORKERS_BLOCK == 0)
        blocks = newLimitSize / DEFAULT_WORKERS_BLOCK;
    else
        blocks = newLimitSize / DEFAULT_WORKERS_BLOCK + 1;

    kernelCopyHashTable<<< blocks, DEFAULT_WORKERS_BLOCK >>>(hashTableBuckets, limitSize, hashTableBucketsReshaped);

    cudaDeviceSynchronize();
    cudaFree(hashTableBuckets);

    hashTableBuckets = hashTableBucketsReshaped;
    limitSize = newLimitSize;
}

/* INSERT BATCH
 */
bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) {
    int futureLoadFactor = (float) (currentSize + numKeys) / limitSize;
    if (futureLoadFactor > LOAD_FACTOR) {
        reshape(2 * limitSize);
    }
	
	currentSize += numKeys;
	
    int *deviceKeys;
    int *deviceValues;
    int blocks;
    if (numKeys % DEFAULT_WORKERS_BLOCK == 0)
        blocks = numKeys / DEFAULT_WORKERS_BLOCK;
    else
        blocks = numKeys / DEFAULT_WORKERS_BLOCK + 1;

    cudaMallocManaged(&deviceKeys, numKeys * sizeof(int));
    cudaMallocManaged(&deviceValues, numKeys * sizeof(int));

    if (deviceValues == 0 || deviceKeys == 0) {
        cerr << "[HOST] Couldn't allocate memory for device keys or values arrays!\n";
        return FAIL;
    }

    cudaMemcpy(deviceKeys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceValues, values, numKeys * sizeof(int), cudaMemcpyHostToDevice);

    kernelInsertEntry<<< blocks, DEFAULT_WORKERS_BLOCK >>>(
            deviceKeys,
            deviceValues,
            numKeys,
            hashTableBuckets,
            limitSize
            );

    cudaDeviceSynchronize();

    cudaFree(deviceKeys);
    cudaFree(deviceValues);

	return SUCCESS;
}

/* GET BATCH
 */
int* GpuHashTable::getBatch(int* keys, int numKeys) {
    int *deviceKeys;
    int *values;
    int *deviceValues;

    cudaMallocManaged(&deviceKeys, numKeys * sizeof(int));
    cudaMallocManaged(&deviceValues, numKeys * sizeof(int));
    values = (int *) malloc(numKeys * sizeof(int));

    if (deviceKeys == 0 || deviceValues == 0 || values == 0) {
        cerr << "[HOST] Couldn't allocate memory for device keys or values arrays!\n";
        return NULL;
    }

    cudaMemset(deviceValues, 0, numKeys * sizeof(int));
    cudaMemcpy(deviceKeys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);

    int blocks;
    if (numKeys % DEFAULT_WORKERS_BLOCK == 0)
        blocks = numKeys / DEFAULT_WORKERS_BLOCK;
    else
        blocks = numKeys / DEFAULT_WORKERS_BLOCK + 1;

    kernelGetEntry<<< blocks, DEFAULT_WORKERS_BLOCK >>>(
            keys,
            values,
            numKeys,
            limitSize,
            hashTableBuckets
            );

    cudaDeviceSynchronize();
    cudaMemcpy(values, deviceValues, numKeys * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(deviceValues);
    cudaFree(deviceKeys);
	
    return values;
}

/* GET LOAD FACTOR
 * num elements / hash total slots elements
 */
float GpuHashTable::loadFactor() {
    if (currentSize != 0) {
        // No elements in HashTable
        return 0.f;
    } else {
        return (float) currentSize / limitSize;
    }
}

/*********************************************************/

#define HASH_INIT GpuHashTable GpuHashTable(1);
#define HASH_RESERVE(size) GpuHashTable.reshape(size);

#define HASH_BATCH_INSERT(keys, values, numKeys) GpuHashTable.insertBatch(keys, values, numKeys)
#define HASH_BATCH_GET(keys, numKeys) GpuHashTable.getBatch(keys, numKeys)

#define HASH_LOAD_FACTOR GpuHashTable.loadFactor()

#include "test_map.cpp"
 