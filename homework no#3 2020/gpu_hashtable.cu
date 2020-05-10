/**
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
 *  Device functions cannot be called from host functions
 *  so basically is a API exposure problem to make this
 *  a kernel function or even a classical host function
 *
 **/

__device__ int getHash(int data, int limit) {
    return ((long)abs(data) * 805306457) % 196613 % limit;
}

__global__ void kernelInsertEntry(
        int *keys,
        int *values,
        int numKeys,
        HashTableEntry *hashTableBuckets,
        int limitSize
        ) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx > numKeys)
        return;
    int currentKey = keys[idx];
    int currentValue = values[idx];
    int hash = getHash(currentKey, limitSize);
    int status = DEFAULT_STATUS;

    for (int i = 0; i < BUCKET_SIZE; i++) {
        status = atomicCAS(&hashTableBuckets[hash * BUCKET_SIZE + i].HashTableEntryKey,
                KEY_INVALID,
                currentKey);

        if (status == KEY_INVALID || status == currentKey) {
            /* Add new or replace */
            hashTableBuckets[hash * BUCKET_SIZE + i].HashTableEntryKey = currentKey;
            hashTableBuckets[hash * BUCKET_SIZE + i].HashTableEntryValue = currentValue;
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

    for (int i = 0; i < BUCKET_SIZE; i++) {
        if ( hashTableBuckets[hash * BUCKET_SIZE + i].HashTableEntryKey == currentKey ) {
            /* Insert in the values vector */
            values[idx] = hashTableBuckets[hash * BUCKET_SIZE + i].HashTableEntryValue;
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
    int notEmpty = 0;

    for (int i = 0; i < BUCKET_SIZE; i++) {
        status = atomicCAS(&hashTableBuckets[idx * BUCKET_SIZE + i].HashTableEntryKey,
                           KEY_INVALID,
                           notEmpty);
        if (status == notEmpty) {
            hashTableBuckets[idx * BUCKET_SIZE + i].HashTableEntryKey =
                    hashTableBuckets[idx * BUCKET_SIZE + i].HashTableEntryKey;
            hashTableBuckets[idx * BUCKET_SIZE + i].HashTableEntryValue =
                    hashTableBuckets[idx * BUCKET_SIZE + i].HashTableEntryValue;
            return;
        }

    }

}

/* INIT HASH
 */
GpuHashTable::GpuHashTable(int size) {
    limitSize = size;
    currentSize = 0;

    cudaMallocManaged(&hashTableBuckets,
            limitSize * BUCKET_SIZE * sizeof(HashTableEntry));
    if (hashTableBuckets == 0) {
        cerr << "[HOST] Couldn't allocate memory for GpuHashTable!\n";
    }
    cudaMemset(hashTableBuckets,
            0,
            limitSize * BUCKET_SIZE * sizeof(HashTableEntry));

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

    cudaMallocManaged(&hashTableBucketsReshaped,
                      newLimitSize * BUCKET_SIZE * sizeof(HashTableEntry));

    if (hashTableBucketsReshaped == 0) {
        cerr << "[HOST] Couldn't allocate memory for GpuHashTable Reshape!\n";
    }

    cudaMemset(hashTableBucketsReshaped,
               0,
               newLimitSize * BUCKET_SIZE * sizeof(HashTableEntry));

    int blocks;
    if (limitSize % DEFAULT_WORKERS_BLOCK == 0)
        blocks = newLimitSize / DEFAULT_WORKERS_BLOCK;
    else
        blocks = newLimitSize / DEFAULT_WORKERS_BLOCK + 1;

    kernelCopyHashTable<<< blocks, DEFAULT_WORKERS_BLOCK >>>(
            hashTableBuckets,
            limitSize,
            hashTableBucketsReshaped);

    cudaDeviceSynchronize();
    cudaFree(hashTableBuckets);

    hashTableBuckets = hashTableBucketsReshaped;
    limitSize = newLimitSize;
}

/* INSERT BATCH
 */
bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) {
    int currentLoadFactor = (float) (currentSize + numKeys) / limitSize;
    if (currentLoadFactor > LOAD_FACTOR) {
        reshape(limitSize + 3*numKeys);
    }

    int *deviceKeys;
    int *deviceValues;
    int blocks;
    if (numKeys % DEFAULT_WORKERS_BLOCK == 0)
        blocks = numKeys / DEFAULT_WORKERS_BLOCK;
    else
        blocks = numKeys / DEFAULT_WORKERS_BLOCK + 1;

    cudaMalloc(&deviceKeys, numKeys * sizeof(int));
    cudaMalloc(&deviceValues, numKeys * sizeof(int));

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
            limitSize);

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

    cudaMalloc(&deviceKeys, numKeys * sizeof(int));
    cudaMalloc(&values, numKeys * sizeof(int));

    if (deviceKeys == 0 || values == 0) {
        cerr << "[HOST] Couldn't allocate memory for device keys or values arrays!\n";
        return NULL;
    }

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
            hashTableBuckets);

    cudaDeviceSynchronize();

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
