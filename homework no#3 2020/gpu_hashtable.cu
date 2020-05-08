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

__device__ int getHash(int hashValue, int hashLimit) {
    return hash(hashValue, hashLimit);
}

__global__ void kernelInsertEntry() {

}

__global__ void kernelGetEntry() {

}

__global__ void kernelInsertBatch() {

}

__global__ void kernelGetBatch() {

}

__global__ void kernelCopyEntry(GpuHashTable hashTableOrig, GpuHashTable hashTable) {

}
/* INIT HASH
 */
GpuHashTable::GpuHashTable(int size) {
    limitSize = size;
    currentSize = 0;

    cudaMallocManaged(&hashTableBuckets, limitSize * BUCKET_SIZE * sizeof(HashTableEntry));
    if (hashTableBuckets == 0) {
        cerr << "[HOST] Couldn't allocate memory for GpuHashTable\n";
    }
    cudaMemset(hashTableBuckets, 0, limitSize * BUCKET_SIZE * sizeof(HashTableEntry));

}

/* DESTROY HASH
 */
GpuHashTable::~GpuHashTable() {
    cudaFree(hashTableBuckets);
}

/* RESHAPE HASH
 */
void GpuHashTable::reshape(int numBucketsReshape) {
}

/* INSERT BATCH
 */
bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) {
	return false;
}

/* GET BATCH
 */
int* GpuHashTable::getBatch(int* keys, int numKeys) {
	return NULL;
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
