/*
 * allocation example - with timing
 * * pinned memory
 * * host memory
 * * device memory
 * * host memory registration (pinned)
 */
#include <stdio.h>
#include <cuda_runtime.h>
#include "../colours.h"


typedef enum {
    PINNED,
    DEVICE,
    HOST_REG
} AllocType;

float profileMemory(AllocType alloc, size_t size, unsigned int flags) {
    cudaEvent_t start, stop;
    cudaError_t err = cudaSuccess;
    void* devPtr = NULL;
    void* hostPtr = malloc(size);
    float ms = 0;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);


    switch(alloc) {
        case PINNED:
            err = cudaHostAlloc(&devPtr, size, flags);
            break;

        case DEVICE:
            err = cudaMalloc(&devPtr, size);
            break;

        case HOST_REG:
            err = cudaHostRegister(hostPtr, size, flags);
            break;
            
        default:
            fprintf(stderr, " [!] unknown value\n");
    }
    if (err != cudaSuccess) {
        fprintf(stderr, "[!] Error: %s\n", cudaGetErrorString(err));
        return 0.0f;
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    if (devPtr != NULL) {
        cudaFree(devPtr);
    }
    if (alloc == HOST_REG) {
        cudaHostUnregister(hostPtr);
    }
    if (hostPtr != NULL) {
        free(hostPtr);
    }
    return ms;
}

void printResults(float ms, const char* colour) {
        printf("> %s%f%s ms\n\n", colour, ms, WHITE);
}
/**
 * Host main routine
 */
int
main(void)
{
    size_t samples[] = {512, 1024*1024, 1024*1024*200, 1024*1024*500};
    size_t num_samples = sizeof(samples) / sizeof(samples[0]);
    float ms = 0;
    for(int i=0; i < num_samples; i++) {
        size_t size = samples[i];
        printf("[*] using size %d bytes ( %s%.2f%s MB)\n", size, CYAN, (float)size/1024/1024, WHITE);

        // test device memory
        printf("> profile device memory\n");
        ms = profileMemory(DEVICE, size, 0);
        printResults(ms, RED);

        // test pinned memory
        printf("> profile default pinned memory\n");
        ms = profileMemory(PINNED, size, 0);
        printResults(ms, RED);

        printf("> profile portable pinned memory\n");
        ms = profileMemory(PINNED, size, cudaHostAllocPortable);
        printResults(ms, RED);

        printf("> profile mapped pinned memory\n");
        ms = profileMemory(PINNED, size, cudaHostAllocMapped);
        printResults(ms, RED);

        printf("> profile wc pinned memory\n");
        ms = profileMemory(PINNED, size, cudaHostAllocWriteCombined);
        printResults(ms, RED);

        printf("> profile registered host memory\n");
        ms = profileMemory(HOST_REG, size, cudaHostRegisterDefault);
        printResults(ms, RED);
    }
    return 0;
}

