/*
 * Zero-Copy example, using vector addition as showcase
 */
#include <stdio.h>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

#define SIZE 1000

// CUDA kernel, using zerocopy
__global__ void
vectorAdd(const float *A, const float *B, float *C, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
        C[i] = A[i] + B[i];
    }
}

/**
 * Host main routine
 */
int
main(void)
{
    cudaError_t err = cudaSuccess;
    float *h_A, *map_A, *h_B, *map_B;
    cudaDeviceProp prop;

    cudaGetDeviceProperties(&prop, 0);
    if (!prop.canMapHostMemory) {
        fprintf(stderr, "device can't map host memory -> no zero-copy\n");
        exit(0);
    }
    cudaSetDeviceFlags(cudaDeviceMapHost); // enable mapped hostmem
    cudaHostAlloc(&h_A, SIZE, cudaHostAllocMapped);
    cudaHostAlloc(&h_B, SIZE, cudaHostAllocMapped);

    if (prop.unifiedAddressing) { // UVA is enabled
        //kernel<<<gridSize, blockSize>>> (h_A, h_B);
    }else {
        cudaHostGetDevicePointer(&map_A, h_A, 0); // get mapped ptr
        cudaHostGetDevicePointer(&map_B, h_B, 0); // get mapped ptr
        // kernel<<<gridSize, blockSize>> (map_A, map_B);
        // find a suitable kernel:
        // * short
        // * uses two pointer, store result in them aswell
    }

    err = cudaDeviceReset();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    printf("Done\n");
    return 0;
}

