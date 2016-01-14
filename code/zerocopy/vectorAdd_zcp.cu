/*
 * Zero-Copy example, using vector addition as showcase
 */
#include <stdio.h>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

#define SIZE (1048576)

// CUDA kernel, using zerocopy
__global__ void
vectorAdd(float *A, float *B, float *C, int numElements)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id < SIZE) {
        C[id] = A[id] + B[id];
    }
}

/**
 * Host main routine
 */
int
main(void)
{
    float *h_A, *map_A, *h_B, *map_B, *h_C, *map_C;

    cudaSetDeviceFlags(cudaDeviceMapHost); // enable mapped hostmem
    cudaHostAlloc(&h_A, SIZE*sizeof(float), cudaHostAllocMapped);
    cudaHostAlloc(&h_B, SIZE*sizeof(float), cudaHostAllocMapped);
    cudaHostAlloc(&h_C, SIZE*sizeof(float), cudaHostAllocMapped);

    for (int i=0; i < SIZE; i++) {
        h_A[i] = rand() / (float) RAND_MAX;
        h_B[i] = rand() / (float) RAND_MAX;
    }

    cudaHostGetDevicePointer(&map_A, h_A, 0); // get mapped ptr
    cudaHostGetDevicePointer(&map_B, h_B, 0); // get mapped ptr
    cudaHostGetDevicePointer(&map_C, h_C, 0); // get mapped ptr
    printf("> run vectorAdd using mapped host memory...\n");
    dim3 block(256);
    dim3 grid((unsigned int) ceil(SIZE / block.x));

    vectorAdd<<<grid,block>>>(h_A, h_B, h_C, SIZE);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("err: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();

    printf("> releasing host memory...\n");
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C);

    printf("> done\n");
    return 0;
}

