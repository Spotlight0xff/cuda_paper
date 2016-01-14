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
    float *h_A, *d_A, *h_B, *d_B, *h_C, *d_C;

    // allocate host memory
    h_A = (float*) malloc(SIZE * sizeof(float));
    h_B = (float*) malloc(SIZE * sizeof(float));
    h_C = (float*) malloc(SIZE * sizeof(float));

    // allocate memory on device
    cudaMalloc(&d_A, SIZE * sizeof(float));
    cudaMalloc(&d_B, SIZE * sizeof(float));
    cudaMalloc(&d_C, SIZE * sizeof(float));


    for (int i=0; i < SIZE; i++) {
        h_A[i] = rand() / (float) RAND_MAX;
        h_B[i] = rand() / (float) RAND_MAX;
    }

    cudaMemcpy(d_A, h_A, SIZE * sizeof(float), cudaMemcpyDefault); // we're using UVA...
    cudaMemcpy(d_B, h_B, SIZE * sizeof(float), cudaMemcpyDefault);

    printf("> run vectorAdd using copied device memory...\n");
    dim3 block(256);
    dim3 grid((unsigned int) ceil(SIZE / block.x));

    // kernel call
    vectorAdd<<<grid, block>>>(d_A, d_B, d_C, SIZE);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("err: %s\n", cudaGetErrorString(err));
    }
    //vectorAdd<<<1024,1024>>>(d_A, d_B, d_C, SIZE);

    cudaMemcpy(h_C, d_C, SIZE * sizeof(float), cudaMemcpyDefault);

    cudaDeviceSynchronize();
    printf("> kernel call synchronized\n");
    printf("%f vs %f\n", h_A[123]+h_B[123], h_C[123]);

    printf("> releasing host memory...\n");

    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(h_A);
    cudaFree(h_B);
    cudaFree(h_C);

    printf("> done\n");
    return 0;
}

