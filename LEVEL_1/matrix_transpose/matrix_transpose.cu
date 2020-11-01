#include <cassert>
#include <cuda_runtime.h>
#include <cstdio>

__global__ 
void naiveTransposeKernel(const float *input, float *output, int n) {
    const int i = threadIdx.x + 64 * blockIdx.x;
    int j = 4 * threadIdx.y + 64 * blockIdx.y;
    const int end_j = j + 4;

    for (; j < end_j; j++)
	{ 
		output[j + n * i] = input[i + n * j];
	}
        
}

__global__ 
void shmemTransposeKernel(const float *input, float *output, int n) {
    __shared__ float tile[65][65];
    int i = threadIdx.x + 64 * blockIdx.x;
    int j = threadIdx.y + 64 * blockIdx.y;

    for (int shift=0; shift<64; shift+=16)
		tile[threadIdx.y + shift][threadIdx.x] = input[i + n * (j+shift)];

    __syncthreads();

    i = threadIdx.x + 64 * blockIdx.y;
    j = threadIdx.y + 64 * blockIdx.x;
    for (int shift=0; shift<64; shift+=16)
        output[i + n * (j+shift)] = tile[threadIdx.x][threadIdx.y + shift];

}

__global__ 
void optimalTransposeKernel(const float *input, float *output, int n) {
    __shared__ float tile[65][65];
    int i = threadIdx.x + 64 * blockIdx.x;
    int j = threadIdx.y + 64 * blockIdx.y;

    for (int shift=0; shift<64; shift+=16)
    {
		tile[threadIdx.y + shift][threadIdx.x+0]  = input[i+0 + n * (j+shift)];
		tile[threadIdx.y + shift][threadIdx.x+32] = input[i+32 + n * (j+shift)];
    }
    __syncthreads();

    i = threadIdx.x + 64 * blockIdx.y;
    j = threadIdx.y + 64 * blockIdx.x;
    for (int shift=0; shift<64; shift+=16 )
    {
        output[i+0 + n * (j+shift)]  = tile[threadIdx.x+0][threadIdx.y + shift];
        output[i+32 + n * (j+shift)] = tile[threadIdx.x+32][threadIdx.y + shift];    
    }

}

void cudaTranspose(
    const float *d_input,
    float *d_output,
    int n,
    TransposeImplementation type)
{
    if (type == NAIVE) {
        dim3 blockSize(64, 16);
        dim3 gridSize(n / 64, n / 64);
        naiveTransposeKernel<<<gridSize, blockSize>>>(d_input, d_output, n);
    }
    else if (type == SHMEM) {
        dim3 blockSize(64, 16);
        dim3 gridSize(n / 64, n / 64);
        shmemTransposeKernel<<<gridSize, blockSize>>>(d_input, d_output, n);
    }
    else if (type == OPTIMAL) {
        dim3 blockSize(32, 16);
        dim3 gridSize(n / 64, n / 64);
        optimalTransposeKernel<<<gridSize, blockSize>>>(d_input, d_output, n);
    }
    else
        assert(false);
}