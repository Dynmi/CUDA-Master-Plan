/**
 * 
 * author@Haris Wang       
 * 	2020.10.15
 * 
 * */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define BLOCK_SIZE 16


// initial matrix with random value
void init_matrix(float *arr, int size)
{
	for (int i = 0; i < size; i++)
	{
		arr[i] = (float)(rand() % 8 + 1);
	}
}

// matrix multiply on CPU
void matrix_mul_on_host(float *A, float *B, float *C, int M, int K, int N)
{
	for (int i = 0; i < M; i++)
	{
		for (int j = 0; j < N; j++)
		{
			C[i*N + j] = 0;
			for (int k = 0; k < K; k++)
			{
				C[i*N + j] += A[i*K + k] * B[k*N + j];
			}
		}
	}

}

// matrix multiply on GPU without shared memory
__global__ void matrix_mul_on_device(float *array_A, float *array_B, float *array_C, int M, int K, int N)
{
	int ix = threadIdx.x + blockDim.x*blockIdx.x;
	int iy = threadIdx.y + blockDim.y*blockIdx.y;

	if (ix < N && iy < M)
	{
		array_C[iy*N + ix] = 0;
		for (int k = 0; k < K; k++)
		{
			array_C[iy*N + ix] += array_A[iy*K + k] * array_B[k*N + ix];
		}
	}
}

// matrix multiply on GPU with shared memory
__global__ void matrix_mul_sharedMem(float *A, float *B, float *C, int M, int K, int N)
{

	__shared__ float sharedM[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ float sharedN[BLOCK_SIZE][BLOCK_SIZE];

	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;


	int row = by * BLOCK_SIZE + ty;
	int col = bx * BLOCK_SIZE + tx;

	float Csub = 0.0;

	for (int i = 0; i < (K+BLOCK_SIZE-1) / BLOCK_SIZE; i++)
	{

		if (i*BLOCK_SIZE + tx < K && row < M)
			sharedM[ty][tx] = A[row*K + i * BLOCK_SIZE + tx];
		else
			sharedM[ty][tx] = 0.0;

		if (i*BLOCK_SIZE + ty < K && col < N)
			sharedN[ty][tx] = B[(i*BLOCK_SIZE + ty)*N + col];
		else
			sharedN[ty][tx] = 0.0;
		__syncthreads();


		for (int j = 0; j < BLOCK_SIZE; j++)
			Csub += sharedM[ty][j] * sharedN[j][tx];
		__syncthreads();
	}


	if (row < M && col < N)
		C[row*N + col] = Csub;

}


int main(void)
{	
	int M,K,N;
	printf("Please enter M K N :\n");
    scanf("%d %d %d", &M, &K, &N);
		
	int Axy = M * K;
	int Bxy = K * N;
	int Cxy = M * N;


	float *h_A, *h_B, *hostRef, *deviceRef;
	h_A = (float*)malloc(Axy * sizeof(float));
	h_B = (float*)malloc(Bxy * sizeof(float));
	init_matrix(h_A, Axy);
	init_matrix(h_B, Bxy);

	hostRef = (float*)malloc(Cxy * sizeof(float));
	deviceRef = (float*)malloc(Cxy * sizeof(float));



	printf("\n");
	printf("------------------------------------------------------------------------------------\n");
	printf("Computing matrix product using matrix_mul_on_host \n");
	clock_t start = clock();
	matrix_mul_on_host(h_A, h_B, hostRef, M, K, N);
	clock_t finish = clock();
	float time = (float)(finish - start) / CLOCKS_PER_SEC * 1000;
	printf("Time cost on CPU is %.2f ms \n",time);
	printf("------------------------------------------------------------------------------------\n\n\n");




	printf("------------------------------------------------------------------------------------\n");
	printf("Computing matrix product using multiplicateMatrixshared \n");
	float *d_A, *d_B, *d_C;
	cudaMalloc((void**)&d_A, Axy * sizeof(float));
	cudaMalloc((void**)&d_B, Bxy * sizeof(float));
	cudaMalloc((void**)&d_C, Cxy * sizeof(float));

	cudaMemcpy(d_A, h_A, Axy * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, Bxy * sizeof(float), cudaMemcpyHostToDevice);

	dim3 block(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

	cudaEvent_t gpustart, gpustop;
	cudaEventCreate(&gpustart);
	cudaEventCreate(&gpustop);
	cudaEventRecord(gpustart, 0);
	matrix_mul_sharedMem <<< grid, block >>> (d_A, d_B, d_C, M, K, N);
	cudaDeviceSynchronize();
	cudaEventRecord(gpustop, 0);
	cudaEventSynchronize(gpustop);

	cudaEventElapsedTime(&time, gpustart, gpustop);
	cudaEventDestroy(gpustart);
	cudaEventDestroy(gpustop);

	cudaMemcpy(deviceRef, d_C, Cxy * sizeof(float), cudaMemcpyDeviceToHost);
	printf("Time cost on GPU using sharedMem is %.2f ms \n",time);
	printf("------------------------------------------------------------------------------------\n\n\n");



	printf("------------------------------------------------------------------------------------\n");
	printf("Computing matrix product using matrix_mul_on_device \n");

	cudaEventCreate(&gpustart);
	cudaEventCreate(&gpustop);
	cudaEventRecord(gpustart, 0);
	dim3	grid_2(256,256), 
			block_2(32,32);
	matrix_mul_on_device <<<grid_2,block_2>>> (d_A, d_B, d_C, M, K, N);
	cudaDeviceSynchronize();
	cudaEventRecord(gpustop, 0);
	cudaEventSynchronize(gpustop);

	cudaEventElapsedTime(&time, gpustart, gpustop);
	cudaEventDestroy(gpustart);
	cudaEventDestroy(gpustop);

	cudaMemcpy(deviceRef, d_C, Cxy * sizeof(float), cudaMemcpyDeviceToHost);
	printf("Time cost on GPU without sharedMem is %.2f ms \n",time);
	printf("------------------------------------------------------------------------------------\n\n\n");



	// Check the results
	/*     
	for(int i=0; i<Cxy; i++)
    {
        if(deviceRef[i]==hostRef[i])
        {
            printf("idx: %d passed !! \n", i);
        }
	} 
	*/

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	free(h_A);
	free(h_B);
	free(hostRef);
	free(deviceRef);
	
	return 0;
}