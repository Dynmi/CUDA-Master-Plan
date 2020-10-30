/**
 * 
 * author@Haris Wang       
 * 	2020.10.7
 * 
 * */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"



// initialize vector with random value
void init_vector(float *x, int n)
{
	for (int i=0; i<n; i++)
	{
		x[i] = (float)rand() % 1000;
	}
}


__global__ void vector_add(float *A, float *B, float *C)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	C[idx] = A[idx] + B[idx];
}


int main(void)
{	
		
	float *h_a, *h_b, *h_c, *d_a, *d_b, *d_c;
	int n = 1024;

	// alloc host memory
	h_a = (float*)malloc(n * sizeof(float));
	h_b = (float*)malloc(n * sizeof(float));
	h_c = (float*)malloc(n * sizeof(float));

	// alloc gpu memory
	cudaMalloc((void**)&d_a, n * sizeof(float));
	cudaMalloc((void**)&d_b, n * sizeof(float));
	cudaMalloc((void**)&d_c, n * sizeof(float));

	// copy data from host memory to gpu memory
	cudaMemcpy(d_a, h_a, 4 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, 4 * sizeof(float), cudaMemcpyHostToDevice);


	float time;
	cudaEvent_t gpustart, gpustop;
	cudaEventCreate(&gpustart);
	cudaEventCreate(&gpustop);
	cudaEventRecord(gpustart, 0);
	vector_add <<< ceil(n/512), 512 >>> (d_a, d_b, d_c);
	cudaDeviceSynchronize();
	cudaEventRecord(gpustop, 0);
	cudaEventSynchronize(gpustop);
	cudaEventElapsedTime(&time, gpustart, gpustop);
	cudaEventDestroy(gpustart);
	cudaEventDestroy(gpustop);

	// copy data from gpu memory to host memory
	cudaMemcpy(h_c, d_c, n * sizeof(float), cudaMemcpyDeviceToHost);

	printf("------------------------------------------------------------------------------------\n\n\n");
	printf("Time cost on GPU is %.2f ms \n",time);
	printf("------------------------------------------------------------------------------------\n\n\n");


	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
	free(h_a);
	free(h_b);
	free(h_c);

	return 0;
}