#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <ctime>
#include <iostream>
#define DATA_SIZE 1048576
#define THREAD_SIZE 256
#define BLOCK_SIZE 32
int data[DATA_SIZE];

void Generate(int* number, int size) {
	for (int i = 0; i < size; ++i)
		number[i] = rand() % 10;
}

__global__ static void sumOf(int* data, int* result, clock_t* time) {
	extern __shared__ int shared[];
	const int tid = threadIdx.x;
	const int bid = blockIdx.x;
	int sum = 0;
	clock_t start;
	int offset = 1, mask = 1;
	if (tid == 0)  time[bid] = clock();
	shared[tid] = 0;
	for (int i = bid * THREAD_SIZE + tid; i < DATA_SIZE; 
		i += BLOCK_SIZE * THREAD_SIZE) {
		shared[tid] += data[i] * data[i];
	}
	__syncthreads();
	offset = THREAD_SIZE / 2;
	while (offset > 0) {
		if (tid < offset) {
			shared[tid] += shared[tid + offset];
		}
		offset >>= 1;
		__syncthreads();
	}
	/*
	while (offset < THREAD_SIZE){
		if ((tid & mask) == 0) {
			shared[tid] += shared[tid + offset];
		}
		offset += offset;
		mask += offset;
		__syncthreads();
	}*/

	if (tid == 0) {
		result[bid] = shared[0];
		time[bid + BLOCK_SIZE] = clock();
	}

	/*
	if (tid == 0) {
		for (int i = 0; i < THREAD_SIZE; ++i)
			shared[0] += shared[i];
		result[bid] = shared[0];
		time[bid + BLOCK_SIZE] = clock();
	}
	*/
}

int main(int argc, char** argv) {
	Generate(data, DATA_SIZE);
	int* gpudata, *result;
	clock_t* time;
	cudaMalloc((void**)&gpudata, sizeof(int)* DATA_SIZE);
	cudaMalloc((void**)&result, sizeof(int)* THREAD_SIZE * BLOCK_SIZE);
	cudaMalloc((void**)&time, sizeof(int)* 2 * BLOCK_SIZE);
	cudaMemcpy(gpudata, data, sizeof(int)* DATA_SIZE, cudaMemcpyHostToDevice);
	sumOf << <BLOCK_SIZE, THREAD_SIZE, THREAD_SIZE * sizeof(int) >> >(gpudata, result, time);

	clock_t usetime[BLOCK_SIZE * 2];
	int sum[THREAD_SIZE * BLOCK_SIZE];
	cudaMemcpy(sum, result, sizeof(int)* THREAD_SIZE * BLOCK_SIZE, cudaMemcpyDeviceToHost);
	cudaMemcpy(usetime, time, sizeof(int)* 2 * BLOCK_SIZE, cudaMemcpyDeviceToHost);
	cudaFree(gpudata);
	cudaFree(result);
	cudaFree(time);
	int finalsum = 0;
	for (int i = 0; i < THREAD_SIZE * BLOCK_SIZE; ++i)
		finalsum += sum[i];
	clock_t minp, maxp;
	minp = usetime[0], maxp = usetime[BLOCK_SIZE];
	for (int i = 0; i < BLOCK_SIZE; ++i){
		if (usetime[i] < minp)  minp = usetime[i];
		if (usetime[i + BLOCK_SIZE] > maxp)  maxp = usetime[i + BLOCK_SIZE];
	}
	int val = 0;
	for (int i = 0; i < DATA_SIZE; ++i)
		val += data[i] * data[i];
	printf("%d\n", val);
	printf("%d %d\n", finalsum, maxp - minp);
	system("pause");
	return 0;
}