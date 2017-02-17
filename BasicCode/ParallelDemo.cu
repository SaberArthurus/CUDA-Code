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
	const int tid = threadIdx.x;
	const int bid = blockIdx.x;
	int sum = 0;
	clock_t start;
	if (tid == 0)  time[bid] = clock();
	for (int i = bid * THREAD_SIZE + tid; i < DATA_SIZE; i += BLOCK_SIZE * THREAD_SIZE) {
		sum += data[i];
	}
	result[bid * THREAD_SIZE + tid] = sum;
	if (tid == 0) time[bid + BLOCK_SIZE] = clock();
}

int main(int argc, char** argv) {
	Generate(data, DATA_SIZE);
	int* gpudata, *result;
	clock_t* time;
	cudaMalloc((void**)&gpudata, sizeof(int)* DATA_SIZE);
	cudaMalloc((void**)&result, sizeof(int)* THREAD_SIZE * BLOCK_SIZE);
	cudaMalloc((void**)&time, sizeof(int) * 2 * BLOCK_SIZE);
	cudaMemcpy(gpudata, data, sizeof(int)* DATA_SIZE, cudaMemcpyHostToDevice);
	sumOf<<<BLOCK_SIZE, THREAD_SIZE, 0 >>>(gpudata, result, time);

	clock_t usetime[BLOCK_SIZE * 2];
	int sum[THREAD_SIZE * BLOCK_SIZE];
	cudaMemcpy(sum, result, sizeof(int)* THREAD_SIZE * BLOCK_SIZE, cudaMemcpyDeviceToHost);
	cudaMemcpy(usetime, time, sizeof(int) * 2 * BLOCK_SIZE, cudaMemcpyDeviceToHost);
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

	printf("%d %d\n", finalsum, maxp - minp);
	system("pause");
	return 0;
}