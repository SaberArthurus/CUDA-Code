#include <cuda_runtime.h>
#include <iostream>
#include <ctime>
using namespace std;
const int N = 1024;
const int THREAD_SIZE = 256;
#define CUDA_CHECK cuda_check(__FILE__, __LINE__)


void cuda_check(string file, int line){
	cudaError_t e = cudaGetLastError();
	if (e != cudaSuccess){
		cout << endl << file << ", line" << line <<  ": " << cudaGetErrorString(e) << "(" << e << ")" << endl;
		exit(1);
	}
}
__device__ float add(float a, float b){
    return a + b;
}

__global__ void add_arrays(float* a, float* b, float* c, int n){
    int ind = threadIdx.x + blockDim.x * blockIdx.x;
    if (ind < n) {
        c[ind] = add(a[ind], b[ind]);
    }
}


int main(int argc, char** argv){
    float *a = new float[N];
    float *b = new float[N];
    float *c = new float[N];
    for (int i = 0; i < N; ++ i){
        a[i] = rand();
        b[i] = rand();
        c[i] = 0;
    }
    for (int i = 0; i < N; ++ i) c[i] = a[i] + b[i];

    cout << "CPU:" << endl;
    for (int i = 0; i < N; ++ i)
        cout << a[i] << "+" << b[i] << "=" << c[i] << endl;
    cout << endl;
    float *d_a = NULL;
    float *d_b = NULL;
    float *d_c = NULL;
    
    size_t nbytes = N * sizeof(float);
    cudaMalloc(&d_a, nbytes); CUDA_CHECK;
    cudaMalloc(&d_b, nbytes); CUDA_CHECK;
    cudaMalloc(&d_c, nbytes); CUDA_CHECK;
    cudaMemcpy(d_a, a, nbytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, nbytes, cudaMemcpyHostToDevice);
    dim3 block = dim3(THREAD_SIZE, 1, 1);
    dim3 grid = dim3((N + block.x - 1) / block.x, 1, 1);
    add_arrays <<<grid, block>>> (d_a, d_b, d_c, N);
    cudaMemcpy(c, d_c, nbytes, cudaMemcpyDeviceToHost);CUDA_CHECK;
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cout << "GPU" << endl;
    for (int i = 0; i < N; ++ i)
        cout << i << ":" << a[i] << "+"  << b[i] << "=" << c[i] << endl;
    delete[] a;
    delete[] b;
    delete[] c;
    return 0;
}