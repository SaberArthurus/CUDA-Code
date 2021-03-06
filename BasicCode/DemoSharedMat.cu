#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <iostream>
#define THREAD_SIZE 256
using namespace std;


void matgen(float* a, int lda, int n) {
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; ++j) {
			a[i * lda + j] = (float)rand() / RAND_MAX +
				(float)rand() / (RAND_MAX * RAND_MAX);
		}
		//printf("%.2f\n", a[i]);
	}
}

void matmult(const float* a, int lda, const float* b, int ldb, float* c, int ldc, int n) {
	int i, j, k;
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; ++j) {
			double t = 0;
			for (int k = 0; k < n; ++k) {
				t += a[i * lda + k] * b[k * ldb + j];
			}
			c[i * ldc + j] = t;
		}
	}
}

void compare_mat(const float* a, int lda, const float* b, int ldb, int n) {
	float max_err = 0;
	float average_err = 0;
	for (int i = 0; i < n; ++i){
		for (int j = 0; j < n; ++j) {
			if (b[i * ldb + j] != 0) {
				float err = fabs((a[i * lda + j] - b[i * ldb + j]) / b[i * ldb + j]);
				//printf("%.2f\n", max_err);
				if (max_err < err) max_err = err;
				average_err += err;
			}
		}
	}
	printf("max_err : %f, average_err: %.2f\n", max_err, average_err);
}


__global__ static void matMultCUDA(const float* a, size_t lda,
	const float* b, size_t ldb, float* c, size_t ldc, int n) {
	extern __shared__ float data[];
	const int tid = threadIdx.x;
	const int row = blockIdx.x;
	int i, j;
	for (i = tid; i < n; i += blockDim.x) {
		data[i] = a[row * lda + i];
	}
	__syncthreads();
	for (j = tid; j < n; j += blockDim.x) {
		float t = 0;
		float y = 0;
		for (i = 0; i < n; ++i) {
			float r;
			y -= data[i] * b[i * ldb + j];
			r = t - y;
			y = (r - t) + y;
			t = r;
		}
		c[row * ldc + j] = t;
	}
}

clock_t matmultCUDA(const float* a, int lda,
	const float* b, int ldb, float*c, int ldc, int n) {
	float *ac, *bc, *cc;
	clock_t start, end;
	start = clock();
	size_t pitch_a, pitch_b, pitch_c;
	cudaMallocPitch((void**)&ac, &pitch_a, sizeof(float)* n, n);
	cudaMallocPitch((void**)&bc, &pitch_b, sizeof(float)* n, n);
	cudaMallocPitch((void**)&cc, &pitch_c, sizeof(float)* n, n);
	cudaMemcpy2D(ac, pitch_a, a, sizeof(float)* lda, sizeof(float)* n, n, cudaMemcpyHostToDevice);
	cudaMemcpy2D(bc, pitch_b, b, sizeof(float)* ldb, sizeof(float)* n, n, cudaMemcpyHostToDevice);

	int blocks = (n + THREAD_SIZE - 1) / THREAD_SIZE;
	matMultCUDA<<<n, THREAD_SIZE, sizeof(float) * n >>>
		(ac, pitch_a / sizeof(float), bc, pitch_b / sizeof(float), cc, pitch_c / sizeof(float), n);
	cudaMemcpy2D(c, sizeof(float)* n, cc, pitch_c, sizeof(float) * n, n, cudaMemcpyDeviceToHost);
	cudaFree(ac);
	cudaFree(bc);
	cudaFree(cc);
	end = clock();
	return end - start;

}
int main(int argc, char** argv) {
	float *a, *b, *c, *d;
	const int n = 1000;
	a = (float*)malloc(sizeof(float)* n * n);
	b = (float*)malloc(sizeof(float)* n * n);
	c = (float*)malloc(sizeof(float)* n * n);
	d = (float*)malloc(sizeof(float)* n * n);
	srand(10);
	matgen(a, n, n);
	matgen(b, n, n);
	clock_t time = matmultCUDA(a, n, b, n, c, n, n);

	matmult(a, n, b, n, d, n, n);
	compare_mat(c, n, d, n, n);

	double sec = (double)time / CLOCKS_PER_SEC;
	printf("Time used: %.2f  (%.2lf GFLOATS)\n", sec, 2.0 * n * n * n / (sec * 1e9));
	system("pause");
	return 0;
}