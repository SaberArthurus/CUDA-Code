#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <ctime>
#include <cuda_runtime.h>
using namespace std;
int main(int argc, char** argv){
	int count = 0;
	cudaGetDeviceCount(&count);
	if (count == 0) {
		cerr << "There is no device" << endl;
		system("pause");
		return 0;
	}
	int i; 
	for (int i = 0; i < count; ++i) {
		cudaDeviceProp prop;
		if (cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
			if (prop.major >= 1) {
				break;
			}
		}
	}
	if (i == count) {
		cerr << "no cida 1.x" << endl;
	}
	cudaSetDevice(i);
	system("pause");
	return 0;
}