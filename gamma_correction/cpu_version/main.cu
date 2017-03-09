#include "aux.h"
#include <iostream>
#include <cmath>
using namespace std;


int main(int argc, char** argv){
	cudaDeviceSynchronize(); CUDA_CHECK;
#ifdef CAMERA
#else
	string image = "";
	bool ret = getParam("i", image, argc, argv);
	if (argc <= 1) { cout << "Usage: " << argv[0] << " -i <image>  -g <gamma>[-repeats <repeats>] [-gray]" << endl; return 1; }
#endif
	int repeats = 1;
	getParam("repeats", repeats, argc, argv);
	cout << "repeats" << repeats << endl;

	bool gray = false;
	getParam("gray", gray, argc, argv);
	cout << "gray" << gray << endl;

	float gamma = 0;
	getParam("g", gamma, argc, argv);
	cout << "gamma" << gamma << endl;

#ifdef  CAMERA
	cv::ViedoCapture camera(0);
	if (!camera.isOpened()) {
		cerr << "ErrorL Could not open camera" << endl;
	}
	int camW = 640;
	int camH = 480;
	camera.set(CV_CAP_PROP_FRAME_WIDTH, camW);
	camera.set(CV_CAP_PROP_FRAME_HEIGHT, camH);
	cv::Mat mIn;
	camera >> min;
#else
	cv::Mat mIn = cv::imread(image.c_str(), (gray ? CV_LOAD_IMAGE_GRAYSCALE : -1));
	if (mIn.data == NULL) {
		cerr << "Error : Could not load image" << image << endl; return 1;
	}
#endif
	mIn.convertTo(mIn, CV_32F);
	mIn /= 255.f;
	int w = mIn.cols;
	int h = mIn.rows;
	int nc = mIn.channels();
	cout << "image: " << w << "x" << h << endl;
	cv::Mat mOut(h, w, mIn.type());

	float *imgIn = new float[(size_t) w * h * nc];
	float *imgOut = new float[(size_t)w * h * mOut.channels()];
#ifdef CAMER
	while (cv::waitKey(30) < 0) {
		camera >> mIn;
		mIn.convertTo(mIn, CV_32F);
		mIn /= 255.f;
#endif 
	convert_mat_to_layered (imgIn, mIn);
	Timer timer;
	timer.start();
	for (int c = 0; c < nc; ++ c){
		for (int x = 0; x < w; ++ x){
			for (int y = 0; y < h; ++ y) {
				imgOut[x + w * y + w * h * c] = \
					pow(imgIn[x + w * y + w * h * c], gamma);
			}
		}
	}
	timer.end(); float t = timer.get();
	cout << "time: " << t * 1000 << "ms" << endl;
	showImage("Input", mIn, 100, 100);
	convert_layered_to_mat(mOut, imgOut);
	showImage("Output", mOut, 100 + w + 40, 100);
#ifdef CAMERA
	}
#endif
	cv::waitKey(0);
	return 0;
}