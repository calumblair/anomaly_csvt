//Standard header file for all opencv stuff
//C Blair

#ifndef CB_GPU_OPENCV
#define CB_GPU_OPENCV
//Standard Libraries
#include <iostream>
#include <string>
#include <sstream>
#include <iomanip>
#include <stdexcept>
#endif


//cuda stuff 
#include <cuda.h>

#ifndef CB_CUDA
#include <opencv2/gpu/gpu.hpp>
#include <helper_cuda.h>
#include <helper_timer.h>
#include <cuda_runtime.h>		
#include <device_launch_parameters.h>
#endif

//OpenCV libraries
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/imgproc/imgproc.hpp>


//General IO
/*void myimread(const  char* filename, int size, void* in);
void myimwrite( char* filename, int size, void* in);

void median_filter_gpu(const cv::gpu::DevMem2D& src, cv::gpu::DevMem2D& dst);
*/