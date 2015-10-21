#include "stdafx.h"
#include "mogProcessor.h"

using namespace cv;
using namespace std;
//this is a wrapper for an opencv 2.4.4 class which has support for gpu mog2 
#define BANK_ST 1
void MOGProcessor_GPU::getForeground(cv::gpu::GpuMat g_frame, cv::Mat& foreground, bool use_fast_lrate){
	float lrate = learningrate;
	if (iterations < 100 || use_fast_lrate){ //this will cause problems at intmax
		lrate = initialLearningRate;
		iterations++;
	}
	gpu_mog2(g_frame, g_mask, lrate, stream);
	//if nShadowdetection set to >0, will need to threshold here
#ifdef BANK_ST
	gpu::erode(g_mask, g_ping, k2, g_buf, anchor_centre, 1, stream);
	gpu::dilate(g_ping, g_mask, k1, g_buf, anchor_centre, 1, stream);
#else
	gpu::erode (g_mask,g_ping,k2,g_buf,anchor_centre,1,stream);
	stream.waitForCompletion();
	gpu::erode (g_ping,g_mask,Mat(),g_buf,anchor_centre,1,stream); //3x3 mat
	gpu::dilate(g_mask, g_ping, k1, g_buf,anchor_centre,1,stream);
	stream.waitForCompletion();
	gpu::dilate(g_ping, g_mask,k1, g_buf,anchor_centre,1,stream);

#endif
	//cant do 2 calls to same func in streaming mode, because of constants(see opencv docs)
	stream.enqueueDownload(g_mask, foreground);
	stream.waitForCompletion();
}


//overload for Mat on CPU
void MOGProcessor_GPU::getForeground(cv::Mat frame, cv::Mat& foreground, bool use_fast_lrate){
	assert(imSize == frame.size());
	stream.enqueueUpload(frame, g_frame);
	getForeground(g_frame, foreground, use_fast_lrate);
}


void MOGProcessor_GPU::init(cv::Mat frame){
	init(frame, 0.01f, 0.0001f);
}

void MOGProcessor_GPU::init(cv::Mat frame, float initialLearningRate_, float finalLearningRate){
	gpu_mog2 = cv::gpu::MOG2_GPU(); //construct (if not done already?)
	gpu_mog2.fTau = 0.4f; //shadow detection slightly wider than normal
	gpu_mog2.bShadowDetection = true; //do shadow detection but next line says dont draw anything
	gpu_mog2.nShadowDetection = 0;
	//set learning rate
	iterations = 0;
	learningrate = finalLearningRate;
	initialLearningRate = initialLearningRate_;

	//set up kernels for erode/dilate
	k1 = getStructuringElement(MORPH_ELLIPSE, Size(9, 7));
	k2 = getStructuringElement(MORPH_ELLIPSE, Size(3, 2));
	anchor_centre = Point(-1, -1);

	imSize = frame.size();//set imSize (not really necessary)
	//if we're passed a frame, may as well initialise with it
	gpu::GpuMat g_res;
	stream.enqueueUpload(frame, g_frame);
	gpu_mog2(g_frame, g_res, initialLearningRate, stream);
}
