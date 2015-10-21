//quick class for old gaussian mixture model (mixture of gaussians) background subtraction functions
//this was before they were part of openCV

#ifndef CB_MOG_PROCESSOR_GPU
#define CB_MOG_PROCESSOR_GPU 1
class MOGProcessor_GPU{
public:
	void getForeground(const cv::Mat frame, cv::Mat& foreground, bool use_fast_lrate=false);
	void getForeground(const cv::gpu::GpuMat g_frame, cv::Mat& foreground,
	bool use_fast_lrate=false); //overload if source already on GPU
	void init(cv::Mat frame);
	void init(cv::Mat frame, float initialLearningRate, float finalLearningRate);
	float learningrate;
private:
	cv::Size imSize;


	//extended to use the class built in to opencv
	int iterations;
	float initialLearningRate;
	cv::Mat k1, k2;  //2 mats to hold structuring elements
	cv::gpu::MOG2_GPU gpu_mog2;
	cv::gpu::GpuMat g_frame, g_mask, g_ping, g_buf;
	cv::gpu::Stream stream;
	cv::Point anchor_centre;
};
#endif