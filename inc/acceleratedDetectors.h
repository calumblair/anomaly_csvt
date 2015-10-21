#ifndef CB_ACCELERATED_DETECTORS
#define CB_ACCELERATED_DETECTORS

#include "acceleratedAlgorithm.h"
//each individual detector goes here
///////////////////////////////////////////accelerated hog GGG (gpu)
class AcceleratedHOG_GGG : public AcceleratedAlgorithm
{
public:
	cv::AlgorithmInfo* info() const;
	void init(std::vector<float> SVMdetector);
	void detect(const cv::Mat frame, std::vector<Detection>& detections);
	void detect(HeisenFrame* hf, std::vector<Detection>& detections);
	AcceleratedHOG_GGG();


private:
	bool initialised;
	bool available;
	bool enabled;

	int nlevels;
	int requiredChannels;
	double scale;
	int gr_threshold;
	double hit_threshold;
	cv::gpu::HOGDescriptor gpu_hog;
	std::vector<cv::Rect> found;
	std::vector<float> scores;
};
bool initModule_accel_hog_ggg(void);


///////////////////////////////////////////accelerated hog GFG (fpga/gpu)
class AcceleratedHOG_GFG : public AcceleratedAlgorithm
{
public:
	cv::AlgorithmInfo* info() const;
	void init(std::vector<float> SVMdetector);
	void detect(const cv::Mat frame, std::vector<Detection>& detections);
	void detect(HeisenFrame* hf, std::vector<Detection>& detections);
	AcceleratedHOG_GFG();

private:
	bool initialised;
	bool available;
	bool enabled;

	int nlevels;
	int requiredChannels;
	double scale;
	int gr_threshold;
	double hit_threshold;
	cv::gpu::HOGDescriptor gpu_hog;
	fpgaHOGProcessor fpga_hog;
	std::vector<cv::Rect> found;
	std::vector<float> scores;
};
bool initModule_accel_hog_gfg(void);


///////////////////////////////////////////accelerated hog GFF (fpga/gpu)
class AcceleratedHOG_GFF : public AcceleratedAlgorithm
{
public:
	cv::AlgorithmInfo* info() const;
	void init(std::vector<float> SVMdetector);
	void detect(const cv::Mat frame, std::vector<Detection>& detections);
	void detect(HeisenFrame* hf, std::vector<Detection>& detections);
	AcceleratedHOG_GFF();

private:
	bool initialised;
	bool available;
	bool enabled;

	int nlevels;
	int requiredChannels;
	double scale;
	int gr_threshold;
	double hit_threshold;
	cv::gpu::HOGDescriptor gpu_hog;
	fpgaHOGProcessor fpga_hog;
	std::vector<cv::Rect> found;
	std::vector<float> scores;
};
bool initModule_accel_hog_gff(void);


///////////////////////////////////////////accelerated hog CFF (cpu/fpga)
class AcceleratedHOG_CFF : public AcceleratedAlgorithm
{
public:
	cv::AlgorithmInfo* info() const;
	void init(std::vector<float> SVMdetector);
	void detect(const cv::Mat frame, std::vector<Detection>& detections);
	void detect(HeisenFrame* hf, std::vector<Detection>& detections);
	AcceleratedHOG_CFF();

private:
	bool initialised;
	bool available;
	bool enabled;

	int nlevels;
	int requiredChannels;
	double scale;
	int gr_threshold;
	double hit_threshold;
	cv::HOGDescriptor cpu_hog;
	fpgaHOGProcessor fpga_hog;
	std::vector<cv::Rect> found;
	std::vector<float> scores;
};
bool initModule_accel_hog_cff(void);


///////////////////////////////////////////accelerated hog CFC (fpga/cpu)
class AcceleratedHOG_CFC : public AcceleratedAlgorithm
{
public:
	cv::AlgorithmInfo* info() const;
	void init(std::vector<float> SVMdetector);
	void detect(const cv::Mat frame, std::vector<Detection>& detections);
	void detect(HeisenFrame* hf, std::vector<Detection>& detections);
	AcceleratedHOG_CFC();

private:
	bool initialised;
	bool available;
	bool enabled;

	int nlevels;
	int requiredChannels;
	double scale;
	int gr_threshold;
	double hit_threshold;
	cv::HOGDescriptor cpu_hog;
	fpgaHOGProcessor fpga_hog;
	std::vector<cv::Rect> found;
	std::vector<float> scores;
};
bool initModule_accel_hog_cfc(void);


///////////////////////////////////////////not-really-accelerated HOG on CPU
//used for car detector
class AcceleratedHOG_Car_CCC : public AcceleratedAlgorithm
{
public:
	cv::AlgorithmInfo* info() const;
	void init(std::vector<float> SVMdetector);
	void detect(const cv::Mat frame, std::vector<Detection>& detections);
	void detect(HeisenFrame* hf, std::vector<Detection>& detections);
	AcceleratedHOG_Car_CCC();

private:
	bool initialised;
	bool available;
	bool enabled;

	int nlevels;
	int requiredChannels;
	double scale;
	int gr_threshold;
	double hit_threshold;
	cv::HOGDescriptor cpu_hog;
	std::vector<cv::Rect> found;
	std::vector<double> scores;
};
bool initModule_accel_hog_car_ccc(void);

///////////////////////////////////////////accelerated hog - gpu
//used for car detector 
class AcceleratedHOG_Car_GGG : public AcceleratedAlgorithm
{
public:
	cv::AlgorithmInfo* info() const;
	void init(std::vector<float> SVMdetector);
	void detect(const cv::Mat frame, std::vector<Detection>& detections);
	void detect(HeisenFrame* hf, std::vector<Detection>& detections);
	AcceleratedHOG_Car_GGG();

private:
	bool initialised;
	bool available;
	bool enabled;

	int nlevels;
	int requiredChannels;
	double scale;
	int gr_threshold;
	double hit_threshold;
	cv::gpu::HOGDescriptor gpu_hog;
	std::vector<cv::Rect> found;
	std::vector<float> scores;
};
bool initModule_accel_hog_car_ggg(void);


///////////////////////////////////////////accelerated car hog GFG (fpga/gpu)
class AcceleratedHOG_Car_GFG : public AcceleratedAlgorithm
{
public:
	cv::AlgorithmInfo* info() const;
	void init(std::vector<float> SVMdetector);
	void detect(const cv::Mat frame, std::vector<Detection>& detections);
	void detect(HeisenFrame* hf, std::vector<Detection>& detections);
	AcceleratedHOG_Car_GFG();

private:
	bool initialised;
	bool available;
	bool enabled;

	int nlevels;
	int requiredChannels;
	double scale;
	int gr_threshold;
	double hit_threshold;
	cv::gpu::HOGDescriptor gpu_hog;
	fpgaHOGProcessor fpga_hog;
	std::vector<cv::Rect> found;
	std::vector<float> scores;
};
bool initModule_accel_hog_car_gfg(void);


///////////////////////////////////////////accelerated hog CFC (fpga/cpu)
class AcceleratedHOG_Car_CFC : public AcceleratedAlgorithm
{
public:
	cv::AlgorithmInfo* info() const;
	void init(std::vector<float> SVMdetector);
	void detect(const cv::Mat frame, std::vector<Detection>& detections);
	void detect(HeisenFrame* hf, std::vector<Detection>& detections);
	AcceleratedHOG_Car_CFC();

private:
	bool initialised;
	bool available;
	bool enabled;

	int nlevels;
	int requiredChannels;
	double scale;
	int gr_threshold;
	double hit_threshold;
	cv::HOGDescriptor cpu_hog;
	fpgaHOGProcessor fpga_hog;
	std::vector<cv::Rect> found;
	std::vector<float> scores;
};
bool initModule_accel_hog_car_cfc(void);


///////////////////////////////////////////accelerated background subtraction - gpu
//based on Zikovic via Pham (for the GPU version)
#include "mogProcessor.h"
class AcceleratedMOG_GPU : public AcceleratedAlgorithm
{
public:
	cv::AlgorithmInfo* info() const;
	void init(cv::Mat frame);
	void detect(const cv::Mat frame, std::vector<Detection>& detections);
	void detect(HeisenFrame* hf, std::vector<Detection>& detections);
	AcceleratedMOG_GPU();
	double bbConfidence;
private:
	bool initialised;
	bool available;
	bool enabled;

	int requiredChannels;
	int area_threshold;//keep area if it's over an absolute min threshold
	float area_y_coeff; //or if it's even bigger and closer to the camera
	int gr_threshold;
	double hit_threshold;


	MOGProcessor_GPU mogProcessor;
	std::vector<cv::Rect> found;
	std::vector<double> scores;
	cv::Mat frame1, foreground;
	bool use_fast_lrate; //use if too many false pos in one frame. speed up learning rate for next frame

};
bool initModule_accel_mog_gpu(void);


///////////////////////////////////////////not-really-accelerated HOG on CPU
//used on laptop / something with only a slow GPU
class AcceleratedHOG_CCC : public AcceleratedAlgorithm
{
public:
	cv::AlgorithmInfo* info() const;
	void init(std::vector<float> SVMdetector);
	void detect(const cv::Mat frame, std::vector<Detection>& detections);
	void detect(HeisenFrame* hf, std::vector<Detection>& detections);
	AcceleratedHOG_CCC();

private:
	bool initialised;
	bool available;
	bool enabled;

	int nlevels;
	int requiredChannels;
	double scale;
	int gr_threshold;
	double hit_threshold;
	cv::HOGDescriptor cpu_hog;
	std::vector<cv::Rect> found;
	std::vector<double> scores;
};
bool initModule_accel_hog_ccc(void);


#endif
