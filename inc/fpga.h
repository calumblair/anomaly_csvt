/*fpgaHOGPpocessor class definition for use by wider HOG application*/
/*fpgaHOGProcessor is a wrapper for the fpgaInterface class itself */
/* CB 2/4/12 */

/*FPGA HOG Histogram processor exposed to application - does not rely on Windriver library to compile*/
#ifndef CB_FPGA_HOG_PROCESSOR
#define CB_FPGA_HOG_PROCESSOR 1
#include "fpgaImageSize.h"

//execution paths
#ifndef FPGA_DEFS
#define FPGA_DEFS
#define PROC_CCC 0
#define PROC_GGG 1
#define PROC_GFG 2
#define PROC_CFC 3
#define PROC_GFG_SINGLE_PASS 4
#define PROC_CFF 5
#define PROC_GFF 6
#define PROC_GGG_RBF 7
#define PROC_GFG_RBF 8
#endif



#define FPGA_CAR_HOG_VBLOCKS ( 6)
#define FPGA_PED_HOG_VBLOCKS (15)

//thread-safe exception if fpga not found
class fpga_not_found_error : public std::runtime_error
{
public:
	fpga_not_found_error(const std::string& msg) :std::runtime_error(msg){};
};

class fpgaHOGProcessor
{
public:
	//can return hists either uploaded to the GPU:
	int getCells(const cv::Mat& img, cv::gpu::GpuMat* cell_hists_g, int opType = FPGA_USE_PED_HOG | FPGA_HOG_GET_HISTS);
	//or return a Mat which has a zero-copy pointer to the floats in the hist buffer
	int getCells(const cv::Mat& img, cv::Mat* cell_hists, int opType = FPGA_USE_PED_HOG | FPGA_HOG_GET_HISTS);

	//GPU versions: GFF and GFG, controlled with getScores flag
	//Version: takes a pointer to a gpu hogdescriptor
	//for post-histogram processing
	//returns fpga status (0 if OK, !0 if unreliable)
	int detectMultiScale(const cv::Mat& img, std::vector<cv::Rect>& found_locations,
		std::vector<float>& scores, cv::gpu::HOGDescriptor*, double hit_threshold = 0,
		double scale0 = 1, int group_threshold = 0, bool getScores = false);

	//CPU Versions: only with foundWeights
	//Takes a pointer to a cpu hogdescriptor
	//returns fpga status (0 if OK, !0 if unreliable
	//this covers CFC and CFF - switch the getScores flag to use
	int detectMultiScale(const cv::Mat& img, std::vector<cv::Rect>& foundLocations,
		std::vector<float>& foundWeights, const cv::HOGDescriptor* cpu_hog,
		double hitThreshold = 0, double scale0 = 1,
		double finalThreshold = 0, bool useMeanshiftGrouping = false, bool getScores = false);
	int detectMultiScaleMultiThreaded(
		const cv::Mat& img, std::vector<cv::Rect>& foundLocations, std::vector<float>& foundWeights,
		const cv::HOGDescriptor* cpu_hog, double hitThreshold, double scale0,
		double finalThreshold, bool useMeanshiftGrouping = false, bool getScores = false);

	std::vector<float> getPeopleDetector64x128(void);
	std::vector<float> getFPGACarDetector104x56(void);
	std::vector<float> getGPUCarDetector104x56(void);

	fpgaHOGProcessor();
	fpgaHOGProcessor(int object_to_detect);
	~fpgaHOGProcessor();
	int nlevels; //number of scales to scan over: big effect on performance
	//the folling three shouldnt really be public but are needed by fpgaHOGInvoker
	cv::Size win_stride;
	cv::Size win_size;
	void thresholdScores(cv::Mat& cell_hists, const cv::Size imsz, std::vector<cv::Point>& foundLocations,
		std::vector<float> &foundWeights, double hitThreshold, int lp = 7, int rp = 7, int tp = 0, int bp = 0);

	int objectType;

protected:
	cv::Size padding;
	bool hogResize(cv::gpu::GpuMat& img_g, cv::gpu::GpuMat& dst, cv::Size smaller_size);
	void groupScores(std::vector<cv::Rect>& locations, std::vector<float>& weights,
		std::vector<double>& scales, double group_threshold, bool useMeanshiftGrouping = false);
private:
	int hogcore(const cv::Mat& img, float** hists_buf_ptr, int opType = FPGA_USE_PED_HOG | FPGA_HOG_GET_HISTS);

	int cell_hist_sz; //this is filled by fpgaInterface

	bool useFPGA;
	//use this to store preloaded detections for a single
	//frame while running with no FPGA attached, or on the laptop
	//gets freed in destructor
	float* dummy_fpga_results_ptr;

	//some stuff for dealing with the laptop when the FPGA is not connected
	std::string laptopName;
};


//strcmp function for different platforms
#ifdef LINUX
#define STRING_COMPARE strcasecmp
#else
#define STRING_COMPARE _stricmp
#endif
#endif
