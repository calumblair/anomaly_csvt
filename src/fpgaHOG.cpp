#include "stdafx.h"

//if profiling
#include <cuda_profiler_api.h>
#include <nvToolsExt.h>
#include "profiling.h"

#ifndef USE_WINDRIVER_FPGA_INTERFACE
#include "fpga.h"
using namespace std;
using namespace cv;
int fpgaHOGProcessor::getCells(const cv::Mat& , cv::gpu::GpuMat* , int){return 0;}
int fpgaHOGProcessor::getCells(const cv::Mat&, cv::Mat* , int ){return 0;}
int fpgaHOGProcessor::detectMultiScale(const cv::Mat& , std::vector<cv::Rect>&,
		std::vector<float>& , cv::gpu::HOGDescriptor*, double,
		double , int , bool ){return 0;}
int fpgaHOGProcessor::detectMultiScale(const cv::Mat& , std::vector<cv::Rect>&,
		std::vector<float>& , const cv::HOGDescriptor* ,
		double , double ,double , bool , bool ){return 0;}
int fpgaHOGProcessor::detectMultiScaleMultiThreaded(
		const cv::Mat& , std::vector<cv::Rect>& , std::vector<float>& ,
		const cv::HOGDescriptor* , double , double ,
		double , bool , bool ){return 0;}

//std::vector<float> fpgaHOGProcessor::getPeopleDetector64x128(void){return (std::vector<float>(0.f,1));}
//std::vector<float> fpgaHOGProcessor::getFPGACarDetector104x56(void){return (std::vector<float>(0.f,1));}
//std::vector<float> fpgaHOGProcessor::getGPUCarDetector104x56(void){return (std::vector<float>(0.f,1));}

fpgaHOGProcessor::fpgaHOGProcessor(){;}
fpgaHOGProcessor::fpgaHOGProcessor(int ){;}
fpgaHOGProcessor::~fpgaHOGProcessor(){;}
void fpgaHOGProcessor::thresholdScores(cv::Mat& , const cv::Size , std::vector<cv::Point>& ,
		std::vector<float> &, double , int , int , int , int ){;}


#else

//end if profiling
#include "wdc_lib.h"
//will need to link against bmd_lib.c/.h
#include "fpga.h"

//note that fpgaHOGProcessor relies on a modified version of the objdetect HOGDescriptor and gpu::HOGDescriptor objects. 
//I looked at trying to extract as much of this as possible from inside OpenCV and putting it into a separate class, but 
//changing the contents of modules/gpu/src/cuda/hog.cu was next to impossible without modification.
//As things stand, the cpu version should be easy enough to access/modify through inheritance, but the GPU version is much more of a problem.
//I dont propose to revisit this unless bits of OpencV get changed dramatically.
//this is still true for OpenCV 2.4.4 although I am using computeConfidence now instead of detect.

using namespace std;
using namespace cv;
//Constructor : disables FPGA if we are on the wrong machine
//note now also sets win_stride and padding
fpgaHOGProcessor::fpgaHOGProcessor()
{
	nlevels = cv::HOGDescriptor::DEFAULT_NLEVELS;
	win_size = cv::Size(64, 128);
	win_stride = cv::Size(8, 8);
	padding = cv::Size(0, 0);
	objectType = FPGA_USE_PED_HOG;
	//here we can switch on and off FPGA dependency
	//either manually or based on hostname
	//without depending on the windriver interface
	useFPGA = false;

	//work out hostname
#define HOSTNAMELEN 50
	DWORD hostNameLen = 50;
#ifdef WIN32
	TCHAR thisHost[HOSTNAMELEN];
	if (GetComputerName(thisHost, &hostNameLen)){ //if no errors - returns nonzero if succeeded
#else
	char thisHost[HOSTNAMELEN];
	// non-windows/cross-platform version:
	if (!gethostname(thisHost, hostNameLen)){ //returns 0 if success
#endif
		useFPGA = !STRING_COMPARE(thisHost, "CUDABOX");
	}
	dummy_fpga_results_ptr = NULL;
}

fpgaHOGProcessor::fpgaHOGProcessor(int hogType_)
{
	objectType = hogType_;
	nlevels = cv::HOGDescriptor::DEFAULT_NLEVELS;
	switch (objectType){
	case FPGA_USE_PED_HOG:
		win_size = cv::Size(64, 128);
		break;
	case FPGA_USE_CAR_HOG:
		win_size = cv::Size(104, 56);
		break;
	default:
		throw("unknown object type for fpgaHOG");
	}
	win_stride = cv::Size(8, 8);
	padding = cv::Size(0, 0);
	//here we can switch on and off FPGA dependency
	//either manually or based on hostname
	//without depending on the windriver interface
	useFPGA = false;

	//work out hostname
	//TODO make this cross-platform
#define HOSTNAMELEN 50
	DWORD hostNameLen = 50;
#ifdef WIN32
	TCHAR thisHost[HOSTNAMELEN];
	if (GetComputerName(thisHost, &hostNameLen)){ //if no errors - returns nonzero if succeeded
#else
	char thisHost[HOSTNAMELEN];
	// non-windows/cross-platform version:
	if (!gethostname(thisHost, hostNameLen)){ //returns 0 if success
#endif
		useFPGA = !STRING_COMPARE(thisHost, "CUDABOX");
	}
	dummy_fpga_results_ptr = NULL;
}

fpgaHOGProcessor::~fpgaHOGProcessor()
{
	if (!useFPGA && dummy_fpga_results_ptr)
		free(dummy_fpga_results_ptr);
}

//getCells function exposed from fpga hog processor is really a call to the fpga interface class
//which is built with the Jungo libraries.
//single-copy GpuMat version
int fpgaHOGProcessor::getCells(const Mat& img, gpu::GpuMat* cell_hists_g, int opType)
{
	int status;
	float* cellHistsfp = NULL;
	status = hogcore(img, &cellHistsfp, opType);
	if (!cellHistsfp)
		cout << "cellHistsfp not filled properly" << endl;
	//make a temp mat header
	Mat cellHists(1, cell_hist_sz, CV_32FC1, cellHistsfp);
	//define with single row and many cols so gpuMat will be continuous
	// img_width in cells * img_height in cells * nbins * 4(as 32bit float data)
	cell_hists_g->upload(cellHists);
	return status;
}

//zero copy Mat version
int fpgaHOGProcessor::getCells(const Mat& img, Mat* cell_hists, int opType)
{
	int status;
	float* cellHistsfp = NULL;
	status = hogcore(img, &cellHistsfp, opType);
	if (!cellHistsfp)
		cout << "cellHistsfp not filled properly" << endl;
	//make a temp mat header
	Mat cellHistsMat(1, cell_hist_sz, CV_32FC1, cellHistsfp);
	*cell_hists = cellHistsMat;
	return status;
}



//this thresholds the scores returned from the fpga-hog detector and locates them within the
//image
//the advantage of doing this here is that i dont have to change the internals of openCV.
//drawback: we have lots more magic numbers
//this is also easier since we dont have to do scaling
//the scores are received in a matrix from the fpga in the format
// jjjjjjjssss....ssssjjj scores surrounded by junk.
// jjjjjjjssss....ssssjjj remove the junk by padding the matrix with lp,tp, rp, bp
// jjjjjjjjjjjjjjjjjjjjjj
void fpgaHOGProcessor::thresholdScores(Mat& cell_hists, const Size imsz, vector<Point>& locations,
	vector<float> &weights, double hitThreshold, int lp, int rp, int tp, int bp)
{
	int scoresPerRow = (imsz.width - 2) / 8;
	float* baseptr = cell_hists.ptr<float>(0), *rowptr;
	for (int ih = (0 + tp); ih < ((imsz.height - 2) / 8 - bp); ih++) {
		rowptr = baseptr + ih*scoresPerRow;
		int iw = (0 + lp);
		for (iw; iw < ((imsz.width - 2) / 8 - rp - 4); iw += 4) {
			if (rowptr[iw] >(float)hitThreshold){
				locations.push_back(Point((iw - lp) * 8, (ih - tp) * 8));
				weights.push_back(rowptr[iw]);
			}
			if (rowptr[iw + 1] > (float)hitThreshold){
				locations.push_back(Point((iw + 1 - lp) * 8, (ih - tp) * 8));
				weights.push_back(rowptr[iw + 1]);
			}
			if (rowptr[iw + 2] > (float)hitThreshold){
				locations.push_back(Point((iw + 2 - lp) * 8, (ih - tp) * 8));
				weights.push_back(rowptr[iw + 2]);
			}
			if (rowptr[iw + 3] > (float)hitThreshold){
				locations.push_back(Point((iw + 3 - lp) * 8, (ih - tp) * 8));
				weights.push_back(rowptr[iw + 3]);
			}
		}
		for (iw; iw < ((imsz.width - 2) / 8 - rp); iw++) {
			if (rowptr[iw] >(float)hitThreshold){
				locations.push_back(Point((iw - lp) * 8, (ih - tp) * 8));
				weights.push_back(rowptr[iw]);
			}
		}
	}
}


//pad and resize image for GPU - this gets called in a couple places
//returns 0 if source image is unchanged, 1 if changed
inline bool fpgaHOGProcessor::hogResize(gpu::GpuMat& img_g, gpu::GpuMat& dst, Size smaller_size){
	if (smaller_size == img_g.size() || smaller_size == Size(0, 0)) {
		return false; //no resize needed
	}
	else {
		gpu::GpuMat smaller_img_g;
		//Resize image to appropriate scale then pad back out
		//to the (hardcoded) size accepted by fpga

		resize(img_g, smaller_img_g, smaller_size, INTER_LINEAR);
		Size small_size = smaller_img_g.size();
		if (small_size != Size(FPGA_COLS, FPGA_ROWS)) {
			int top = 0,
				left = 0,
				right = FPGA_COLS - small_size.width,
				/* pad to frame size  */
				//bottom	= FPGA_ROWS - small_size.height;
				/* pad to sensible size */
				bottom = MIN(FPGA_ROWS - small_size.height,
				((small_size.height / 8 + 1) * 8 + 2) - small_size.height);
			copyMakeBorder(smaller_img_g, dst,
				top, bottom, left, right, BORDER_CONSTANT, 0);
		}
		else
			smaller_img_g.copyTo(dst);
		return true;
	}
}

void fpgaHOGProcessor::groupScores(vector<Rect>& locations,
	vector<float>& weights, vector<double>& scales,
	double group_threshold, bool useMeanshiftGrouping)
{
	if (useMeanshiftGrouping)
	{
		//convert to doubles again...
		vector<double> weights_d;
		weights_d.assign(weights.begin(), weights.end());
		groupRectangles_meanshift(locations, weights_d, scales, group_threshold, win_size);
		weights.assign(weights_d.begin(), weights_d.end());
	}
	else
	{
		vector<int> weights_int;
		weights_int.assign(weights.begin(), weights.end());
		cv::groupRectangles(locations, weights_int, (int)group_threshold,
			0.2/*magic number copied from CPU version*/);
		if (group_threshold != 0) //return grouped stuff
			weights.assign(weights_int.begin(), weights_int.end());
	}
}

//the actual HOG processors - this is the gfg and gff version
int fpgaHOGProcessor::detectMultiScale(
	const Mat& img, vector<Rect>& found_locations, vector<float>& weights,
	cv::gpu::HOGDescriptor* gpu_hog, double hit_threshold, double scale0, int group_threshold,
	bool getScores)
{
	double scale = 1.;
	int levels = 0;
	vector<double> level_scale;
	int fpgaStatus = 0;

	CV_Assert(img.type() == CV_8UC1 || img.type() == CV_8UC4);
	CV_Assert(img.rows > win_size.height);
	for (levels = 0; levels < nlevels; levels++) {
		level_scale.push_back(scale);
		if (cvRound(img.cols / scale) < win_size.width ||
			cvRound(img.rows / scale) < win_size.height || scale0 <= 1)
			break;
		scale *= scale0;
	}
	levels = max(levels, 1);
	level_scale.resize(levels);

	std::vector<Rect> all_candidates;
	vector<float> all_weights;
	vector<double> all_scales;

	Mat small_padded_img;
	Mat cell_hists;
	//resize is easily the slowest part of this function so do it on the GPU
	static gpu::GpuMat img_g;
	if (levels > 1)
		img_g.upload(img);
	static gpu::GpuMat small_padded_img_g;
	Size small_size;

	static gpu::GpuMat cell_hists_g;

	for (size_t i = 0; i < level_scale.size(); i++) {
		vector<Point> locations;
		vector<float> level_weights;
		scale = level_scale[i];
		Size sz(cvRound(img.cols / scale), cvRound(img.rows / scale));
		/*resize*/
		profilerMarkA("gff/gfg:starting resize");
		if (scale != 1.){
			if (hogResize(img_g, small_padded_img_g, sz))//if returns 1 then resize needed
				small_padded_img_g.download(small_padded_img);
		}
		else{ //edge case, levels==1 && size != FPGA_ROWS, FPGA_COLS
			if (img.size() == Size(FPGA_COLS, FPGA_ROWS))
				small_padded_img = img;
			else if (img.size() == Size(FPGA_COLS - 2, FPGA_ROWS - 2))
				copyMakeBorder(img, small_padded_img,
				1, 1, 1, 1, BORDER_CONSTANT, 0);
			else{
				int top = 0,
					left = 0,
					right = FPGA_COLS - img.cols,
					/*edge case so no option to change size here*/
					bottom = MIN(FPGA_ROWS - img.rows, ((img.rows / 8 + 1) * 8 + 2) - img.rows);

				copyMakeBorder(img, small_padded_img,
					top, bottom, left, right, BORDER_CONSTANT, 0);
			}
		}
		profilerMarkA("gff/gfg:finished resize");
		/*get histograms and classify*/
#ifdef FPGA_IMAGE_IS_PAL
		int nstripes = 12;
#else
		int nstripes=16;
#endif
		if (getScores){
			profilerMarkA("gff:starting getScores call");
			fpgaStatus = getCells(small_padded_img, &cell_hists, objectType | (getScores ? FPGA_HOG_GET_SCORES : FPGA_HOG_GET_HISTS));
			profilerMarkA("gff:finished getScores call");
			/*threshold and group scores*/
			thresholdScores(cell_hists, Size(64 * nstripes + 2, small_padded_img.rows),
				locations, level_weights, hit_threshold, 7, 0, 0, /*15*/
				objectType == FPGA_USE_CAR_HOG ? FPGA_CAR_HOG_VBLOCKS : FPGA_PED_HOG_VBLOCKS);
		}
		else {
			/*get histograms*/
			fpgaStatus = getCells(small_padded_img, &cell_hists_g, objectType | (getScores ? FPGA_HOG_GET_SCORES : FPGA_HOG_GET_HISTS));
			//always get cells not scores

			/*classify*/
			profilerMarkA("gfg:processing on gpu");
			gpu_hog->computeConfidenceFromCells(cell_hists_g, locations, hit_threshold, win_stride,
				padding, small_padded_img.rows - 2, 64 * nstripes + 2 - 2, level_weights);
			profilerMarkA("gfg:got scores from gpu");
		}

		Size scaled_win_size(cvRound(win_size.width * scale), cvRound(win_size.height * scale));
		for (size_t j = 0; j < locations.size(); j++) {
			//sanity check for detections: as long as scaled top corner is inside the original image
			Point pt = locations[j] * scale;
			if (pt.x <= img.cols && pt.y <= img.rows){
				all_candidates.push_back(Rect(pt, scaled_win_size));
				all_scales.push_back(scale);
				all_weights.push_back(level_weights[j]);
			}
		}
	}

	groupScores(all_candidates, all_weights, all_scales, group_threshold);
	found_locations.assign(all_candidates.begin(), all_candidates.end());
	weights.assign(all_weights.begin(), all_weights.end());

	profilerMarkA("gff/gfg:done");
	return fpgaStatus;
}

//CFC and CFF versions - select using getScores
int fpgaHOGProcessor::detectMultiScale(
	const Mat& img, vector<cv::Rect>& foundLocations, vector<float>& foundWeights,
	const cv::HOGDescriptor* cpu_hog, double hitThreshold, double scale0,
	double finalThreshold, bool useMeanshiftGrouping, bool getScores)
{
	double scale = 1.;
	int levels = 0;
	vector<double> levelScale;
	int fpgaStatus = 0;

	CV_Assert(img.type() == CV_8UC1 || img.type() == CV_8UC4);
	for (levels = 0; levels < nlevels; levels++) {
		levelScale.push_back(scale);
		if (cvRound(img.cols / scale) < win_size.width ||
			cvRound(img.rows / scale) < win_size.height ||
			scale0 <= 1)
			break;
		scale *= scale0;
	}
	levels = max(levels, 1);
	levelScale.resize(levels);

	//unrolled tbb paralleliser for working on different threads/cores. no more
	//than 1 thing gets to access the fpga at once
	//this is OK because we are parallelising by using an accelerator rather than using multiple cores/threads

	vector<Rect> allCandidates;
	vector<double> scales; //same for scales
	vector<float> weights; //must clear these for each separate scale?
	vector<double> foundScales;

	double minScale = levelScale[0] > 0 ? levelScale[0] : levelScale.back() > 1 ? levelScale[1] : max(img.cols, img.rows);
	Size maxSz(cvCeil(img.cols / minScale), cvCeil(img.rows / minScale));

	Mat smaller_img_buf(maxSz, img.type()), padded_img_buf(maxSz, img.type());

	//reinitialise this in here
	Mat cell_hists;

	profilerMarkA("cfc/cff:starting resize");
	for (size_t i = 0; i < levelScale.size(); i++) {
		vector<Point> locations;
		vector<float> hitsWeights;
		double scale = levelScale[i];

		Size sz(cvRound(img.cols / scale), cvRound(img.rows / scale));
		Mat smaller_img(sz, img.type(), smaller_img_buf.data);
		Mat small_padded_img(sz, img.type(), padded_img_buf.data);
		Size small_size;

		profilerMarkA("cfc/cff:starting resize");
		/*resize*/
		if (scale != 1.) { //usual resizing operations
			resize(img, smaller_img, sz, INTER_LINEAR);
			small_size = smaller_img.size();
			if (small_size != Size(FPGA_COLS, FPGA_ROWS)) {
				int top = 0,
					left = 0,
					/* leave the next 2 lines in to pad to frame size each time */
					right = FPGA_COLS - small_size.width,
					//bottom	= FPGA_ROWS - small_size.height;
					/*the alternative is to pad to nearest group of 8 pixels high,
					so as not to send redundant data, and let the existing interface code
					deal with sending enough extra pixels to flush the gradient & classifier pipelines
					*/
					bottom = MIN(FPGA_ROWS - small_size.height,
					((small_size.height / 8 + 1) * 8 + 2) - small_size.height);
				copyMakeBorder(smaller_img, small_padded_img,
					top, bottom, left, right, BORDER_CONSTANT, 0);
			}
			else
				small_padded_img = smaller_img;
		}
		else{ //edge case, levels==1 && size != FPGA_ROWS, FPGA_COLS
			if (img.size() == Size(FPGA_COLS, FPGA_ROWS))
				small_padded_img = img;
			else if (img.size() == Size(FPGA_COLS - 2, FPGA_ROWS - 2)){
				copyMakeBorder(img, small_padded_img,
					1, 1, 1, 1, BORDER_CONSTANT, 0);
			}
			else{
				int top = 0,
					left = 0,
					right = FPGA_COLS - img.cols,
					/* pad to frame size -use this one for 1024x768 */
					//bottom	= FPGA_ROWS - img.rows;
					/* pad to sensible size */
					bottom = MIN(FPGA_ROWS - img.rows, ((img.rows / 8 + 1) * 8 + 2) - img.rows);
				copyMakeBorder(img, small_padded_img,
					top, bottom, left, right, BORDER_CONSTANT, 0);
			}
		}

		//////////////////////////////////////////////////////////////////////////
#ifdef FPGA_IMAGE_IS_PAL
		int nstripes = 12;
#else
		int nstripes=16;
#endif
		if (getScores){
			profilerMarkA("cfc/cff:starting getScores call");
			fpgaStatus = getCells(small_padded_img, &cell_hists, objectType | (getScores ? FPGA_HOG_GET_SCORES : FPGA_HOG_GET_HISTS));
			profilerMarkA("cfc/cff:finished getScores call");
			thresholdScores(cell_hists, Size(64 * nstripes + 2, small_padded_img.rows),
				locations, hitsWeights, hitThreshold, 7, 0, 0, /*15*/
				objectType == FPGA_USE_CAR_HOG ? FPGA_CAR_HOG_VBLOCKS : FPGA_PED_HOG_VBLOCKS);
		}
		else{ //get cells instead
			profilerMarkA("fpga-cpu:starting getCells call");
			fpgaStatus = getCells(small_padded_img, &cell_hists, objectType | (getScores ? FPGA_HOG_GET_SCORES : FPGA_HOG_GET_HISTS));
			profilerMarkA("fpga-cpu:starting detectFromCells call");
			//cpu_hog takes doubles in weights vector
			vector<double> weights_d;
			cpu_hog->detectFromCells(cell_hists, locations, weights_d,
				Size(64 * nstripes + 2 - 2, small_padded_img.rows - 2),
				hitThreshold, win_stride, padding);
			//convert to float again
			hitsWeights.assign(weights_d.begin(), weights_d.end());
		}
		//////////////////////////////////////////////////////////////////////////

		Size scaledWinSize = Size(cvRound(win_size.width*scale), cvRound(win_size.height*scale));
		for (size_t j = 0; j < locations.size(); j++) {
			//sanity check for detections: as long as scaled top corner is inside the original image
			Point pt = locations[j] * scale;
			if (pt.x <= img.cols && pt.y <= img.rows){
				allCandidates.push_back(Rect(pt, scaledWinSize));
				scales.push_back(scale);
				weights.push_back(hitsWeights[j]);
			}
		}
	}
	//end of parallel section
	foundScales.assign(scales.begin(), scales.end());
	foundWeights.assign(weights.begin(), weights.end());
	foundLocations.assign(allCandidates.begin(), allCandidates.end());
	groupScores(foundLocations, foundWeights, foundScales, finalThreshold, useMeanshiftGrouping);

	profilerMarkA("fpga-cpu:done");
	return fpgaStatus;
}


//multi-threaded fpgaHOG launcher taken from inside OpenCV
class fpgaHOGInvoker : public ParallelLoopBody {
public:
	fpgaHOGInvoker( /*const*/ fpgaHOGProcessor* _hog, const HOGDescriptor* _hog_cpu,
		const Mat& _img, double _hitThreshold, Size _winStride, Size _padding,
		const double* _levelScale, std::vector<Rect> * _vec,
		std::vector<float>* _weights, std::vector<double>* _scales,
		Mutex* _vecMtx, Mutex* _fpgaMtx, Mutex* _errMtx, const int _nstripes,
		const bool _getScores, const int _objectType, boost::exception_ptr* worker_exception_)	{
		hog = _hog;
		hog_cpu = _hog_cpu;
		img = _img;
		hitThreshold = _hitThreshold;
		winStride = _winStride;
		padding = _padding;
		levelScale = _levelScale;
		vec = _vec;
		weights = _weights;
		scales = _scales;
		vecMtx = _vecMtx;
		fpgaMtx = _fpgaMtx;
		errMtx = _errMtx;
		nstripes = _nstripes;
		getScores = _getScores;
		objectType = _objectType;
		worker_exception = worker_exception_;
	}

	void operator()(const Range& range) const	{
		int i, i1 = range.start, i2 = range.end;
		double minScale = i1 > 0 ? levelScale[i1] : i2 > 1 ? levelScale[i1 + 1] : std::max(img.cols, img.rows);
		Size maxSz(cvCeil(img.cols / minScale), cvCeil(img.rows / minScale));
		Mat cell_hists;

		Mat smaller_img_buf(maxSz, img.type()), padded_img_buf(maxSz, img.type());
		vector<Point> locations;
		vector<float> hitsWeights;
		vector<double> weights_d;
		try{
			for (i = i1; i < i2; i++) {
				double scale = levelScale[i];
				Size sz(cvRound(img.cols / scale), cvRound(img.rows / scale));
				Mat smaller_img(sz, img.type(), smaller_img_buf.data);
				Mat small_padded_img(sz, img.type(), padded_img_buf.data);
				Size small_size;

				/*resize*/
				profilerMarkA("cff:starting resize");
				if (scale != 1.) { //usual resizing operations
					resize(img, smaller_img, sz, INTER_LINEAR);
					small_size = smaller_img.size();
					if (small_size != Size(FPGA_COLS, FPGA_ROWS)) {
						int top = 0,
							left = 0,
							/* leave the next 2 lines in to pad to frame size each time */
							right = FPGA_COLS - small_size.width,
							//bottom	= FPGA_ROWS - small_size.height;
							bottom = MIN(FPGA_ROWS - small_size.height,
							((small_size.height / 8 + 1) * 8 + 2) - small_size.height);
						/*the alternative is to pad to nearest group of 8 pixels high,
						so as not to send redundant data, and let the existing interface code
						deal with sending enough extra pixels to flush the gradient & classifier pipelines
						*/
						copyMakeBorder(smaller_img, small_padded_img,
							top, bottom, left, right, BORDER_CONSTANT, 0);
					}
					else
						small_padded_img = smaller_img;
				}
				else{ //edge case, levels==1 && size != FPGA_ROWS, FPGA_COLS
					if (img.size() == Size(FPGA_COLS, FPGA_ROWS))
						small_padded_img = img;
					else if (img.size() == Size(FPGA_COLS - 2, FPGA_ROWS - 2)){
						copyMakeBorder(img, small_padded_img,
							1, 1, 1, 1, BORDER_CONSTANT, 0);
					}
					else{
						int top = 0,
							left = 0,
							right = FPGA_COLS - img.cols,
							/* pad to frame size -use this one for 1024x768 */
							//bottom	= FPGA_ROWS - img.rows;
							/* pad to sensible size */
							bottom = MIN(FPGA_ROWS - img.rows, ((img.rows / 8 + 1) * 8 + 2) - img.rows);

						copyMakeBorder(img, small_padded_img,
							top, bottom, left, right, BORDER_CONSTANT, 0);
					}
				}
				/*get histograms and/or classify*/
				if (getScores){
					/*do histograms and classification on the fpga*/
					profilerMarkA("cff:starting getScores call");
					fpgaMtx->lock();
					hog->getCells(small_padded_img, &cell_hists, objectType | (getScores ? FPGA_HOG_GET_SCORES : FPGA_HOG_GET_HISTS));
					profilerMarkA("cff:finished getScores call");
					hog->thresholdScores(cell_hists, Size(64 * nstripes + 2, small_padded_img.rows),
						locations, hitsWeights, hitThreshold, 7, 0, 0, /*15*/
						objectType == FPGA_USE_CAR_HOG ? FPGA_CAR_HOG_VBLOCKS : FPGA_PED_HOG_VBLOCKS);
					//i guess thresholdScores needs to be inside the mutex as
					//cell_hists points to the same mem location for every thread?
					fpgaMtx->unlock();
				}
				else{ //get cells instead
					profilerMarkA("cfc:starting getCells call");
					fpgaMtx->lock();
					hog->getCells(small_padded_img, &cell_hists, objectType | (getScores ? FPGA_HOG_GET_SCORES : FPGA_HOG_GET_HISTS));
					//copy cell_hists to somewhere else
					//cell_hists points to the same mem loc for every thread
					cell_hists.copyTo(smaller_img);
					profilerMarkA("cfc:starting detectFromCells call");
					fpgaMtx->unlock();
					//cpu_hog takes doubles in weights vector
					weights_d.clear();
					hog_cpu->detectFromCells(smaller_img/*cell_hists*/, locations, weights_d,
						Size(img.cols/*64*nstripes+2*/ - 2, /*img.rows -2*/small_padded_img.rows - 2),
						hitThreshold, hog->win_stride, padding);

					//convert to float again
					std::copy(weights_d.begin(), weights_d.end(), back_inserter(hitsWeights));
					profilerMarkA("cfc:finished this scale");
				}

				Size scaledWinSize = Size(cvRound(hog->win_size.width*scale), cvRound(hog->win_size.height*scale));
				vecMtx->lock();
				for (size_t j = 0; j < locations.size(); j++) {
					//sanity check for detections: as long as scaled top corner is inside the original image
					Point pt = locations[j] * scale;
					if (pt.x <= img.cols && pt.y <= img.rows){
						vec->push_back(Rect(pt, scaledWinSize));
						weights->push_back(hitsWeights[j]);
						scales->push_back(scale);
					}
				}
				vecMtx->unlock();
			}
		}
		catch (...) //throw error back to main thread
		{
			cout << "uh oh, error caught in hog worker thread" << endl;
			errMtx->lock();
			*worker_exception = boost::current_exception();
			errMtx->unlock();
		}
	}

	fpgaHOGProcessor* hog;
	const HOGDescriptor *hog_cpu;
	Mat img;
	double hitThreshold;
	Size winStride;
	Size padding;
	const double* levelScale;
	std::vector<Rect>* vec;
	std::vector<float>* weights;
	std::vector<double>* scales;
	Mutex* vecMtx, *fpgaMtx, *errMtx;
	//error handling an mutex in worker threads
	boost::exception_ptr* worker_exception;
	int nstripes;
	bool getScores;
	int objectType;
};


//CFC and CFF versions - select using getScores
int fpgaHOGProcessor::detectMultiScaleMultiThreaded(
	const Mat& img, vector<cv::Rect>& foundLocations, vector<float>& foundWeights,
	const cv::HOGDescriptor* cpu_hog, double hitThreshold, double scale0,
	double finalThreshold, bool useMeanshiftGrouping, bool getScores)
{
	double scale = 1.;
	int levels = 0;
	vector<double> levelScale;
	int fpgaStatus = 0;

	CV_Assert(img.type() == CV_8UC1 || img.type() == CV_8UC4);
	for (levels = 0; levels < nlevels; levels++) {
		levelScale.push_back(scale);
		if (cvRound(img.cols / scale) < win_size.width ||
			cvRound(img.rows / scale) < win_size.height ||
			scale0 <= 1)
			break;
		scale *= scale0;
	}
	levels = max(levels, 1);
	levelScale.resize(levels);

	//unrolled tbb paralleliser for working on different threads/cores. no more
	//than 1 thing gets to access the fpga at once
	//this is OK because we are parallelising by using an accelerator rather than using multiple cores/threads
#ifdef FPGA_IMAGE_IS_PAL
	int nstripes = 12;
#else
	int nstripes=16;
#endif
	vector<double> foundScales;

	vector<Rect>   parLocations;
	vector<float>  parWeights;
	vector<double> parScales;
	Mutex fpgaMtx, vecMtx, errMtx;
	boost::exception_ptr worker_exception;


	profilerMarkA("cfc/cff:starting threading");

	parallel_for_(Range(0, (int)levelScale.size()),
		fpgaHOGInvoker(this, cpu_hog, img, hitThreshold, win_stride, padding, &levelScale[0],
		&parLocations, &parWeights, &parScales, &vecMtx, &fpgaMtx, &errMtx,
		nstripes, getScores, objectType, &worker_exception));

	if (worker_exception) //error handling (tbb seems to break try/catch)
		boost::rethrow_exception(worker_exception);

	profilerMarkA("cfc/cff:done");

	foundLocations.clear();
	foundWeights.clear();
	foundScales.clear();
	std::copy(parLocations.begin(), parLocations.end(), back_inserter(foundLocations));
	std::copy(parWeights.begin(), parWeights.end(), back_inserter(foundWeights));
	std::copy(parScales.begin(), parScales.end(), back_inserter(foundScales));

	//end of parallel section
	groupScores(foundLocations, foundWeights, foundScales, finalThreshold, useMeanshiftGrouping);

	return fpgaStatus;
}
#endif

//these get called no matter if we use  the FPGA or not
//returns a vector of floats trained for the histograms from the FPGA
std::vector<float> fpgaHOGProcessor::getPeopleDetector64x128(void)
{
#include "mydescriptor22.txt"
	return vector<float>(detector, detector + sizeof(detector) / sizeof(detector[0]));
}

//
//returns a vector of floats trained for the histograms from the FPGA
std::vector<float> fpgaHOGProcessor::getFPGACarDetector104x56(void)
{
#include "HOGCarDetector26_20130815.txt"
	return vector<float>(detector, detector + sizeof(detector) / sizeof(detector[0]));
}
//returns a vector of floats trained for the histograms from the GPU/CPU
std::vector<float> fpgaHOGProcessor::getGPUCarDetector104x56(void)
{
#include "HOGCarDetector27_20130815.txt" //
	return vector<float>(detector, detector + sizeof(detector) / sizeof(detector[0]));
}
