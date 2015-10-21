#include "stdafx.h"
#include "fpga.h"

#include "acceleratedDetectors.h"
#include "tracking.h"


using namespace std;
using namespace cv;

//this file contains individual implementations of each detector for car/pedestrains ad on various gpu/fpga combinations
std::vector<float> getCarDetector104x56(void)
{
#include "HOGCarDetector27_20130815.txt"
	return vector<float>(detector, detector + sizeof(detector) / sizeof(detector[0]));
}


///////////////////////////////////////////accelerated hog - car cpu	
//map internal functionality of algorithm onto this 'detect' function
//constructor
AcceleratedHOG_Car_CCC::AcceleratedHOG_Car_CCC()
{
	cout << "constructor of AcceleratedHOG_Car_CCC called" << endl;
	initialised = false;
	id = getID();
}
void AcceleratedHOG_Car_CCC::init(std::vector<float> SVMdetector){
	scale = 1.05;
	nlevels = 13;//cv::gpu::HOGDescriptor::DEFAULT_NLEVELS;
	gr_threshold = 1;
	requiredChannels = 3;
	hit_threshold = 0.;
	cpu_hog = HOGDescriptor(Size(104, 56), Size(16, 16), Size(8, 8), Size(8, 8), 18,
		1, -1, HOGDescriptor::L1Sqrt, 0, true, nlevels, true); //initialse car detector
	if (SVMdetector.size() > 1)//dont do this
		cpu_hog.setSVMDetector(SVMdetector);
	else
		cpu_hog.setSVMDetector(getCarDetector104x56());

	enabled = true;
	available = true;
	initialised = true;
	//set implementation (uses no resources)
	implementation = Implementation((enabled && available && initialised), 318., 194, 0.89f, ALGORITHM_HOG_CAR, id, 0);
}

void AcceleratedHOG_Car_CCC::detect(HeisenFrame* hf, vector<Detection>& detections){
	Mat img;
	if (hf->usePatch)
		img = hf->getPatch(CV_8UC3);
	else
		img = hf->getFrame(CV_8UC3);
	detect(img, detections);
}

void AcceleratedHOG_Car_CCC::detect(Mat frame, vector<Detection>& detections){
	cout << "calling detect inside cpu car hog" << endl;
	assert(initialised); //need to make sure detector is initialised

	found.clear();
	scores.clear();
	cpu_hog.nlevels = nlevels;
	//detect
	cpu_hog.detectMultiScale(frame, found, scores, hit_threshold, Size(8, 8), Size(0, 0), scale,
		gr_threshold, false);
	//format detections
	int area_limit = (int)(0.25 * frame.rows * frame.cols);
	for (size_t i = 0; i < found.size(); i++)
	{
		if (found[i].width * found[i].height < area_limit)
			detections.push_back(Detection(found[i], scores[i], id));
	}
}

//init definition and call
#ifdef WIN32
Algorithm* createAcceleratedHOG_Car_CCC();
#endif
CV_INIT_ALGORITHM(AcceleratedHOG_Car_CCC, "AcceleratedAlgorithm.HOG_Car_CCC",
	obj.info()->addParam(obj, "nlevels", obj.nlevels);
obj.info()->addParam(obj, "requiredChannels", obj.requiredChannels);
obj.info()->addParam(obj, "scale", obj.scale);
obj.info()->addParam(obj, "gr_threshold", obj.gr_threshold);
obj.info()->addParam(obj, "hit_threshold", obj.hit_threshold);

obj.info()->addParam(obj, "enabled", obj.enabled);
obj.info()->addParam(obj, "available", obj.available);
obj.info()->addParam(obj, "initialised", obj.initialised);
);

//also need a init function in the same file as above
//this doesn't do anything except call the info() implementation
bool initModule_accel_hog_car_ccc(void)
{
	Ptr<Algorithm> phog_car_ccc = createAcceleratedHOG_Car_CCC();
	bool flag = (phog_car_ccc->info() != 0);
	if (!flag)
		cout << "algorithm structure not created properly" << endl;
	return flag;
}

///////////////////////////////////////////accelerated hog - car gpu
//map internal functionality of algorithm onto this 'detect' function
//constructor
AcceleratedHOG_Car_GGG::AcceleratedHOG_Car_GGG()
{
	cout << "constructor of AcceleratedHOG_Car_GGG called" << endl;
	initialised = false;
	id = getID();
}
void AcceleratedHOG_Car_GGG::init(std::vector<float> SVMdetector){
	scale = 1.05;
	nlevels = 13;//cv::gpu::HOGDescriptor::DEFAULT_NLEVELS;
	gr_threshold = 1;
	requiredChannels = 4;
	hit_threshold = 0.;
	gpu_hog = cv::gpu::HOGDescriptor(Size(104, 56), Size(16, 16), Size(8, 8), Size(8, 8), 18,
		cv::gpu::HOGDescriptor::DEFAULT_WIN_SIGMA, 0.2, true, nlevels,
		cv::gpu::HOGDescriptor::NORM_L1Sqrt, true);//initialise car detector
	if (SVMdetector.size() > 1)//dont do this
		gpu_hog.setSVMDetector(SVMdetector);

	else
		gpu_hog.setSVMDetector(getCarDetector104x56());

	enabled = true;
	available = true;
	initialised = true;
	//set implementation
	implementation = Implementation((enabled && available && initialised), 34.3f, 229, 0.89f, ALGORITHM_HOG_CAR, id, RESOURCES_GPU);
}

void AcceleratedHOG_Car_GGG::detect(HeisenFrame* hf, vector<Detection>& detections){
	cout << "calling smart detect inside gpu car hog" << endl;
	assert(initialised); //need to make sure detector is initialised

	gpu::GpuMat g_frame;
	if (hf->usePatch)
		g_frame = hf->getGpuPatch(CV_8UC4);
	else
		g_frame = hf->getGpuFrame(CV_8UC4);

	found.clear();
	scores.clear();
	gpu_hog.nlevels = nlevels;
	//detect
	gpu_hog.computeConfidenceMultiScale(g_frame, found, hit_threshold, Size(8, 8),
		Size(0, 0), scores, gr_threshold, scale);
	//format detections
	int area_limit = (int)(0.25 * g_frame.rows * g_frame.cols);
	for (size_t i = 0; i < found.size(); i++){
		if (found[i].width * found[i].height < area_limit)
			detections.push_back(Detection(found[i], scores[i], id));
	}
}

void AcceleratedHOG_Car_GGG::detect(Mat frame, vector<Detection>& detections){
	cout << "calling detect inside gpu car hog" << endl;
	assert(initialised); //need to make sure detector is initialised
	static Mat frame4;
	static gpu::GpuMat gpu_img;
	//convert to right number of channels
	if (frame.channels() == requiredChannels)
		frame4 = frame;
	else
		cvtColor(frame, frame4, CV_BGR2BGRA);
	gpu_img.upload(frame4);
	found.clear();
	scores.clear();
	gpu_hog.nlevels = nlevels;
	//detect
	gpu_hog.computeConfidenceMultiScale(gpu_img, found, hit_threshold, Size(8, 8),
		Size(0, 0), scores, gr_threshold, scale);
	//format detections
	int area_limit = (int)(0.25 * frame.rows * frame.cols);
	for (size_t i = 0; i < found.size(); i++)	{
		if (found[i].width * found[i].height < area_limit)
			detections.push_back(Detection(found[i], scores[i], id));
	}
}

//init definition and call
#ifdef WIN32
Algorithm* createAcceleratedHOG_Car_GGG();
#endif
CV_INIT_ALGORITHM(AcceleratedHOG_Car_GGG, "AcceleratedAlgorithm.HOG_Car_GGG",
	obj.info()->addParam(obj, "nlevels", obj.nlevels);
obj.info()->addParam(obj, "requiredChannels", obj.requiredChannels);
obj.info()->addParam(obj, "scale", obj.scale);
obj.info()->addParam(obj, "gr_threshold", obj.gr_threshold);
obj.info()->addParam(obj, "hit_threshold", obj.hit_threshold);

obj.info()->addParam(obj, "enabled", obj.enabled);
obj.info()->addParam(obj, "available", obj.available);
obj.info()->addParam(obj, "initialised", obj.initialised);
);

//also need a init function in the same file as above
//this doesn't do anything except call the info() implementation
bool initModule_accel_hog_car_ggg(void)
{
	Ptr<Algorithm> phog_car_ggg = createAcceleratedHOG_Car_GGG();
	bool flag = (phog_car_ggg->info() != 0);
	if (!flag)
		cout << "algorithm structure not created properly" << endl;
	return flag;
}

///////////////////////////////////////////accelerated car hog - gpu - fpga
//map internal functionality of algorithm onto this 'detect' function
//constructor
AcceleratedHOG_Car_GFG::AcceleratedHOG_Car_GFG()
{
	cout << "constructor of AcceleratedHOG_Car_GFG called\n";
	initialised = false;
	id = getID();
}
void AcceleratedHOG_Car_GFG::init(std::vector<float> SVMdetector){
	scale = 1.05;
	nlevels = 13;//cv::gpu::HOGDescriptor::DEFAULT_NLEVELS;
	gr_threshold = 1;
	requiredChannels = 1;
	hit_threshold = 0.;
	gpu_hog = cv::gpu::HOGDescriptor(Size(104, 56), Size(16, 16), Size(8, 8), Size(8, 8), 18,
		cv::gpu::HOGDescriptor::DEFAULT_WIN_SIGMA, 0.2, true,
		nlevels, cv::gpu::HOGDescriptor::NORM_L1Sqrt, true);
	if (SVMdetector.size() > 1)
		gpu_hog.setSVMDetector(SVMdetector);
	else
		gpu_hog.setSVMDetector(fpga_hog.getFPGACarDetector104x56());
	fpga_hog = fpgaHOGProcessor(FPGA_USE_CAR_HOG);

	enabled = true;
	available = true;
	initialised = true;
	//set implementation
	implementation = Implementation((enabled && available && initialised), 60., 200, 0.92f, ALGORITHM_HOG_CAR, id, RESOURCES_GPU | RESOURCES_FPGA, false);
}

void AcceleratedHOG_Car_GFG::detect(HeisenFrame* hf, vector<Detection>& detections){
	assert(initialised); //need to make sure detector is initialised
	Mat frame;
	if (hf->usePatch)
		frame = hf->getPatch(CV_8UC1);
	else
		frame = hf->getFrame(CV_8UC1);

	detect(frame, detections);
}


void AcceleratedHOG_Car_GFG::detect(Mat frame, vector<Detection>& detections){
	cout << "calling detect inside gfg car hog\n";
	assert(initialised); //need to make sure detector is initialised
	static Mat frame1;
	int fpgaStatus;
	//convert to right number of channels
	if (frame.channels() == requiredChannels)
		frame1 = frame;
	else
		cvtColor(frame, frame1, CV_BGR2GRAY);
	found.clear();
	scores.clear();
	fpga_hog.nlevels = nlevels;
	//detect
	//fix for 1024x768 images
	if (frame1.size() == Size(FPGA_COLS - 2, FPGA_ROWS - 2)){
		Mat frame2;
		copyMakeBorder(frame1, frame2, 1, 1, 1, 1, BORDER_CONSTANT);
		frame2.copyTo(frame1);
	}
	//when multiple gpu opbject detectors are live, we need to re-upload constants 
	//every time the detected object changes. this means one call to set_up_constants before 
	//we do anything with the gpu.
	gpu_hog.reUploadConstants();

	fpgaStatus = fpga_hog.detectMultiScale(frame1, found, scores, &gpu_hog, hit_threshold,
		scale, gr_threshold, false);
	//format detections
	for (size_t i = 0; i < found.size(); i++)	{
		detections.push_back(Detection(found[i], scores[i], id));
	}
}

//init definition and call
#ifdef WIN32
Algorithm* createAcceleratedHOG_Car_GFG();
#endif
CV_INIT_ALGORITHM(AcceleratedHOG_Car_GFG, "AcceleratedAlgorithm.HOG_Car_GFG",
	obj.info()->addParam(obj, "nlevels", obj.nlevels);
obj.info()->addParam(obj, "requiredChannels", obj.requiredChannels);
obj.info()->addParam(obj, "scale", obj.scale);
obj.info()->addParam(obj, "gr_threshold", obj.gr_threshold);
obj.info()->addParam(obj, "hit_threshold", obj.hit_threshold);

obj.info()->addParam(obj, "enabled", obj.enabled);
obj.info()->addParam(obj, "available", obj.available);
obj.info()->addParam(obj, "initialised", obj.initialised);
);

//also need a init function in the same file as above
//this doesn't do anything except call the info() implementation
bool initModule_accel_hog_car_gfg(void)
{
	Ptr<Algorithm> pHOG_Car_GFG = createAcceleratedHOG_Car_GFG();
	bool flag = (pHOG_Car_GFG->info() != 0);
	if (!flag)
		cout << "algorithm structure not created properly\n";
	return flag;
}


///////////////////////////////////////////accelerated hog - cpu - fpga - cpu
//map internal functionality of algorithm onto this 'detect' function
//constructor
AcceleratedHOG_Car_CFC::AcceleratedHOG_Car_CFC()
{
	cout << "constructor of AcceleratedHOG_Car_CFC called\n";
	initialised = false;
	id = getID();
}
void AcceleratedHOG_Car_CFC::init(std::vector<float> SVMdetector){
	scale = 1.05;
	nlevels = 13;//cv::HOGDescriptor::DEFAULT_NLEVELS;
	gr_threshold = 1;
	requiredChannels = 1;
	hit_threshold = 0.;
	fpga_hog = fpgaHOGProcessor(FPGA_USE_CAR_HOG);
	cpu_hog = cv::HOGDescriptor(Size(104, 56), Size(16, 16), Size(8, 8), Size(8, 8), 18, 1, -1,
		HOGDescriptor::L1Sqrt, 0.2, true, nlevels, true);
	if (SVMdetector.size() > 1)
		cpu_hog.setSVMDetector(SVMdetector);
	else
		cpu_hog.setSVMDetector(fpga_hog.getFPGACarDetector104x56());

	enabled = true;
	available = true;
	initialised = true;
	//set implementation
	implementation = Implementation((enabled && available && initialised), 175.6f, 189, 0.94f, ALGORITHM_HOG_CAR, id, RESOURCES_FPGA, false);
}

void AcceleratedHOG_Car_CFC::detect(HeisenFrame* hf, vector<Detection>& detections){
	Mat img;
	if (hf->usePatch)
		img = hf->getPatch(CV_8UC1);
	else
		img = hf->getFrame(CV_8UC1);
	detect(img, detections);
}

void AcceleratedHOG_Car_CFC::detect(Mat frame, vector<Detection>& detections){
	cout << "calling detect inside fpga-cpu car hog\n";
	assert(initialised); //need to make sure detector is initialised
	Mat frame1, frame2;
	int fpgaStatus;

	found.clear();
	scores.clear();
	fpga_hog.nlevels = nlevels;
	//convert to right number of channels
	if (frame.channels() == requiredChannels)
		frame1 = frame;
	else
		cvtColor(frame, frame1, CV_BGR2GRAY);

	//fix for 1024x768 images
	if (frame1.size() == Size(FPGA_COLS - 2, FPGA_ROWS - 2))
		copyMakeBorder(frame1, frame2, 1, 1, 1, 1, BORDER_CONSTANT);
	//fix for other images: unfortunately this breaks any speed advantage we would have when working on smaller 
	//images as we don't pass on this info to the fpgaHOGProcessor
	else if (frame1.size() != Size(FPGA_COLS, FPGA_ROWS)){
		int	top = 0,
			left = 0,
			/* leave the next 2 lines in to pad to frame size each time */
			right = FPGA_COLS - frame1.cols,
			//bottom = MIN(FPGA_ROWS-frame1.height, 
			//	((frame1.height/8 +1) *8 +2)-frame1.height);
			bottom = FPGA_ROWS - frame1.rows;
		copyMakeBorder(frame1, frame2,
			top, bottom, left, right, BORDER_CONSTANT);
	}
	else
		frame2 = frame1;
	//the fact that the scores here is a vector of doubles instead of floats doesn't matter
	//using the getScores version rather than getCells
	fpgaStatus = fpga_hog.detectMultiScaleMultiThreaded(frame2, found, scores, &cpu_hog, hit_threshold,
		scale, gr_threshold, false, false);
	//format detections
	for (size_t i = 0; i < found.size(); i++)	{
		detections.push_back(Detection(found[i], scores[i], id));
	}
}

//init definition and call
#ifdef WIN32
Algorithm* createAcceleratedHOG_Car_CFC();
#endif
CV_INIT_ALGORITHM(AcceleratedHOG_Car_CFC, "AcceleratedAlgorithm.HOG_Car_CFC",
	obj.info()->addParam(obj, "nlevels", obj.nlevels);
obj.info()->addParam(obj, "requiredChannels", obj.requiredChannels);
obj.info()->addParam(obj, "scale", obj.scale);
obj.info()->addParam(obj, "gr_threshold", obj.gr_threshold);
obj.info()->addParam(obj, "hit_threshold", obj.hit_threshold);

obj.info()->addParam(obj, "enabled", obj.enabled);
obj.info()->addParam(obj, "available", obj.available);
obj.info()->addParam(obj, "initialised", obj.initialised);
);

//also need a init function in the same file as above
//this doesn't do anything except call the info() implementation
bool initModule_accel_hog_car_cfc(void)
{
	Ptr<Algorithm> pHOG_Car_CFC = createAcceleratedHOG_Car_CFC();
	bool flag = (pHOG_Car_CFC->info() != 0);
	if (!flag)
		cout << "algorithm structure not created properly" << endl;
	return flag;
}


///////////////////////////////////////////accelerated hog cpu	
//map internal functionality of algorithm onto this 'detect' function
//constructor
AcceleratedHOG_CCC::AcceleratedHOG_CCC()
{
	cout << "constructor of AcceleratedHOG_CCC called" << endl;
	initialised = false;
	id = getID();
}
void AcceleratedHOG_CCC::init(std::vector<float> SVMdetector){
	scale = 1.05;
	nlevels = 13;//cv::gpu::HOGDescriptor::DEFAULT_NLEVELS;
	gr_threshold = 1;
	requiredChannels = 3;
	hit_threshold = 0.;
	cpu_hog = HOGDescriptor(Size(64, 128), Size(16, 16), Size(8, 8), Size(8, 8), 9, 1, -1,
		HOGDescriptor::L2Hys, 0.2, true, cv::HOGDescriptor::DEFAULT_NLEVELS);
	//initialse ped detector
	if (SVMdetector.size() > 1)//dont do this
		cpu_hog.setSVMDetector(SVMdetector);
	else
		cpu_hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());

	enabled = true;
	available = true;
	initialised = true;
	//set implementation (uses no resources)
	implementation = Implementation((enabled && available && initialised), 282.f, 191, 0.53f, ALGORITHM_HOG, id, 0);
}

void AcceleratedHOG_CCC::detect(HeisenFrame* hf, vector<Detection>& detections){
	Mat img;
	if (hf->usePatch)
		img = hf->getPatch(CV_8UC3);
	else
		img = hf->getFrame(CV_8UC3);
	detect(img, detections);
}

void AcceleratedHOG_CCC::detect(Mat frame, vector<Detection>& detections){
	cout << "calling detect inside cpu car hog" << endl;
	assert(initialised); //need to make sure detector is initialised

	found.clear();
	scores.clear();
	cpu_hog.nlevels = nlevels;
	//detect
	cpu_hog.detectMultiScale(frame, found, scores, hit_threshold, Size(8, 8), Size(0, 0), scale,
		gr_threshold, false);
	//format detections
	int area_limit = (int)(0.25 * frame.rows * frame.cols);
	for (size_t i = 0; i < found.size(); i++)	{
		if (found[i].width * found[i].height < area_limit)
			detections.push_back(Detection(found[i], scores[i], id));
	}
}

//init definition and call
#ifdef WIN32
Algorithm* createAcceleratedHOG_CCC();
#endif
CV_INIT_ALGORITHM(AcceleratedHOG_CCC, "AcceleratedAlgorithm.HOG_CCC",
	obj.info()->addParam(obj, "nlevels", obj.nlevels);
obj.info()->addParam(obj, "requiredChannels", obj.requiredChannels);
obj.info()->addParam(obj, "scale", obj.scale);
obj.info()->addParam(obj, "gr_threshold", obj.gr_threshold);
obj.info()->addParam(obj, "hit_threshold", obj.hit_threshold);

obj.info()->addParam(obj, "enabled", obj.enabled);
obj.info()->addParam(obj, "available", obj.available);
obj.info()->addParam(obj, "initialised", obj.initialised);
);

//also need a init function in the same file as above
//this doesn't do anything except call the info() implementation
bool initModule_accel_hog_ccc(void)
{
	Ptr<Algorithm> phog_ccc = createAcceleratedHOG_CCC();
	bool flag = (phog_ccc->info() != 0);
	if (!flag)
		cout << "algorithm structure not created properly" << endl;
	return flag;
}
