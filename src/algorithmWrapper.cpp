//algorithmWrapper.cpp
//contains wrappers and instructions to allow various detection algorithms to be treated 
//as a single type of object
//based on OpenCV's Algorithm class
//calum blair
//10/12/12
#include "stdafx.h"
#include "fpga.h"
#include "acceleratedDetectors.h"

//there is a separate source file for pedestrian detectors.


using namespace std;
using namespace cv;
//Constructors for Detection struct
Detection::Detection(int x, int y, int w, int h, double score_, int source_ = -1){
	bb = Rect(x, y, w, h);
	score = score_;
	source = source_;
}
Detection::Detection(cv::Rect bb_, double score_, int source_ = -1){
	bb = bb_;
	score = score_;
	source = source_;
}
//Default constructor
Detection::Detection(){ ; }


//global map of algorithms and IDs
//every algorithm calls this to get a unique ID number 
//and it stores the names and IDs in a map, then returns the names 
//if called with the ID later
//this also behaves correctly in that it stores an *unordered* map, ie
//doesnt reorder alphabetically when new data arrives
typedef boost::bimap<int, std::string> algoMap;
typedef algoMap::value_type position;
class AlgorithmList{
public:
	AlgorithmList()
	{
		count = 0;
	};
	int getAlgorithmID(std::string algoName){
		algoMap::right_map::const_iterator it = idList.right.find(algoName);
		int key;
		if (it == idList.right.end()){ //if not found
			count++;
			idList.insert(position(count, algoName));
			key = count;
		}
		else
			key = it->second;
		return key;
	};
	std::string getAlgorithmID(int id_){
		return idList.left.at(id_);
	}
	int getNumAlgorithms(bool debug){
		algoMap::left_map::const_iterator it, iend;
		if (debug){
			cout << "AcceleratedAlgorithm Map contents (" << count << " algorithms)" << endl;

			for (it = idList.left.begin(), iend = idList.left.end(); it != iend; it++)
				cout << it->first << " " << it->second << endl;
		}
		return idList.size();
	};
private:
	algoMap idList;
	int count;
};
AlgorithmList algoList;

//making this global as can't see a sensible other way to handle global access to this algoList
int getNumAlgorithms(bool debug = false){
	return algoList.getNumAlgorithms(debug);
};

////////////////////////////////////////////////////////////////////////////////////////////////
////base acceleratedAlgorithm class
//constructor
AcceleratedAlgorithm::AcceleratedAlgorithm() : initialised(false), available(false), enabled(true)
{
	//cout << "constructor of acceleratedAlgorithm called\n";
}

//init definition and call
#ifdef WIN32
cv::Algorithm* createAcceleratedAlgorithm();
#endif
CV_INIT_ALGORITHM(AcceleratedAlgorithm, "AcceleratedAlgorithm.base", false);


//dummy detect() function
void AcceleratedAlgorithm::detect(Mat frame, vector<Detection>& detections){
	cout << "AcceleratedAlgorithm::detect() - this shouldn't get called" << endl;
}
void AcceleratedAlgorithm::detect(HeisenFrame* hf, vector<Detection>& detections){
	cout << "AcceleratedAlgorithm::detect() HeisenFrame- this shouldn't get called" << endl;
}
inline int AcceleratedAlgorithm::getID(void){
	return algoList.getAlgorithmID(this->name());
}
/*inline*/ std::string AcceleratedAlgorithm::getID(int id){
	return algoList.getAlgorithmID(id);
}


///////////////////////////////////////////accelerated hog - gpu
//map internal functionality of algorithm onto this 'detect' function
//constructor
AcceleratedHOG_GGG::AcceleratedHOG_GGG()
{
	cout << "constructor of AcceleratedHOG_GGG called\n";
	initialised = false;
	id = getID();
}
void AcceleratedHOG_GGG::init(std::vector<float> SVMdetector){
	scale = 1.05;
	nlevels = 13;//cv::gpu::HOGDescriptor::DEFAULT_NLEVELS;
	gr_threshold = 1;
	requiredChannels = 4;
	hit_threshold = 0.;
	gpu_hog = cv::gpu::HOGDescriptor(Size(64, 128), Size(16, 16), Size(8, 8), Size(8, 8), 9,
		cv::gpu::HOGDescriptor::DEFAULT_WIN_SIGMA, 0.2, true,
		nlevels);
	if (SVMdetector.size() > 1)
		gpu_hog.setSVMDetector(SVMdetector);
	else
		gpu_hog.setSVMDetector(cv::gpu::HOGDescriptor::getPeopleDetector64x128());

	enabled = true;
	available = true;
	initialised = true;
	//set implementation
	implementation = Implementation((enabled && available && initialised), 17.6f, 229, 0.52f, ALGORITHM_HOG, id, RESOURCES_GPU);
}

void AcceleratedHOG_GGG::detect(HeisenFrame* hf, vector<Detection>& detections){
	cout << "calling smart detect inside gpu hog\n";
	assert(initialised); //need to make sure detector is initialised

	gpu::GpuMat g_frame;
	if (hf->usePatch)
		g_frame = hf->getGpuPatch(CV_8UC4);
	else
		g_frame = hf->getGpuFrame(CV_8UC4);

	//#define HOG_GGG_DEBUG
#ifdef HOG_GGG_DEBUG
	Mat temp;
	g_frame.download(temp);
	imshow("test",temp);
	cvWaitKey(10);
#endif

	found.clear();
	scores.clear();
	gpu_hog.nlevels = nlevels;
	//detect
	gpu_hog.computeConfidenceMultiScale(g_frame, found, hit_threshold, Size(8, 8),
		Size(0, 0), scores, gr_threshold, scale);
	//format detections
	for (size_t i = 0; i < found.size(); i++)
	{
		detections.push_back(Detection(found[i], scores[i], id));
	}
}

void AcceleratedHOG_GGG::detect(Mat frame, vector<Detection>& detections){
	cout << "calling detect inside gpu hog\n";
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
	for (size_t i = 0; i < found.size(); i++)
	{
		detections.push_back(Detection(found[i], scores[i], id));
	}
}

//init definition and call
#ifdef WIN32
Algorithm* createAcceleratedHOG_GGG();
#endif
CV_INIT_ALGORITHM(AcceleratedHOG_GGG, "AcceleratedAlgorithm.HOG_GGG",
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
bool initModule_accel_hog_ggg(void)
{
	Ptr<Algorithm> phog_ggg = createAcceleratedHOG_GGG();
	bool flag = (phog_ggg->info() != 0);
	if (!flag)
		cout << "algorithm structure not created properly\n";
	return flag;
}


///////////////////////////////////////////accelerated hog - gpu - fpga
//map internal functionality of algorithm onto this 'detect' function
//constructor
AcceleratedHOG_GFG::AcceleratedHOG_GFG()
{
	cout << "constructor of AcceleratedHOG_GFG called\n";
	initialised = false;
	id = getID();
}
void AcceleratedHOG_GFG::init(std::vector<float> SVMdetector){
	scale = 1.05;
	nlevels = 13;//cv::gpu::HOGDescriptor::DEFAULT_NLEVELS;
	gr_threshold = 1;
	requiredChannels = 1;
	hit_threshold = 0.;
	gpu_hog = cv::gpu::HOGDescriptor(Size(64, 128), Size(16, 16), Size(8, 8), Size(8, 8), 9,
		cv::gpu::HOGDescriptor::DEFAULT_WIN_SIGMA, 0.2, true,
		nlevels);
	if (SVMdetector.size() > 1)
		gpu_hog.setSVMDetector(SVMdetector);
	else
		gpu_hog.setSVMDetector(fpga_hog.getPeopleDetector64x128());

	enabled = true;
	available = true;
	initialised = true;
	//set implementation
	implementation = Implementation((enabled && available && initialised), 39., 200, 0.59f, ALGORITHM_HOG, id, RESOURCES_GPU | RESOURCES_FPGA, true);
}

void AcceleratedHOG_GFG::detect(HeisenFrame* hf, vector<Detection>& detections){
	cout << "calling smart detect inside gfg hog\n";
	assert(initialised); //need to make sure detector is initialised
	Mat frame;
	if (hf->usePatch)
		frame = hf->getPatch(CV_8UC1);
	else
		frame = hf->getFrame(CV_8UC1);
	detect(frame, detections);
}


void AcceleratedHOG_GFG::detect(Mat frame, vector<Detection>& detections){
	cout << "calling detect inside gfg hog\n";
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
	for (size_t i = 0; i < found.size(); i++){
		detections.push_back(Detection(found[i], scores[i], id));
	}
}

//init definition and call
#ifdef WIN32
Algorithm* createAcceleratedHOG_GFG();
#endif
CV_INIT_ALGORITHM(AcceleratedHOG_GFG, "AcceleratedAlgorithm.HOG_GFG",
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
bool initModule_accel_hog_gfg(void)
{
	Ptr<Algorithm> phog_gfg = createAcceleratedHOG_GFG();
	bool flag = (phog_gfg->info() != 0);
	if (!flag)
		cout << "algorithm structure not created properly\n";
	return flag;
}


///////////////////////////////////////////accelerated hog - fpga
//map internal functionality of algorithm onto this 'detect' function
//constructor
AcceleratedHOG_GFF::AcceleratedHOG_GFF()
{
	cout << "constructor of AcceleratedHOG_GFF called\n";
	initialised = false;
	id = getID();
}
void AcceleratedHOG_GFF::init(std::vector<float> SVMdetector){
	scale = 1.05;
	nlevels = 13;//cv::HOGDescriptor::DEFAULT_NLEVELS;
	gr_threshold = 1;
	requiredChannels = 1;
	hit_threshold = 0.25;

	gpu_hog = cv::gpu::HOGDescriptor(Size(64, 128), Size(16, 16), Size(8, 8), Size(8, 8), 9,
		cv::gpu::HOGDescriptor::DEFAULT_WIN_SIGMA, 0.2, true,
		nlevels);
	if (SVMdetector.size() > 1)
		gpu_hog.setSVMDetector(SVMdetector);
	else
		gpu_hog.setSVMDetector(fpga_hog.getPeopleDetector64x128());

	enabled = true;
	available = true;
	initialised = true;
	//set implementation
	implementation = Implementation((enabled && available && initialised), 27.5f, 186, 0.61f, ALGORITHM_HOG, id, RESOURCES_GPU | RESOURCES_FPGA, true);

}

void AcceleratedHOG_GFF::detect(HeisenFrame* hf, vector<Detection>& detections){
	cout << "calling smart detect inside gff hog\n";
	assert(initialised); //need to make sure detector is initialised
	Mat frame;
	if (hf->usePatch)
		frame = hf->getPatch(CV_8UC1);
	else
		frame = hf->getFrame(CV_8UC1);

	detect(frame, detections);
}

void AcceleratedHOG_GFF::detect(Mat frame, vector<Detection>& detections){
	cout << "calling detect inside gff hog\n";
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
	fpgaStatus = fpga_hog.detectMultiScale(frame1, found, scores, &gpu_hog, hit_threshold,
		scale, gr_threshold, true);
	//format detections
	for (size_t i = 0; i < found.size(); i++){
		detections.push_back(Detection(found[i], scores[i], id));
	}
}

//init definition and call
#ifdef WIN32
Algorithm* createAcceleratedHOG_GFF();
#endif
CV_INIT_ALGORITHM(AcceleratedHOG_GFF, "AcceleratedAlgorithm.HOG_GFF",
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
bool initModule_accel_hog_gff(void)
{
	Ptr<Algorithm> phog_gff = createAcceleratedHOG_GFF();
	bool flag = (phog_gff->info() != 0);
	if (!flag)
		cout << "algorithm structure not created properly" << endl;
	return flag;
}


///////////////////////////////////////////accelerated hog - cpu - fpga - cpu
//map internal functionality of algorithm onto this 'detect' function
//constructor
AcceleratedHOG_CFC::AcceleratedHOG_CFC()
{
	cout << "constructor of AcceleratedHOG_CFC called\n";
	initialised = false;
	id = getID();
}
void AcceleratedHOG_CFC::init(std::vector<float> SVMdetector){
	scale = 1.05;
	nlevels = 13;//cv::HOGDescriptor::DEFAULT_NLEVELS;
	gr_threshold = 1;
	requiredChannels = 1;
	hit_threshold = 0.;
	cpu_hog = cv::HOGDescriptor(Size(64, 128), Size(16, 16), Size(8, 8), Size(8, 8), 9, 1, -1,
		HOGDescriptor::L2Hys, 0.2, true, nlevels);
	if (SVMdetector.size() > 1)
		cpu_hog.setSVMDetector(SVMdetector);
	else
		cpu_hog.setSVMDetector(fpga_hog.getPeopleDetector64x128());

	enabled = true;
	available = true;
	initialised = true;
	//set implementation
	implementation = Implementation((enabled && available && initialised), 117.3f, 187, 0.59f, ALGORITHM_HOG, id, RESOURCES_FPGA, true);
}

void AcceleratedHOG_CFC::detect(HeisenFrame* hf, vector<Detection>& detections){
	Mat img;
	if (hf->usePatch)
		img = hf->getPatch(CV_8UC1);
	else
		img = hf->getFrame(CV_8UC1);
	detect(img, detections);
}

void AcceleratedHOG_CFC::detect(Mat frame, vector<Detection>& detections){
	cout << "calling detect inside fpga-cpu hog\n";
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
	for (size_t i = 0; i < found.size(); i++){
		detections.push_back(Detection(found[i], scores[i], id));
	}
}

//init definition and call
#ifdef WIN32
Algorithm* createAcceleratedHOG_CFC();
#endif
CV_INIT_ALGORITHM(AcceleratedHOG_CFC, "AcceleratedAlgorithm.HOG_CFC",
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
bool initModule_accel_hog_cfc(void)
{
	Ptr<Algorithm> phog_cfc = createAcceleratedHOG_CFC();
	bool flag = (phog_cfc->info() != 0);
	if (!flag)
		cout << "algorithm structure not created properly" << endl;
	return flag;
}

///////////////////////////////////////////accelerated hog - cpu - fpga 
//map internal functionality of algorithm onto this 'detect' function
//constructor
AcceleratedHOG_CFF::AcceleratedHOG_CFF()
{
	cout << "constructor of AcceleratedHOG_CFF called\n";
	initialised = false;
	id = getID();
}
void AcceleratedHOG_CFF::init(std::vector<float> SVMdetector){
	scale = 1.05;
	nlevels = 13;//cv::HOGDescriptor::DEFAULT_NLEVELS;
	gr_threshold = 1;
	requiredChannels = 1;
	hit_threshold = 0.;
	cpu_hog = cv::HOGDescriptor(Size(64, 128), Size(16, 16), Size(8, 8), Size(8, 8), 9, 1, -1,
		HOGDescriptor::L2Hys, 0.2, true, nlevels);
	if (SVMdetector.size() > 1)
		cpu_hog.setSVMDetector(SVMdetector);
	else
		cpu_hog.setSVMDetector(fpga_hog.getPeopleDetector64x128());

	enabled = true;
	available = true;
	initialised = true;
	//set implementation
	implementation = Implementation((enabled && available && initialised), 23., 190, 0.62f, ALGORITHM_HOG, id, RESOURCES_FPGA, true);
}

void AcceleratedHOG_CFF::detect(HeisenFrame* hf, vector<Detection>& detections){
	Mat img;
	if (hf->usePatch)
		img = hf->getPatch(CV_8UC1);
	else
		img = hf->getFrame(CV_8UC1);
	detect(img, detections);
}

void AcceleratedHOG_CFF::detect(Mat frame, vector<Detection>& detections){
	cout << "calling detect inside fpga hog\n";
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
			bottom = FPGA_ROWS - frame1.rows;
		//bottom = MIN(FPGA_ROWS-frame1.height, 
		//	((frame1.height/8 +1) *8 +2)-frame1.height);
		copyMakeBorder(frame1, frame2,
			top, bottom, left, right, BORDER_CONSTANT);
	}
	else
		frame2 = frame1;
	//the fact that the scores here is a vector of doubles instead of floats doesn't matter
	//using the getScores version rather than getCells
	fpgaStatus = fpga_hog.detectMultiScaleMultiThreaded(frame2, found, scores, &cpu_hog, hit_threshold,
		scale, gr_threshold, false, true);
	//format detections
	for (size_t i = 0; i < found.size(); i++){
		detections.push_back(Detection(found[i], scores[i], id));
	}
}

//init definition and call
#ifdef WIN32
Algorithm* createAcceleratedHOG_CFF();
#endif
CV_INIT_ALGORITHM(AcceleratedHOG_CFF, "AcceleratedAlgorithm.HOG_CFF",
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
bool initModule_accel_hog_cff(void)
{
	Ptr<Algorithm> phog_cff = createAcceleratedHOG_CFF();
	bool flag = (phog_cff->info() != 0);
	if (!flag)
		cout << "algorithm structure not created properly" << endl;
	return flag;
}

///////////////////////////////////////////acceleratedmog - gpu
//map internal functionality of algorithm onto this 'detect' function
//constructor
AcceleratedMOG_GPU::AcceleratedMOG_GPU()
{
	cout << "constructor of AcceleratedMOG_GPU called\n";
	initialised = false;
	id = getID();
}

void AcceleratedMOG_GPU::init(cv::Mat frame){
	gr_threshold = 1;
	requiredChannels = 1;
	hit_threshold = 0.;
	area_threshold = 10;
	area_y_coeff = 0;

	//set frame1 and foreground to size of frame and page-lock them
	frame1 = Mat::zeros(frame.size(), CV_8UC1);
	foreground = Mat::zeros(frame.size(), CV_8UC1);
	gpu::registerPageLocked(frame1);
	gpu::registerPageLocked(foreground);

	mogProcessor.init(frame);

	bbConfidence = 0.5;
	enabled = true;
	available = true;
	initialised = true;
	//set implementation
	implementation = Implementation((enabled && available && initialised), 8., 202, -1, ALGORITHM_MOG, id, RESOURCES_GPU);
}

void AcceleratedMOG_GPU::detect(HeisenFrame* hf, vector<Detection>& detections){
	cout << "calling detect inside gpu mog\n";
	assert(initialised); //need to make sure detector is initialised

	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	vector<Detection> temp;

	//detect
	gpu::GpuMat g_frame = hf->getGpuFrame(CV_8UC1);
	mogProcessor.getForeground(g_frame, foreground, use_fast_lrate);
	//#define DEBUG_BGSUB 1
#ifdef DEBUG_BGSUB	//debug
	Mat im2,im3;
	vector<Mat> mv1(3);
	foreground.convertTo(im3,CV_8UC1);

	split(hf->getFrame(CV_8UC3),mv1);
	mv1[2] = im3;
	merge(mv1,im2);
	//imshow("fgmask",im2);
#endif
	//do connected components
	findContours(foreground, contours, hierarchy, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);
	use_fast_lrate = false;
	for (size_t i = 0; i < contours.size(); i++){
		Rect tempRect = boundingRect(contours[i]);
#ifndef BANK_ST
		if (tempRect.area() > 0.25*g_frame.rows*g_frame.cols){
			use_fast_lrate = true;
		}
#endif
		if (tempRect.area() >= (area_threshold + tempRect.y*area_y_coeff)){
			detections.push_back(Detection(tempRect, bbConfidence, id));
#ifdef DEBUG_BGSUB
			rectangle(im2,tempRect,CV_RGB(255,255,0));
#endif
		}
	}
#ifdef DEBUG_BGSUB
	imshow("mdet",im2);
#endif
}

void AcceleratedMOG_GPU::detect(Mat frame, vector<Detection>& detections){
	cout << "calling detect inside gpu mog\n";
	assert(initialised); //need to make sure detector is initialised

	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	vector<Detection> temp;

	//convert to right number of channels
	if (frame.channels() == requiredChannels)
		frame1 = frame;
	else
		cvtColor(frame, frame1, CV_BGR2GRAY);
	mogProcessor.getForeground(frame, foreground, use_fast_lrate);

	//#define DEBUG_BGSUB 1
#ifdef DEBUG_BGSUB	//debug
	Mat im2,im3;
	vector<Mat> mv1(3);
	foreground.convertTo(im3,CV_8UC1);

	split(frame,mv1);
	mv1[2] = im3;
	merge(mv1,im2);
	//imshow("fgmask",im2);
#endif
	//do connected components
	findContours(foreground, contours, hierarchy, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);
	use_fast_lrate = false;
	for (size_t i = 0; i < contours.size(); i++){
		Rect tempRect = boundingRect(contours[i]);
		if (tempRect.area() > 0.25*frame.rows*frame.cols){
			use_fast_lrate = true;
		}
		if (tempRect.area() >= (area_threshold + tempRect.y*area_y_coeff)){
			detections.push_back(Detection(tempRect, bbConfidence, id));
			//if debug
#ifdef DEBUG_BGSUB
			rectangle(im2,tempRect,CV_RGB(255,255,0));
#endif
		}
	}
#ifdef DEBUG_BGSUB
	imshow("mdet",im2);
#endif
}

//init definition and call
#ifdef WIN32
Algorithm* createAcceleratedMOG_GPU();
#endif
CV_INIT_ALGORITHM(AcceleratedMOG_GPU, "AcceleratedAlgorithm.MOG_GPU",
	obj.info()->addParam(obj, "requiredChannels", obj.requiredChannels);
	obj.info()->addParam(obj, "hit_threshold", obj.hit_threshold);
	obj.info()->addParam(obj, "area_threshold", obj.area_threshold);
	obj.info()->addParam(obj, "area_y_coeff", obj.area_y_coeff);
	obj.info()->addParam(obj, "bbConfidence", obj.bbConfidence);

	obj.info()->addParam(obj, "enabled", obj.enabled);
	obj.info()->addParam(obj, "available", obj.available);
	obj.info()->addParam(obj, "initialised", obj.initialised);
);

//also need a init function in the same file as above
//this doesn't do anything except call the info() implementation
bool initModule_accel_mog_gpu(void)
{
	Ptr<Algorithm> pmog_gpu = createAcceleratedMOG_GPU();
	bool flag = (pmog_gpu->info() != 0);
	if (!flag)
		cout << "algorithm structure not created properly" << endl;
	return flag;
}
