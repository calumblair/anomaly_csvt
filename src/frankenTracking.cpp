#include "stdafx.h"
#ifdef USE_WINDRIVER_FPGA_INTERFACE
#include "wdc_lib.h"
#endif
#include "fpga.h"
#include "acceleratedDetectors.h"
#include "display.h"
#include "tracking.h"
#include "unsupervisedHeatmap.h"
#include "bayesianAnomalyOracle.h"
#include "clustering.h"
#include "trackingApp.h"

//defs for evaluation runs
#define TEST_POWER 1
#define TEST_SPEED 2
#define TEST_AUTO 3


using namespace std;
using namespace cv;

#ifdef HAVE_BOOST
using namespace boost::filesystem;
namespace po = boost::program_options;
#endif

extern void chooseBestMappingFromAlgorithm(vector<Ptr<AcceleratedAlgorithm> > candidates, std::vector<int> functions,
	simplepriorities p, vector<Ptr<AcceleratedAlgorithm> >* mapping, Implementation* chosenSolution);
extern void getSceneSpecificData(const int scene_id, cv::Mat* H_cam2base, cv::Size* canvas_size,
	cv::Point* canvas_tl, cv::Point* canvas_br);
extern int getHomographyIndex(std::string token_);
extern "C" inline int getNumAlgorithms(bool debug);

////////////////////////////////////////////////////////////////////////

std::list<int> functions; //global function list, modified via callback
int globalPower;
int globalAccuracy;
int globalLatency;
Priorities globalPriorities;// = Priorities::getInstance(); //ditto, extra logic for callbacks
//name these here so they're all the same length
const string powerTrackbar = "Power     |";
const string accTrackbar = "Accuracy  |";
const string timeTrackbar = "Speed     |";
const string threshTrackbar = "Hit threshold";

float hit_threshold;
int globalHitThreshold;
bool globalMergeDetections;
bool globalAutoPrioritise;
bool globalDrawClusters;
bool globalDrawBase;
bool globalDrawTracksOnCam = true;
bool globalAttemptRealtime = false;

//look for an algorithm in a list
// if it's there, remove it
// if not, add it at the end
void toggleAlgorithm(int algoToCheck, list<int>& algoList){
	list<int>::iterator pos = algoList.begin();
	bool found = false;
	for (pos = algoList.begin(); pos != algoList.end(); pos++){
		if (*pos == algoToCheck){ //if found, remove the algorithm from the list
			algoList.erase(pos);
			found = true;
			break;
		}
	}
	if (!found) //add the algorithm
		algoList.push_back(algoToCheck);
}

//method for setting multiple variables from a single callback from
// stackoverflow q 12997441
enum callbackAlgorithmType{
	ENUM_HOG,
	ENUM_MOG,
	ENUM_PATCH,
	ENUM_CAR
};


void functionSelectCallback(int value, void* algorithmType){
	//value param not actually used as we're only toggling the value
	//c++ reinterpret of pointer as one of the values in callbackAlgorithmType  above
	callbackAlgorithmType* whichAlgorithm = reinterpret_cast<callbackAlgorithmType*>(algorithmType);
	switch (*whichAlgorithm){
	case ENUM_HOG:
		toggleAlgorithm(ALGORITHM_HOG, functions);
		break;
	case ENUM_MOG:
		toggleAlgorithm(ALGORITHM_MOG, functions);
		break;
	case ENUM_CAR:
		toggleAlgorithm(ALGORITHM_HOG_CAR, functions);
		break;
	}
}

inline void incrementFrameCounter(int* d_min, int* d_sec, int* d_frame, const int source_fps){
	(*d_frame)++;
	if (*d_frame == source_fps){
		*d_frame = 0;
		(*d_sec)++;
	}
	if (*d_sec == 60){
		(*d_min)++;
		*d_sec = 0;
	}
}

//controls modification of priorities via keypress, ui and programmatic access
Priorities::Priorities(){
	accuracy = 3;
	latency = 4;
	power = 3;
	enableCallback = false;
}
int Priorities::sum(void){
	return (accuracy + latency + power);
}


enum callbackPriorityType{
	ENUM_POWER,
	ENUM_ACCURACY,
	ENUM_TIME,
};


void Priorities::reWeight(int newVal, int which_priority){
	//set pointers based on whichever value just got changed
	static bool upTick = false, downTick = false;
	int *first, *second, *third;
	switch (which_priority){

	case PRIORITY_POWER:
		first = &power;
		second = &latency;
		third = &accuracy;
		break;

	case PRIORITY_ACCURACY:
		first = &accuracy;
		second = &latency;
		third = &power;
		break;

	case PRIORITY_TIME:
	default:
		first = &latency;
		second = &power;
		third = &accuracy;
		break;
	}

	if (newVal > *first && newVal < (numCredits))  //if increased
	{
		(*first)++; //increment whatever was highest priority
		while (this->sum() > numCredits){
			if (*second > 0)		(*second)--;
			if (*third > 0)		(*third)--;
		}
		if (this->sum() < numCredits){ //edge case - switch between 2nd & third
			if (upTick){
				(*second)++;
				upTick = false;
			}
			else{
				(*third)++;
				upTick = true;
			}
		}
	}
	else
	{
		(*first)--;
		while (this->sum() < numCredits){
			if (*second < (numCredits - 1))	(*second)++;
			if (*third <(numCredits - 1))	(*third)++;
		}
		if (this->sum() > numCredits){//edge case
			if (downTick){
				(*second)--;
				downTick = false;
			}
			else{
				(*third)--;
				downTick = true;
			}
		}
	}
	//assert(this->sum() ==numCredits);
	if (this->sum() != numCredits)
		cout << "oh no";//debug HERE
}


void Priorities::callback(int trackbarValue, void* trackerType){
	//c++ reinterpret of pointer as one of the values in callbackAlgorithmType  above
	callbackPriorityType* whichPriority = reinterpret_cast<callbackPriorityType*>(trackerType);
	switch (*whichPriority){
	case ENUM_POWER:
		//in each case we have NUM_CREDITS to distribute between 3 priorities
		reWeight(trackbarValue, PRIORITY_POWER);
		enableCallback = false;
		cvSetTrackbarPos(timeTrackbar.c_str(), NULL, latency);
		cvSetTrackbarPos(accTrackbar.c_str(), NULL, accuracy);
		globalLatency = latency;
		globalAccuracy = accuracy;
		enableCallback = true;
		break;
	case ENUM_ACCURACY:
		//in each case we have NUM_CREDITS to distribute between 3 priorities
		reWeight(trackbarValue, PRIORITY_ACCURACY);
		enableCallback = false;
		cvSetTrackbarPos(timeTrackbar.c_str(), NULL, latency);
		cvSetTrackbarPos(powerTrackbar.c_str(), NULL, power);
		globalLatency = latency;
		globalPower = power;
		enableCallback = true;
		break;
	case ENUM_TIME:
		//in each case we have NUM_CREDITS to distribute between 3 priorities
		reWeight(trackbarValue, PRIORITY_TIME);
		//openCV C++bug ? use C impl
		enableCallback = false;
		cvSetTrackbarPos(powerTrackbar.c_str(), NULL, power);
		cvSetTrackbarPos(accTrackbar.c_str(), NULL, accuracy);
		globalPower = power;
		globalAccuracy = accuracy;
		enableCallback = true;
		break;
	}
}


void Priorities::set(int accuracy_, int latency_, int power_){
	accuracy = accuracy_;
	latency = latency_;
	power = power_;
	assert(sum() <= numCredits);
}
void Priorities::get(int* accuracy_, int* latency_, int* power_){
	*accuracy_ = accuracy;
	*latency_ = latency;
	*power_ = power;
}

simplepriorities Priorities::get(){
	simplepriorities q;
	q.accuracy = accuracy;
	q.latency = latency;
	q.power = power;
	return q;
}


void processFrame(HeisenFrame* hf, std::vector<Ptr<AcceleratedAlgorithm> > mapping, std::vector<Detection>& detections){
	unsigned int i = 0;

	//work through algorithms
	for (i = 0; i < mapping.size(); i++){
		bool flag = mapping[i]->info() != 0;
		//cout << "flag is " << flag <<endl;
		//cout << "algorithm " << mapping[i]->name() <<"\n";
		mapping[i]->detect(hf, detections);
	}
}

//tie perspective warping to generate base image (which has significant compute time) to current global priority
inline void prioritisedWarpPerspective(HeisenFrame* hf, Mat& base_img, Mat H_cam2base, Size base_img_sz, simplepriorities p){
	static gpu::GpuMat g_frame, g_base_img;
	if (p.latency > p.power){
		cout << "doing perspective warp on gpu\n";
		g_frame = hf->getGpuFrame(CV_8UC3);
		gpu::warpPerspective(g_frame, g_base_img, H_cam2base, base_img_sz);
		g_base_img.download(base_img);
	}
	else
		warpPerspective(hf->getFrame(CV_8UC3), base_img, H_cam2base, base_img_sz);
}

void processInput(char inputKey, std::vector<Ptr<AcceleratedAlgorithm> >& mapping, std::list<int>& functions_){
	int tmp = 0;
	list<int> functions_int = functions_; //
	size_t i = 0;
	// find which one is the MOG algorithm
	Ptr<AcceleratedAlgorithm> pmog = NULL;
	for (i = 0; i < mapping.size(); i++){
		if (mapping[i].obj->name() == "AcceleratedAlgorithm.MOG_GPU")		pmog = mapping[i];
	}

	switch (inputKey){
	case 'o':
		//increase area threshold for gpu mog
		//is there a sensible way to select a particular algo?
		if (pmog){
			tmp = pmog->getInt("area_threshold");
			tmp++;
			pmog->set("area_threshold", tmp);
			cout << "MOG area threshold is " << tmp << "\n";
			break;
		}
	case 'k':
		//decrase area threshold for gpu mog
		if (pmog){
			tmp = pmog->getInt("area_threshold");
			tmp--;
			pmog->set("area_threshold", tmp);
			cout << "MOG area threshold is " << tmp << "\n";
			break;
		}
	}
	functions_ = functions; //modify the original list
}

void prioritiesCallbackWrapper(int a, void* b){
	if (globalPriorities.enableCallback) //only eval callback for the actual variable that the mouse moved
		globalPriorities.callback(a, b);
	else{
		int a, l, p;
		globalPriorities.get(&a, &l, &p);
		cvSetTrackbarPos(powerTrackbar.c_str(), NULL, p);
		cvSetTrackbarPos(accTrackbar.c_str(), NULL, a);
		cvSetTrackbarPos(timeTrackbar.c_str(), NULL, l);
	}
}
void hitThreshCallback(int val, void* data){
	hit_threshold = val*1.0f / 10;
}

void mergeDetectionsCallback(int value, void* dummy){
	globalMergeDetections = (value != 0);//->bool
}

void autoPrioritiseCallback(int value, void* dummy){
	globalAutoPrioritise = (value != 0);//->bool
}

void realTimeCallback(int value, void* dummy){
	globalAttemptRealtime = (value != 0);//->bool
}

void drawClustersCallback(int value, void* dummy){
	globalDrawClusters = (value != 0);//->bool
}

void drawBaseCallback(int value, void* dummy){
	globalDrawBase = (value != 0);//->bool
}

void drawDetections(vector<Detection> detections, Mat& canvas,
	bool debug, DisplayColourMap* colours, const int nchannels,
	const int thickness, FILE* fp, int fc)
{
	CvScalar paintcolours = cvScalar(0., 0., 0., 0.);
	int currentsource = 0;
	for (size_t i = 0; i < detections.size(); i++)
	{
		Rect r = detections[i].bb;
		if (currentsource != detections[i].source){
			paintcolours = colours->getColoursFromID(detections[i].source);
		}

		if (debug)
			fprintf(fp, "%d %d %d %d %d %d %f\n", fc, detections[i].source, r.x, r.y, r.width, r.height, detections[i].score);

		//cout << r.x << " " << r.y << " " << r.width << " " << r.height << endl;
		//try alpha blending (doesnt work)
		CvScalar tmppaint;
		if (detections[i].score > hit_threshold){
			if (nchannels == 3)
				tmppaint = cvScalar(paintcolours.val[0] * detections[i].score,
				paintcolours.val[1] * detections[i].score,
				paintcolours.val[2] * detections[i].score);
			else if (nchannels == 4)
				tmppaint = cvScalar(paintcolours.val[0] * detections[i].score,
				paintcolours.val[1] * detections[i].score,
				paintcolours.val[2] * detections[i].score,
				paintcolours.val[3] * detections[i].score);
			rectangle(canvas, r.tl(), r.br(), tmppaint, thickness);
		}
	}
}

void autoSetPriorities(float overall_anomaly_level)
{
	//set these some way apart to allow for hysteresis
	float high_anom_thresh = 15.f;
	float low_anom_thresh = 13.f;
	simplepriorities p = globalPriorities.get();
	bool do_update = false;
	int choice = 0, new_weight = 0;
	if (overall_anomaly_level > high_anom_thresh){
		choice = PRIORITY_TIME;
		new_weight = 9;
		if (p.latency != new_weight){
			do_update = true;
			cout << "adjusting for higher speed\n";
		}
	}
	else if (overall_anomaly_level < low_anom_thresh){
		choice = PRIORITY_POWER;
		new_weight = 9;
		if (p.power != new_weight){
			do_update = true;
			cout << "adjusting for lower power\n";
		}
	}
	if (do_update){
		globalPriorities.reWeight(new_weight, choice);//this changes the values in globalPriorities
		globalPriorities.enableCallback = false; //disable callback
		prioritiesCallbackWrapper(NULL, NULL); //update trackbar positions
		globalPriorities.enableCallback = true; //reenable callback
	}
}

int main(int argc, char* argv[])
{
	VideoCapture vc;
	Mat frame;
	hit_threshold = 0.;
	string filename;
	string heatmap_path = "../data/";
	char inputKey;
	static DisplayColourMap colours;
	bool using_video = false, using_folder = false;
	int seekTo = 0;
	const int nlevels_peds = 8; // max height to allow for detection of peds at bottom of frame
	const int nlevels_cars = /*22*/11; // max height to allow for detection of cars at bottom of frame
	bool debug = false; //TODO add method to turn this on/off from GUI
	bool save_clusters_on_exit = false, load_clusters_at_start = false;
	int run_evaluation = 0;
	string save_cluster_path = "clusters_out", load_cluster_path = "clusters_in";
	string log_path;
	string testName = "";

	bool disable_fpga_access = false; //override: enable this to completely disable all attempts to use FPGA
	//handle cmd line options - from boost program_options tutorial
	po::options_description cmd_desc("Allowed options");
	cmd_desc.add_options()
		("video", po::value<string>(), "input video file")
		("folder", po::value<string>(), "input folder of image files")
		("help", "produce help message")
		("initial_mapping", po::value< vector<int> >(), "specify initial algorithm mapping")
		("algorithms", "List available algorithms")
		("seek", po::value<int>(), "position in number of frames to seek to at start of video file")
		("heatmap_path", po::value<string>(), "load heatmaps from here")
		("save_clusters_path", po::value<string>(), "Cluster save path")
		("load_clusters_path", po::value<string>(), "Cluster load path")
		("evaluation", po::value<string>(), "Do evaluation type (false, power, speed, auto)")
		("log_path", po::value<string>(), "folder to place log files")
		("disable_fpga", po::value<bool>(), "disable fpga completely")
		;
	//handle arguments without labels
	po::positional_options_description cmd_p;
	cmd_p.add("video", 1);
	cmd_p.add("folder", 1);

	po::variables_map cmd_args;
	po::store(po::command_line_parser(argc, argv).
		options(cmd_desc).positional(cmd_p).run(), cmd_args);
	po::notify(cmd_args);
	int source_fps = 25, d_frame = 0, d_sec = 0, d_min = 0;
	double source_frame_time_ms = 1000.0 / source_fps, elapsed = 0;
	if (cmd_args.count("video")){
		using_video = true;
		filename = cmd_args["video"].as<string>();
		vc.open(filename);
		//seek if req
		if (cmd_args.count("seek")){
			seekTo = cmd_args["seek"].as<int>();
			if (seekTo < vc.get(CV_CAP_PROP_FRAME_COUNT))
				vc.set(CV_CAP_PROP_POS_FRAMES, seekTo);

		}
		vc.get(CV_CAP_PROP_FPS);
		vc >> frame;
		cout << "using " << filename << " at " << frame.cols << "W x " << frame.rows << "H\n";
		if (frame.rows == 0){
#ifdef WIN32
			MessageBox(NULL, "Image file doesn't exist.", "Error", MB_OK | MB_ICONERROR);
#else
			//we're not adding a GUI dependency in Linux - just print on command line
			printf("Error: Image file doesn't exist");
#endif
		}
	}
	if (cmd_args.count("folder")){
		using_folder = true;
		filename = cmd_args["folder"].as<string>();
		cout << "using " << filename << " as folder\n";
	}
	if (cmd_args.count("heatmap_path")){
		filename = cmd_args["heatmap_path"].as<string>();
		cout << "using " << heatmap_path << " as heatmap path\n";
	}
	if (cmd_args.count("save_clusters_path")){
		save_clusters_on_exit = true;
		save_cluster_path = cmd_args["save_clusters_path"].as<string>();
		cout << "using " << save_cluster_path << " as clusters save path\n";
	}
	if (cmd_args.count("load_clusters_path")){
		load_clusters_at_start = true;
		load_cluster_path = cmd_args["load_clusters_path"].as<string>();
		cout << "using " << load_cluster_path << " as clusters load path\n";
	}
	if (cmd_args.count("log_path")){
		log_path = cmd_args["log_path"].as<string>();
		cout << "using " << log_path << " as anomaly files log path\n";
	}
	if (cmd_args.count("evaluation")){
		testName = cmd_args["evaluation"].as<string>();
		if (testName == "power") run_evaluation = TEST_POWER;
		if (testName == "speed") run_evaluation = TEST_SPEED;
		if (testName == "auto") run_evaluation = TEST_AUTO;
		if (testName == "false") run_evaluation = 0;

		if (log_path.empty()){
#ifdef WIN32
			MessageBox(NULL, "No log_path present! can't run evaluation without log folder", "Error", MB_OK | MB_ICONERROR);
#else
			//we're not adding a GUI dependency in Linux - just print on command line
			printf("Error: No log_path present! can't run evaluation without log folder");
#endif
		}
#ifdef WIN32
		log_path = log_path + "\\" + testName;
#else
		log_path = log_path + boost::filesystem::path::preferred_separator + testName;
#endif
		cout << "running evaluation " << run_evaluation << " (" << testName << ")\n";
	}
	if (cmd_args.count("disable_fpga")){
		disable_fpga_access = cmd_args["disable_fpga"].as<bool>();
		cout << "disabling attempts to access FPGA completely. using CPU and GPU processors only" << endl;
	}

	//if we're using a folder, start reading frames from it
	YetAnotherFrameReader imagefolder;
	if (using_folder){
		imagefolder = YetAnotherFrameReader(filename.c_str());
		bool success = imagefolder.getFrame(&frame);
	}

	//smart heterogeneous frame store
	HeisenFrame heisenframe;
	try{
		//initialise algorithms
		initModule_features2d();
		initModule_ml();

		initModule_nonfree();
		//init - must call init_my_algorithm before create<my_algorithm>
		//initialise then create each algorithm and get a pointer to them
		//then initialise the internal workings of the algorithm
		initModule_accel_hog_ggg();
		Ptr<AcceleratedHOG_GGG> phog_ggg = Algorithm::create<AcceleratedHOG_GGG>("AcceleratedAlgorithm.HOG_GGG");
		phog_ggg->init(std::vector<float>(0));
		phog_ggg->set("hit_threshold", 0.5);
		phog_ggg->set("gr_threshold", 2);

		phog_ggg->set("nlevels", nlevels_peds);

		//fpga stuff
		bool fpga_available = false; //this will be set programmatically
		Ptr<AcceleratedHOG_GFF> phog_gff;
		Ptr<AcceleratedHOG_GFG> phog_gfg;
		Ptr<AcceleratedHOG_CFC> phog_cfc;
		Ptr<AcceleratedHOG_CFF> phog_cff;
		Ptr<AcceleratedHOG_Car_GFG> phog_car_gfg;
		Ptr<AcceleratedHOG_Car_CFC> phog_car_cfc;

		try{
			initModule_accel_hog_gff();
			phog_gff = Algorithm::create<AcceleratedHOG_GFF>("AcceleratedAlgorithm.HOG_GFF");
			phog_gff->init(std::vector<float>(0));

			initModule_accel_hog_gfg();
			phog_gfg = Algorithm::create<AcceleratedHOG_GFG>("AcceleratedAlgorithm.HOG_GFG");
			phog_gfg->init(std::vector<float>(0));

			initModule_accel_hog_cfc();
			phog_cfc = Algorithm::create<AcceleratedHOG_CFC>("AcceleratedAlgorithm.HOG_CFC");
			phog_cfc->init(std::vector<float>(0));

			initModule_accel_hog_cff();
			phog_cff = Algorithm::create<AcceleratedHOG_CFF>("AcceleratedAlgorithm.HOG_CFF");
			phog_cff->init(std::vector<float>(0));

			{ //if HOG_GFF is called before HOG_CFC, the latter will crash as sth wrong with cellhists
				//to avoid this we call once here
				//used to have a check to avoid this if running on the laptop/no FPGA but this appears to be fixed now
				//this may not work with files with size different to 1024x768
				vector<Detection> temp;
				int tmp = phog_cfc->getInt("nlevels");
				phog_cfc->set("nlevels", 1);
				phog_cfc->detect(frame, temp); //try single level of detection
				phog_cfc->set("nlevels", tmp);
			}

			initModule_accel_hog_car_gfg();
			phog_car_gfg = Algorithm::create<AcceleratedHOG_Car_GFG>("AcceleratedAlgorithm.HOG_Car_GFG");
			phog_car_gfg->init(std::vector<float>(0));

			initModule_accel_hog_car_cfc();
			phog_car_cfc = Algorithm::create<AcceleratedHOG_Car_CFC>("AcceleratedAlgorithm.HOG_Car_CFC");
			phog_car_cfc->init(std::vector<float>(0));

			fpga_available = true;

			phog_gff->set("hit_threshold", 0.5);
			phog_gff->set("gr_threshold", 2);

			phog_gff->set("nlevels", nlevels_peds);
			phog_gfg->set("nlevels", nlevels_peds);
			phog_cfc->set("nlevels", nlevels_peds);
			phog_cff->set("nlevels", nlevels_peds);

			phog_car_gfg->set("nlevels", nlevels_cars);
			phog_car_cfc->set("nlevels", nlevels_cars);

			phog_car_gfg->set("hit_threshold", 0.75);
			phog_car_gfg->set("gr_threshold", 1);
			phog_car_cfc->set("hit_threshold", 0.75);
			phog_car_cfc->set("gr_threshold", 1);

		}
		catch (const exception& e)
		{
			cout << "error initialising fpga-based detectors:" << e.what() << endl;
			cout << "continuing without fpga support" << endl;
			fpga_available = false;
		}

		gpu::registerPageLocked(frame); //pin memory for streaming
		initModule_accel_mog_gpu();
		Ptr<AcceleratedMOG_GPU> pmog_gpu = Algorithm::create<AcceleratedMOG_GPU>("AcceleratedAlgorithm.MOG_GPU");
		pmog_gpu->init(frame);
		pmog_gpu->set("area_threshold",/*150*/400);
		pmog_gpu->set("area_y_coeff", 2000.0f / frame.rows);
#ifdef BANK_ST
		pmog_gpu->set("area_threshold", 150);
		pmog_gpu->set("area_y_coeff", 0);
#endif

		initModule_accel_hog_car_ccc();
		Ptr<AcceleratedHOG_Car_CCC> phog_car = Algorithm::create<AcceleratedHOG_Car_CCC>("AcceleratedAlgorithm.HOG_Car_CCC");
		phog_car->init(std::vector<float>(0));
		phog_car->set("hit_threshold", 0.75);
		phog_car->set("gr_threshold", 1);
		phog_car->set("nlevels", nlevels_cars);

		initModule_accel_hog_car_ggg();
		Ptr<AcceleratedHOG_Car_GGG> phog_car_ggg = Algorithm::create<AcceleratedHOG_Car_GGG>("AcceleratedAlgorithm.HOG_Car_GGG");
		phog_car_ggg->init(std::vector<float>(0));
		phog_car_ggg->set("hit_threshold", 0.75);
		phog_car_ggg->set("gr_threshold", 1);
		phog_car_ggg->set("nlevels", nlevels_cars);


		initModule_accel_hog_ccc();
		Ptr<AcceleratedHOG_CCC> phog_ccc = Algorithm::create<AcceleratedHOG_CCC>("AcceleratedAlgorithm.HOG_CCC");
		phog_ccc->init(std::vector<float>(0));
		phog_ccc->set("hit_threshold", 0.5);
		phog_ccc->set("gr_threshold", 2);

		phog_ccc->set("nlevels", nlevels_peds);


		//OK. which algorithms do we have?
		vector<string> algorithms;
		Algorithm::getList(algorithms);

		cout << "Algorithms that OpenCV knows about: " << algorithms.size() << endl;
		for (size_t i = 0; i < algorithms.size(); i++)
			cout << algorithms[i] << endl;


		//declare result vectors etc
		vector<Detection> initialDetections, improvedDetections, dummyDetections;
		vector<Point3i> trackedPoints;
		vector<AnomalousDetection> anomDets;
		vector<int>loggedAnomalyIDs;
		Mat img_to_show;

		vector< Ptr<AcceleratedAlgorithm> > candidates;
		vector< Ptr<AcceleratedAlgorithm> > mapping;

		//fill candidates vector with our available algorithm implementations (known here as AcceleratedAlgorithms)
		//chooseBestActions() chooses mapping from this vector based on priorities
		candidates.push_back(phog_ggg); //disable on laptop
		if (disable_fpga_access){
			cout << "completely disabling FPGA use during runtime due to override" << endl;
			fpga_available = false;
		}
		if (fpga_available){
			candidates.push_back(phog_gff);
			candidates.push_back(phog_gfg);
			//candidates.push_back(phog_cfc);
			candidates.push_back(phog_cff);
			candidates.push_back(phog_car_cfc);
			candidates.push_back(phog_car_gfg);
		}
		candidates.push_back(pmog_gpu);
		candidates.push_back(phog_car);
		candidates.push_back(phog_car_ggg);
		candidates.push_back(phog_ccc);

		{
			int nAlgs = getNumAlgorithms(true);
			cout << " There are " << nAlgs << " accelerated algorithms" << endl;
		}
		//initially,
		mapping = candidates;

		//set up patch processor. 
		PatchProcessor motionProcessor;
		if (run_evaluation == TEST_POWER){
			motionProcessor.car = phog_car_gfg;
			motionProcessor.ped = phog_gff;
		}
		else if (run_evaluation == TEST_SPEED){
			motionProcessor.car = phog_car_ggg;
			motionProcessor.ped = phog_ggg;
		}
		else { //else do normal
			motionProcessor.car = phog_car_ggg;
			motionProcessor.ped = phog_ggg;
		}

		//build map of implementations to algos
		//and map of implementations to objects
		std::map<int, int> impl2algo, impl2obj;
		for (size_t i = 0; i < candidates.size(); i++){
			int alg = candidates[i]->implementation.algorithm;
			impl2algo[candidates[i]->getID()] = alg;
			if (alg == ALGORITHM_HOG)
				impl2obj[candidates[i]->getID()] = OBJECT_PED;
			else if (alg == ALGORITHM_HOG_CAR)
				impl2obj[candidates[i]->getID()] = OBJECT_CAR;
		}
		//add one more to end
		impl2obj[ALGORITHM_MOTION_ONLY] = OBJECT_UNDETERMINED;


		//details of each frame's solution
		Implementation solutionDetails;
		//timing: frame or overall, and work
		ProcessingTimer workTimer, frameTimer, runTimer;
		//set up GUI
		try{
			namedWindow("Detections", CV_WINDOW_AUTOSIZE | CV_WINDOW_KEEPRATIO | CV_GUI_EXPANDED);
			imshow("Detections", frame); //show the first frame to get the size right
			moveWindow("Detections", 50, 50); //put it somewhere sensible
			//change CV_WINDOW_AUTOSIZE above to CV_WINDOW_NORMAL to use fullscreen
			//setWindowProperty("Detections",CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
			if (!fpga_available)
				displayOverlay("Detections", "FPGA Not Found. Using GPU/CPU acceleration.", 30000);
		}
		catch (Exception onCreateWindow)
		{
			cout << onCreateWindow.what() << endl <<
				"This is probably caused by QT problems. Apply bugfix 2255." << endl;
			exit(1);
		}

		//set up control panel
		string pedDetectButtonLabel = "Pedestrian Detection (HOG)";
		string motionButtonLabel = "Motion Detection";
		string carDetectButtonLabel = "Car Detector";

		//set true/false to enable/disable functions by default
		//the new callbackxxxType(ENUM_XXX) gives us a unique number for each slider or button pushed
		createButton(pedDetectButtonLabel, functionSelectCallback, new callbackAlgorithmType(ENUM_HOG), CV_CHECKBOX, true);
		createButton(motionButtonLabel, functionSelectCallback, new callbackAlgorithmType(ENUM_MOG), CV_CHECKBOX, true);
		createButton(carDetectButtonLabel, functionSelectCallback, new callbackAlgorithmType(ENUM_CAR), CV_CHECKBOX, true);

		//set up priorities - done via globalPriorities at top 
		//set default priorities
		globalPriorities.set(3, 4, 3);

		string realTimeButtonLabel = "Realtime (allow framedrop)";
		string mergeButtonLabel = "Merge, Track, Cluster";
		string drawClustersButtonLabel = "Draw Clusters";
		string drawBaseButtonLabel = "Draw Base and Tracks";
		string autoPrioritiseLabel = "Auto Prioritise";
		//get dummy variables for the tracker
		int trackBarLimit = globalPriorities.getNumCredits() - 1;
		globalPriorities.get(&globalAccuracy, &globalLatency, &globalPower);
		globalMergeDetections = true;
		globalAutoPrioritise = false;

		// second argument must be "" not NULL to ensure we select control panel for this
		createTrackbar(powerTrackbar, "", &globalPower, trackBarLimit, &prioritiesCallbackWrapper, new callbackPriorityType(ENUM_POWER));
		createTrackbar(accTrackbar, "", &globalAccuracy, trackBarLimit, &prioritiesCallbackWrapper, new callbackPriorityType(ENUM_ACCURACY));
		createTrackbar(timeTrackbar, "", &globalLatency, trackBarLimit, &prioritiesCallbackWrapper, new callbackPriorityType(ENUM_TIME));

		createButton(realTimeButtonLabel, &realTimeCallback, 0, CV_CHECKBOX, globalAttemptRealtime);
		createButton(autoPrioritiseLabel, &autoPrioritiseCallback, 0, CV_CHECKBOX, globalAutoPrioritise);
		createButton(mergeButtonLabel, &mergeDetectionsCallback, 0, CV_CHECKBOX, globalMergeDetections);
		createButton(drawBaseButtonLabel, &drawBaseCallback, 0, CV_CHECKBOX, false);
		createButton(drawClustersButtonLabel, &drawClustersCallback, 0, CV_CHECKBOX, false);

		if (run_evaluation){ //run on real time and/or auto.
			if (run_evaluation == TEST_AUTO){
				setButtonState(autoPrioritiseLabel, 1);
			}
			if (run_evaluation == TEST_POWER){
				globalPriorities.set(0, 1, 9);
				//globalPriorities.reWeight(9, PRIORITY_POWER);//this changes the values in globalPriorities
				globalPriorities.enableCallback = false; //disable callback
				prioritiesCallbackWrapper(NULL, NULL); //update trackbar positions
				globalPriorities.enableCallback = true; //reenable callback
			}
			if (run_evaluation == TEST_SPEED){
				globalPriorities.set(0, 9, 1);
				//globalPriorities.reWeight(9, PRIORITY_TIME);//this changes the values in globalPriorities
				globalPriorities.enableCallback = false; //disable callback
				prioritiesCallbackWrapper(NULL, NULL); //update trackbar positions
				globalPriorities.enableCallback = true; //reenable callback
			}
			setButtonState(realTimeButtonLabel, 1);
		}
		/////////////////////////////////////
		//tracker stuff

		Mat H_cam2base;
		Size base_img_sz;
		Point canvas_tl;
		Point canvas_br;
		gpu::GpuMat g_frame, g_base_img;
		//these are mappings for ilids scene  3

		const int ilids_scene = 98; //used for loading heatmaps
		int homography_idx = 0;
		try{
			homography_idx = getHomographyIndex(filename);
		}
		catch (std::exception e){
			cout << e.what();
			homography_idx = 3; //for unknown homographies
		}
		getSceneSpecificData(homography_idx, &H_cam2base, &base_img_sz, &canvas_tl, &canvas_br);


		ClusteringApp ca;

		Mat base_img;

		namedWindow("base");
		moveWindow("base", 1100, -25);

		TrackerApp app(H_cam2base, impl2obj, impl2algo, motionProcessor, canvas_tl, canvas_br);
		float anomaly_level = 0;
		//loop
		int fc = 0, skippedFrameCounter = 0;
		//set up file logging

		boost::filesystem::path leaf = filename;
		leaf = leaf.stem();
		leaf = leaf.replace_extension(".txt");
		boost::filesystem::path outfile = boost::filesystem::path(log_path) / leaf;
		boost::filesystem::path snapshotpath = boost::filesystem::path(log_path) / "img" / leaf;

		FILE* fp;
		if ((fp = fopen("frankentracker_log.txt", "a+")) == NULL)
#ifdef WIN32
			MessageBox(NULL, "failed to open main log file", "duh", MB_OK | MB_ICONERROR);
#else
			//we're not adding a GUI dependency in Linux - just print on command line
			printf("Error: failed to open main log file\n");
#endif
		if (fseek(fp, 0, SEEK_END) != 0)
#ifdef WIN32
			MessageBox(NULL, "failed to seek in main log file", "duh", MB_OK | MB_ICONERROR);
#else
			//we're not adding a GUI dependency in Linux - just print on command line
			printf("Error: failed to seek in main log file\n");
#endif
		fseek(fp, EOF, 0);

		time_t rawtime;
		struct tm * timestruct;
		time(&rawtime); //get time and put in rawtime
		timestruct = localtime(&rawtime); //convert to timeestruct


		//fprintf(fp,"%s begin at %s\n",filename,asctime(timestruct)); 
		fprintf(fp, "%s begin at time ???\n", filename);

		FILE* anom_fp;
		if ((anom_fp = fopen((outfile.string()).c_str(), "w")) == NULL)
#ifdef WIN32
			MessageBox(NULL, "failed to open file-specific log file", "duh", MB_OK | MB_ICONERROR);
#else
			//we're not adding a GUI dependency in Linux - just print on command line
			printf("Error: failed to open file-specific log file %s\n",(outfile.string()).c_str());
#endif

		fprintf(anom_fp, "%s\nx\ty\tframe#\tmin\tsec\tf\n", filename.c_str());


		bool running = true;
		bool singleStep = false;
		double est_energy_frame = 0, est_tot_energy = 0, allworktime = 0;

		UnsupervisedHeatmap hm;
		hm.init(base_img_sz, 2);
		LocationLogger ll;
		ll.init(base_img_sz, 2);

		BayesianAnomalyDetector bayesDet(&hm, &ll);
		bool showHeatMaps = false;

		if (load_clusters_at_start){
			ca.readClusters(load_cluster_path);
			stringstream set_ss;
			set_ss << "ilids_pv" << ilids_scene << "_";
			hm.read(heatmap_path, set_ss.str(), "online");
			ll.read(heatmap_path, set_ss.str(), "locationlogger");
		}

		runTimer.workBegin();
		while (running){
			frameTimer.workBegin();
			int frames2skip = 0;

			//get frame
			if (using_folder)
				running = imagefolder.getFrame(&frame, 5);
			else {
				if (globalAttemptRealtime){
					//get n frames to skip
					frames2skip = cvCeil((elapsed + 0.5) / source_frame_time_ms) - 1;
					//cout <<"frames2skip: " << frames2skip <<"\n";
					if (singleStep)
						frames2skip = min(2, frames2skip);
					skippedFrameCounter += frames2skip;
				}

				for (int i = 0; i < frames2skip; i++){
					vc.grab();//vc >> frame;
					fc++;
					incrementFrameCounter(&d_min, &d_sec, &d_frame, source_fps);
				}
				vc >> frame; //one more actual one
				fc++;
				incrementFrameCounter(&d_min, &d_sec, &d_frame, source_fps);

				if (frame.rows == 0){  //if invalid frame/end of video
					running = false;
					break;
				}
			}
			//set up framestore for different types
			heisenframe.setFrame(frame);

			//convert function list to vector
			vector<int> chosenFunctions(functions.begin(), functions.end());
			//dont count this as part of work done; it's display
			if (globalMergeDetections && globalDrawBase){
				//warp image to get ground plane: can be cpu or gpu for power/speed
				prioritisedWarpPerspective(&heisenframe, base_img, H_cam2base, base_img_sz, globalPriorities.get());
			}
			workTimer.workBegin();
			//do late tracker and clustser update for skipped frames, but do it inside workTimer so we can count it
			int skipcnt = frames2skip;
			while (globalMergeDetections && skipcnt > 0) {
				app.runTrackersOnDetections(dummyDetections, base_img, 0, false); //update kfs
				ca.update(app.kfs); //update clusters
				anomaly_level = ca.rateAnomalyAndPlotClusters(base_img, &colours, app.kfs, &bayesDet, false);
				skipcnt--;
			}
			//need to choose implementations BEFORE checking which func
			chooseBestMappingFromAlgorithm(candidates, chosenFunctions, globalPriorities.get(),
				&mapping, &solutionDetails);

			//process frame with normal detectors
			processFrame(&heisenframe, mapping, initialDetections);

			if (globalMergeDetections){
				//using same instances for full-frame and patch detectors so save params first then set new ones
				int car_gr_thresh = motionProcessor.car->getInt("gr_threshold");
				double car_hit_thresh = motionProcessor.car->getDouble("hit_threshold");
				motionProcessor.car->set("gr_threshold", 3);
				motionProcessor.car->set("hit_threshold", 0.75);
				int ped_gr_thresh = motionProcessor.ped->getInt("gr_threshold");
				double ped_hit_thresh = motionProcessor.ped->getDouble("hit_threshold");
				motionProcessor.ped->set("gr_threshold", 3);
				motionProcessor.ped->set("hit_threshold", 0.5);

				app.improveDetections(&initialDetections, &improvedDetections, &heisenframe);
				//restore params
				motionProcessor.ped->set("gr_threshold", ped_gr_thresh);
				motionProcessor.ped->set("hit_threshold", ped_hit_thresh);
				motionProcessor.car->set("gr_threshold", car_gr_thresh);
				motionProcessor.car->set("hit_threshold", car_hit_thresh);

				if (singleStep)
					imshow("Detections", img_to_show);
				//track. output is in kfs
				app.runTrackersOnDetections(improvedDetections, base_img, frames2skip, globalDrawBase);
				//update heatmaps
				hm.updateAll(&app.kfs, 5, false);
				ll.updateAll(&app.kfs, 5, false);
				//cluster
				ca.update(app.kfs);
				//rate anomalies
				anomaly_level = ca.rateAnomalyAndPlotClusters(base_img, &colours, app.kfs, &bayesDet, globalDrawBase, globalDrawClusters);
				app.deleteLostTrackers(); //dont delete these until the clustering algorithm has seen them
			}
			else{
				improvedDetections.clear();
				anomaly_level = 0;
			}

			//no idea where to put this: automatic priority calc at end of frame
			if (globalAutoPrioritise)
				autoSetPriorities(anomaly_level);

			elapsed = workTimer.workEnd();
			allworktime += elapsed;
			//////////////
			//everything below here is display/gui stuff
			if (globalMergeDetections && globalDrawBase){
				//legend for base
				putText(base_img, "???", Point(10, 20), FONT_HERSHEY_SIMPLEX, 0.8, colours.getColoursFromID(OBJECT_UNDETERMINED), 2);
				putText(base_img, "PED", Point(10, 40), FONT_HERSHEY_SIMPLEX, 0.8, colours.getColoursFromID(OBJECT_PED), 2);
				putText(base_img, "CAR", Point(10, 60), FONT_HERSHEY_SIMPLEX, 0.8, colours.getColoursFromID(OBJECT_CAR), 2);

				imshow("base", base_img);
			}
			img_to_show = frame;
			drawDetections(initialDetections, img_to_show, debug, &colours, 3, 1, fp, fc);
			drawDetections(improvedDetections, img_to_show, debug, &colours, 3, 3, fp, fc);

			if (globalDrawTracksOnCam){
				app.transformTracksToCam(&trackedPoints);
				for (size_t j = 0; j < trackedPoints.size(); j++)
					circle(img_to_show, Point(trackedPoints[j].x - 5, trackedPoints[j].y - 5),
					10, colours.getColoursFromID(trackedPoints[j].z), 2);
			}
			if (globalDrawClusters && globalDrawTracksOnCam){
				vector<vector<Point> >camClusters;
				vector<int>clusterLabels;
				ca.transformClusterstoCam(&camClusters, &clusterLabels, app.H_base2cam);
				for (size_t i = 0; i < camClusters.size(); i++){
					Scalar paint = colours.getColoursFromID(clusterLabels[i]);
					for (size_t j = 1; j < camClusters[i].size(); j++){
						line(img_to_show, (camClusters[i])[j - 1], (camClusters[i])[j],
							paint);
					}
				}
			}

			//print anomaly level
			char buf[100]; sprintf(buf, "Anomaly Level:%3.1f", anomaly_level);
			putText(img_to_show, buf, Point(10, 20), FONT_HERSHEY_SIMPLEX, 0.8, CV_RGB(255, 0, 0), 2);


			int y_print0 = frame.rows - 40;
			int y_print = y_print0 - 60;
			//draw legend text directly on output image
			size_t j = 0;
			for (j; j < mapping.size(); j++){
				putText(img_to_show, mapping[j].obj->name(), Point(5, y_print - 30 * j),
					FONT_HERSHEY_SIMPLEX, 0.8, colours.getColoursFromID(mapping[j].obj->getID()), 2);
			}

			//now clear detections for this frame
			initialDetections.clear();
			//allow callbacks next frame
			globalPriorities.enableCallback = true;


			float baseline = 147.f;
			est_energy_frame = (solutionDetails.power - baseline) * elapsed * .001; //power  * time (ms)
			est_tot_energy += est_energy_frame;
			char timestr[100], buf0[100], buf1[100];
			sprintf(timestr, "%02d:%02d:%02df | Frame #%04d | Frameskip %d |", d_min, d_sec, d_frame, fc, frames2skip);
			putText(img_to_show, timestr, Point(5, y_print0 - 30), FONT_HERSHEY_SIMPLEX, 0.6, cvScalar(0), 2);
			sprintf(buf0, "Est. Power %3.0fW | Est. Time %3.1fms | Energy %3.1fJ |", solutionDetails.power, solutionDetails.time, est_energy_frame);
			putText(img_to_show, buf0, Point(5, y_print0), FONT_HERSHEY_SIMPLEX, 0.8, cvScalar(0), 2);
			sprintf(buf1, "Work time %5.1fms | Frame time %5.1fms", elapsed, frameTimer.workEnd());
			putText(img_to_show, buf1, Point(5, y_print0 + 30), FONT_HERSHEY_SIMPLEX, 0.8, cvScalar(0), 2);
			if (!run_evaluation){
				cout << timestr << buf0 << buf1 << "\n";
			}

			//show anom detections on cam and/or dump to file 
			app.transformAnomTracksToCam(&anomDets);
			Scalar white = CV_RGB(255, 255, 255), red = CV_RGB(255, 0, 0);
			int anom_age_thresh = 10 * 25;//75; //10*fps
			for (size_t j = 0; j<anomDets.size(); j++){
				rectangle(img_to_show, Rect(anomDets[j].x - 20, anomDets[j].y - 20, 40, 40),
					anomDets[j].duration>anom_age_thresh ? red : white, 3);
				if (anomDets[j].duration > anom_age_thresh){
					//check if we've already logged this
					std::vector<int>::iterator r = std::find(loggedAnomalyIDs.begin(), loggedAnomalyIDs.end(), anomDets[j].id);
					if (r == loggedAnomalyIDs.end()){	 //if not, print and log
						fprintf(anom_fp, "%3d\t%3d\t%5d\t%2d\t%2d\t%2d \t%4d\n", anomDets[j].x, anomDets[j].y,
							fc, d_min, d_sec, d_frame, anomDets[j].id);
						loggedAnomalyIDs.push_back(anomDets[j].id);
						imwrite(snapshotpath.string() + boost::lexical_cast<std::string>(anomDets[j].id) + ".png", img_to_show);
					}
				}
			}


			imshow("Detections", img_to_show);
			inputKey = cvWaitKey(1);
			if (inputKey != -1){
				stringstream set_ss; //used by r and w
				switch (inputKey){
				case 'o':
				case 'k':
					processInput(inputKey, mapping, functions);
					break;
				case 'w': //write everything
					ca.writeClusters(save_cluster_path);
					set_ss << "ilids_pv" << ilids_scene << "_";
					hm.write(heatmap_path, set_ss.str(), "online");
					ll.write(heatmap_path, set_ss.str(), "locationlogger");
					break;
				case 'r': //read everything
					ca.readClusters(load_cluster_path);
					set_ss << "ilids_pv" << ilids_scene << "_";
					hm.read(heatmap_path, set_ss.str(), "online");
					ll.read(heatmap_path, set_ss.str(), "locationlogger");
					break;
				case 's'://inf loop until space pushed
					singleStep = true;
					inputKey = cvWaitKey(0);
					break;
				case 'q'://q to quit
				case 27://esc to quit
					running = false;
					break;
				case 't':
					printf("%02d:%02d:%02df | Frame #%05d | Source %f| Wall %f\n",
						d_min, d_sec, d_frame, fc, fc*source_frame_time_ms, runTimer.workEnd());
					break;
				}
			}
			else {
				if (singleStep){
					imshow("Detections", img_to_show);
					inputKey = cvWaitKey(0);
					if (inputKey == 32){ //space, else wait for any key
						singleStep = false; //break out of single step
					}
				}
			}

			if (debug)
				fflush(fp);

			//elapsed = frameTimer.workEnd(); // comment out to only use worktime in calulation of nframes to skip
		}
		fflush(anom_fp);
		fclose(anom_fp);

		printf("%02d:%02d:%02df | Frame #%05d | Source %f| Wall time %f |Total work time %f|Energy %f |Avg work power %f| %f work ratio| Test mode %s\n",
			d_min, d_sec, d_frame, fc, fc*source_frame_time_ms, runTimer.workEnd(), allworktime, est_tot_energy, 1000 * est_tot_energy / allworktime,
			allworktime / runTimer.workEnd(),
			testName.c_str());
		printf("%d skipped frames of %d total, ratio %f\n", skippedFrameCounter, fc, (skippedFrameCounter*1. / fc));
		fprintf(fp, "%02d:%02d:%02df | Frame #%05d | Source %f| Wall time %f |Total work time %f|Energy %f |Avg work power %f| %f work ratio| Test mode %s\n",
			d_min, d_sec, d_frame, fc, fc*source_frame_time_ms, runTimer.workEnd(), allworktime, est_tot_energy, 1000 * est_tot_energy / allworktime,
			allworktime / runTimer.workEnd(), testName.c_str());
		fprintf(fp, "%d skipped frames of %d total, ratio %f\n", skippedFrameCounter, fc, (skippedFrameCounter*1. / fc));
		fflush(fp);
		fclose(fp);
		if (save_clusters_on_exit){
			stringstream set_ss; //used by r and w
			set_ss << "ilids_pv" << ilids_scene << "_";
			hm.write(heatmap_path, set_ss.str(), "online");
			ll.write(heatmap_path, set_ss.str(), "locationlogger");
			ca.writeClusters(save_cluster_path);
		}
	}

#ifdef HAVE_BOOST
	catch (const filesystem_error& ex) { return std::cout << "error: " << ex.what() << endl, 1; }
#endif
	catch (const Exception& e) { return std::cout << "error: " << e.what() << endl, 1; }
	catch (const exception& e) { return std::cout << "error: " << e.what() << endl, 1; }

	catch (...) { return cout << "unknown exception" << endl, 1; }
		cout << "exit main" << endl;
	return 0;
}
