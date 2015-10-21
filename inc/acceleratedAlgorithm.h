//acceleratedAlgorithm.hpp - base class
//see notes in algorithmWrapper.cpp

#ifndef CB_ACC_ALGORITHM
#define CB_ACC_ALGORITHM 1
#include "heisenFrame.h"

//This is a detection object. All algorithms must generate or process detections 
//in this format. 
struct Detection{
	cv::Rect bb; //bounding box, x y w h
	double score; //confidence from detector
	int source; //ID of detector. Use algoList.getAlgorithmID(source) to get the name (can probably use this as a type too)
	Detection(int, int, int, int, double, int);
	Detection(cv::Rect, double, int);
	Detection();
};

//This is an implementation object. All algorithms must have one of these. it's used for selection at runtime
class Implementation{
public:
	Implementation(void){ Implementation(0, 0., 0., 0., 0, 0, 0, 0); };
	Implementation(bool valid_, float time_, float power_, float accuracy_,
		int algorithm_, int id_, int resources_ = 0, bool isFpgaExclusive_ = false){
		time = time_;
		power = power_;
		accuracy = accuracy_;
		algorithm = algorithm_;
		id = id_;
		resources = resources_;
		isFpgaExclusive = isFpgaExclusive_;
		valid = valid_;
	};
	//performance characteristics
	float time; //milliseconds for processing
	float power; //watts as measured for system power consumption
	float accuracy; //detectors: miss rate at 10^-4 fppw
	//algorithm characteristics
	int algorithm; //will be one of several #defined types
	//implementation
	bool valid; //is available, enabled, and initialised?
	int resources; //uses fpga, uses gpu? bitmasked.
	//unique label?
	int id; //get this from the AcceleratedAlgorithm
	bool isFpgaExclusive; //does this implementation require exclusive FPGA access?
};

extern "C"  int getNumAlgorithms(bool debug);
///////////////////////////////////////base class - lots of virtual functions
class AcceleratedAlgorithm : public cv::Algorithm
{
public:
	virtual void detect(const cv::Mat frame, std::vector<Detection>& detections);
	virtual void detect(HeisenFrame* hf, std::vector<Detection>& detections);
	virtual cv::AlgorithmInfo* info() const;
	AcceleratedAlgorithm();
	int getID();
	std::string getID(int);
	Implementation implementation;
protected:
	//lets give it labels
	bool initialised;//set once initialised
	bool available;//probably unnecessary
	bool enabled;//probably unnecessary
	int id; //unique ID for this algorithm implementation
};


//easy way of passing pointers to processing algorithms to the improveDetections funcntion for patch processing
//of motion-generated bboxes.
struct PatchProcessor{
	cv::Ptr<AcceleratedAlgorithm> car; //detector for car candidates
	cv::Ptr<AcceleratedAlgorithm> ped; //detector for ped candidates
};


//now we define all the different kinds of algorithm
#define ALGORITHM_HOG	100
#define ALGORITHM_MOG	101
#define ALGORITHM_HOG_CAR	105
#define ALGORITHM_MOTION_ONLY 99

#define ALGORITHM_SOLUTION 200

#define RESOURCES_GPU	1
#define RESOURCES_FPGA	2

#define OBJECT_PED 1
#define OBJECT_CAR 2
#define OBJECT_UNDETERMINED 0

////////////////////////////////////priorities stuff
#define PRIORITY_TIME		900
#define PRIORITY_POWER		901
#define PRIORITY_ACCURACY	902


struct simplepriorities{
	int accuracy; //weighting to give accuracy, 1-10
	int latency;
	int power;
};


class Priorities : simplepriorities {
public:
	Priorities();
	simplepriorities get(void); //return priorities
	void get(int* accuracy, int* latency, int* power);
	bool set(simplepriorities priorities); //set via simplepriorites. this might be dumb
	void set(int accuracy, int latency, int power);
	void callback(int trackbarValue, void* trackerType);
	int getNumCredits(){ return numCredits; };
	int sum(void);
	bool enableCallback; //only allow callbacks once per frame or we start recursing
	friend void autoSetPriorities(float overall_anomaly_level); //global external func
protected:
	static const int numCredits = 10;
	void reWeight(int newValue, int whichValue);
};

void myRectangleGroup(std::vector<Detection>* src, std::vector<Detection>* dst, cv::Size imsz);
#endif
