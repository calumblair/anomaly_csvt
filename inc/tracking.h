#ifndef CB_TRACKING
#define CB_TRACKING

#define CV_PI_F ((float) CV_PI)

//calculate properties of 2 bounding boxes
class BBprops{
public:
	//bounding box intersection code from pascal VOC
	inline static float intersection(cv::Rect a, cv::Rect b){
		float inL = (float)std::max(a.x, b.x), inT = (float)std::max(a.y, b.y);
		float inR = (float)std::min(a.x + a.width, b.x + b.width);
		float inB = (float)std::min(a.y + a.height, b.y + b.height);
		float iw = inR - inL + 1;
		float ih = inB - inT + 1;
		float res = 0.f;
		if (iw > 0 && ih > 0){
			// compute area of intersection / area of bb B
			res = iw*ih / b.area();
		}
		return res;
	};

	//overlap of rects rather than intersection
	inline static float overlap(cv::Rect a, cv::Rect b){
		float inL = (float)std::max(a.x, b.x), inT = (float)std::max(a.y, b.y);
		float inR = (float)std::min(a.x + a.width, b.x + b.width);
		float inB = (float)std::min(a.y + a.height, b.y + b.height);
		float iw = inR - inL + 1;
		float ih = inB - inT + 1;
		float res = 0.f;
		if (iw > 0 && ih > 0){
			// compute overlap as (area of intersection) / (area of union)
			float u = a.area() + b.area() - iw*ih;
			res = iw*ih / u;
		}
		return res;
	};
};


//motion bb with necessary details to allow roi/patch grabbing later
class MotionBB{
public:
	int l, r, t, b, w, h;
	int candidates; //test with ped or car?
	Detection det;
	MotionBB(int l_, int r_, int t_, int b_, int w_, int h_, int candidates_, Detection det_){
		l = l_; r = r_; t = t_; b = b_; w = w_; h = h_; candidates = candidates_; det = det_;
	}
};


//a detection which is a candidate for tracking will also have a centroid point calculated
struct TrackableDetection : public Detection{
public:
	/*Detection det;*/
	cv::Point2f centre;
};


//anomalous detection passed back to main() for display and logging.
//nothing to do with the detection struct
struct AnomalousDetection{
	int x; //cam plane x centre
	int y; //cam plane y centre
	int id;//id of original tracker
	int duration; //anomaly age in frames
};


class ObjectTracker : public cv::KalmanFilter{
public:
	ObjectTracker(cv::Rect bb_, int objectType_); //initialise tracker for xy detections in 2d space, get id, store bb and type
	const cv::Mat& predict(const cv::Mat& control = cv::Mat()); //predict then update age, lastseen
	const cv::Mat& correct(const cv::Mat& measurement); //correct and update hits, zero lastseen
	const cv::Mat& correct(const TrackableDetection& det); //correct and update hits, zero lastseen, check object type.
	int lastseen; //number of frames when a detection was last seen
	bool lost;

	int hits; //number of objects/patches matched to tracker (can be >age)
	int age; //overall age of tracker

	cv::Rect bb; //base plane rectangle
	int type; //object type being tracke
	int id; //unique id

	//cluster-specific extensions
	float anom;  //anomaly measure of object
	bool unseen; // true if not yet added to a cluster
	std::vector<cv::Point2f> record; //unseen points are stored here

	int anom_counter; // counter for number of frames in which object is behaving anomalously

	static const int noise = 2000;

	inline float motionBearingDegrees(void){
		float a = atan2f(statePost.at<float>(3), statePost.at<float>(2))*180.f / CV_PI_F + 90.f;
		return a < 0 ? a + 360 : a;
	}
	inline float motionMagnitude(void){
		return sqrt(statePost.at<float>(2)*statePost.at<float>(2)
			+ statePost.at<float>(3)*statePost.at<float>(3));
	}
};


class TrackerIDList{
public:
	static int getNew(void)	{ return getInstance().getNew_(); }; //get new id
	static int getCount(void)	{ return getInstance().getCount_(); };//net number of ids handed out
	static void init(void)		{ getInstance().init_(); }; //zero id counter
private:
	TrackerIDList(){ count = 0; };
	TrackerIDList(TrackerIDList const&); //hide this
	void operator=(TrackerIDList const&);

	static TrackerIDList& getInstance(){ static TrackerIDList instance; return instance; };
	int getNew_(void)	{ return count++; };
	int getCount_(void)	{ return count; };

	void init_(void)	{ count = 0; };

	int count;
};


//use to determine if we can match type of a to type of b
inline bool are_different_object_types(int a, int b){
	return ((a == OBJECT_PED && b == OBJECT_CAR) || (a == OBJECT_CAR && b == OBJECT_PED));
}


//conversion of svm scores to [0-1] proabibility
inline double convertConfidenceToProb(double conf, bool isCar);


//sort comparison function  for tracked detection sorting
inline bool compareTrackableDetections(TrackableDetection i, TrackableDetection j);

//euclidean distance beween points for anomaly measurment
inline float euclideanDistance(float x0, float y0, float x1, float y1){
	return sqrtf(1.0f*(x0 - x1)*(x0 - x1) + 1.0f*(y0 - y1)*(y0 - y1));
}

inline float euclideanDistance(cv::Point2f p1, cv::Point2f p2){
	return sqrtf((p1.x - p2.x)*(p1.x - p2.x)*1.0f + (p1.y - p2.y)*(p1.y - p2.y)*1.0f);
}

//return rect clipped to frame dimensions. zero-width and height outputs are allowed if they fall
//wholly outside the frame
inline cv::Rect clipRect(cv::Rect roi_, cv::Size frame){
	int x_ = MAX(0, MIN(frame.width - 1, roi_.x));
	int y_ = MAX(0, MIN(frame.height - 1, roi_.y));
	int w_ = MAX(0, MAX(0, MIN(roi_.x + roi_.width, frame.width)) - x_);
	int h_ = MAX(0, MAX(0, MIN(roi_.y + roi_.height, frame.height)) - y_);
	return cv::Rect(x_, y_, w_, h_);
}

#define BANK_ST 1
#endif
