#ifndef CB_BAYES_ANOMALY
#define CB_BAYES_ANOMALY 1

class BayesianAnomalyDetector{
public:
	//set up maxconf and minconf and heatmap pointers
	BayesianAnomalyDetector(UnsupervisedHeatmap* hm_, LocationLogger* ll_) :
		maxconf(0.99f), minconf(0.01f)
	{
		hm = hm_; ll = ll_;
	}

	//calculate a probability measure that an object of type [type] at location [roi]
	//and with a velocity [vx, vy] is similar to existing motion described by hm.xmap and hm.ymap.
	//returns a matrix populated with values for each pixel.
	//supposed to vary between 0 and 1 but clamped to min and max conf set in constructor.
	void calcHamProbability(const cv::Mat mean, cv::Mat& prob_out, float measurement);

	//does what it says on the tin. given a background probability of an anomaly existing [pAnomPrior],
	//such an anomaly will have a probability of backgroundAnom(=1) everywhere. (ie probabilty of anomaly is pAnomPrior and 
	//can occur anywhere). ham (not-anomaly) probability is per-pixel and is passed in. 
	//presence data is per-pixel and describes number of times a detection has been seen in that pixel (frequency).
	//strength is the frequency threshold below which pAnomPrior is weighted more highly than the ham value, usually 3.
	//this is supposed to return smooth number between 0 and 1 but tends to slam from one extreme to the other arbitrarily.
	float calcBayesianProbability(const cv::Mat presence, const cv::Mat ham, float pAnomPrior, float backgroundAnom, float strength);

	void calcAnomalyScores(cv::Rect roi_, float vx, float vy, int objtype,
		float* anom_score_x, float* anom_score_y,
		float pAnomPrior = 0.29f, float backgroundAnom = 1.f, float strength = 50.0f);

	void generateRho(cv::Rect roi_, float vx, float vy, int objtype, float* anom_score_rho, float* anom_score_theta,
		float pAnomPrior = 0.29f, float backgroundAnom = 1.f, float strength = 50.0f);

	void updateAll(std::list<ObjectTracker>* trackers, int age_thresh, bool alsoUpdateTrackers);

	UnsupervisedHeatmap* hm;
private:
	const float maxconf;// 0.99f;
	const float minconf;// 0.01f;
	LocationLogger* ll;
};
#endif
