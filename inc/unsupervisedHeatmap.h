#ifndef CB_UNSUPERVISED_HEATMAP
#define CB_UNSUPERVISED HEATMAP 1

class UnsupervisedHeatmap{
public:
	UnsupervisedHeatmap();
	float learningRate;
	cv::Mat x_ped_map, x_car_map;
	cv::Mat y_ped_map, y_car_map;
	cv::Size mapSize, origMapSize;
	int scale;// fraction to shrink map by, eg 2 will be 1/2w and 1/2 height of input
	void init(cv::Size mapSize_, int scale_);

	void updateAll(std::list<ObjectTracker>* trackers, int age_thresh, bool alsoUpdateTrackers = false); //do all in one step
	//if alsoupdatetrackers then update each tracker with the new anomaly level
	void getRegionMean(cv::Rect roi, int type, float* xmean, float* ymean, float* xvar, float* yvar);

	//used by bayesian stuff. return raw data scaled by <scale> size. pass in clipped rect.
	void getRegionData(cv::Rect clipped_roi, int type, char direction, cv::Mat& outmat);

	void write(std::string heatmap_path, std::string setname, std::string mapname);
	void read(std::string heatmap_path, std::string setname, std::string mapname);

protected:
	inline cv::Rect downScaleRect(cv::Rect r_, int scale);
private:
	void updateRegion(cv::Rect unclipped_roi, float xval, float yval, int type); // do single, called from updateAll

};


class LocationLogger :
	public UnsupervisedHeatmap
{
public:
	void updateAll(std::list<ObjectTracker>* trackers, int age_thresh, bool alsoUpdateTrackers = false);
private:
	void updateRegion(cv::Rect roi, float xval, float yval, int type); // do single
};

#endif
