#ifndef CB_TRACKING_APP
#define CB_TRACKING_APP 1

//note this is in a separate header file to the object tracker <TrackableDetection> and <ObjectTracker>
class TrackerApp{
public:
	TrackerApp(cv::Mat H_cam2base_, std::map<int, int> impl2obj_, std::map<int, int> impl2algo_, PatchProcessor hog_patch_,
		cv::Point validTL, cv::Point validBR);
	void runTrackersOnDetections(std::vector<Detection>detectedCands, cv::Mat& base_img, int frames2skip, bool draw = true);
	//call separately to get output tracks
	//doesnt really return 3d, its xy and type
	void transformTracksToCam(std::vector<cv::Point3i>* trackedPoints);
	int transformAnomTracksToCam(std::vector<AnomalousDetection>* anomDets);
	void improveDetections(cv::vector<Detection>* initialDetections, cv::vector<Detection>*improvedDetections, HeisenFrame* hf);
	int deleteLostTrackers();

	std::list<ObjectTracker> kfs; //list of kalman filters with object info
	cv::Mat H_cam2base; //homography for camera to base plane
	std::map<int, int> impl2obj; //map of implementation ids to object ids
	std::map<int, int> impl2algo;// map of implmentation ids to algorithm ids

	PatchProcessor hog_patch; //pointers to implementations for object detectors

	//parameters
	float thresh_det2track_, thresh_det2track; //euclidean match dist
	float loose_thresh_det2track; //consider for elliptical matching if within this euclidean dist
	//ellipse eccentricity, assuming thresh_det2track is semimajor axis length
	float eccentricity;
	static const bool doSimpleD2TMatching = false; //if true,only match on euclidean distance (ie circular)

	int max_obj_area; //nothing bigger than this allowed = (int) (imsz.area()*0.25);
	float motionBorder; //50px round an object
	float motionScale;  // zoom into motion dets by this amount

	//kill trackers under the following conditions:
	int track_lost_hard_thresh; //kill after this unless stationary
	int track_lost_soft_thresh; //kill after this unless it's being used
	int track_lost_speed_thresh; // speed. if above this then physically impossible -- kill

	//merge nearby tracks under specific conditions
	void mergeNearbyTracks(void);
	int merge_tracks_age_thresh;
	int merge_tracks_dist_thresh;
	float merge_tracks_theta_thresh;

	int lost_xmin, lost_xmax, lost_ymin, lost_ymax;

	cv::Mat H_base2cam; //this is the inverse of base2cam, set in constructor

	float motion_motion_overlap_thresh; //motion <- >motion overlap, was 0.9
	float motion_detection_overlap_thresh; //motion <-> full-frame detecteion overlap, was 0.9
	float intersection_thresh; //final detection intersection thresh  was 0.45

private:
	inline bool isClippedRectWithinWindow(cv::Rect det, cv::Size window){
		if (det.x > 0 && det.x < (window.width - 1) && det.y>0 && det.y < (window.height - 1))
			return true; //if properly within window dont care about zero size
		else //starts somehwere on outer edge. NOT testing if falls outside window cause clipRect has already done that
			//if nonzero size
			if (det.width > 0 || det.height > 0)
				return true;
			else
				return false;
	}
};

#endif //CB_TRACKING_APP
