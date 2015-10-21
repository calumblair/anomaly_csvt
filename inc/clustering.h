#ifndef CB_CLUSTERING
#define CB_CLUSTERING 1

class Cluster{
public:
	std::vector<cv::Point2f> pt;
	std::vector<float> sigma;
	int parent;
	std::vector<std::pair<int, int> > children; //first is id of child, second is frequency

	int type;
	int lastseen;
	int transits;
	//construct from several points
	Cluster(std::vector<cv::Point2f> pt_, std::vector<float> sigma_, int parent_,
		std::vector<std::pair<int, int> > children_, int type_, int lastseen_, int transits_){
		pt = pt_; sigma = sigma_; parent = parent_; children = children_;
		type = type_; lastseen = lastseen_; transits = transits_;
	}

	//construct from single point
	Cluster(cv::Point2f pt_, float sigma_, int parent_,
		std::pair<int, int> children_, int type_, int lastseen_, int transits_){
		pt.push_back(pt_); sigma.push_back(sigma_); parent = parent_;
		children.push_back(children_); type = type_;
		lastseen = lastseen_; transits = transits_;
	}

	//default constructor, only called from resize
	Cluster(){ printf("default cluster constructor called\n"); }

	void write(cv::FileStorage& fs) const; //serialise and write clusters to disk
	void read(const cv::FileNode& node); //read in clusters from file
};


class ClusteringApp{
public:
	ClusteringApp();
	void update(std::list<ObjectTracker>& kfs);
	float rateAnomalyAndPlotClusters(cv::Mat& base_img, DisplayColourMap* colours,
		std::list<ObjectTracker>& kfs, BayesianAnomalyDetector* bayes, bool draw = true, bool drawClusters = true);
	void transformClusterstoCam(
		std::vector<std::vector<cv::Point> >* camClusters,
		std::vector<int>* clusterTypes, cv::Mat H_base2cam);

	//parameters from Piciarelli paper
	float win_width;
	float learning_alpha;
#define SIGMA_DEFAULT (20.f)

	// selected parameters
#define ROOT_NODE (-1)
#define DISABLED_NODE (-2)
#define SIGMA_DEFAULT_ROOT_NODE (20.f)
	int root_cluster_search_depth;

	//anomaly thresholds
	int anom_transit_thresh_root_node;
	int anom_transit_thresh_child_node;
	float anom_freq_thresh;
	float anom_overall_thresh;

	//lost thresholds
	int lost_thresh;
	int max_cluster_age;
	float cluster_age_scaler; //prune if cluster last seen more than length(cluster) * scaler + max_cluster_age ago
	int min_tracker_age; //has to be this old before we can work with it.

	size_t nIDs; //number of unique tracker IDs
	std::vector<std::pair<int, int> > id2cl; //map of tracker ids to clusters and points within
	std::vector<Cluster> clusters;

	int makeCluster(cv::Point2f pt, float sigma, int parent, int objType);

	void writeClusters(std::string filename, std::string filetype = "xml");
	void readClusters(std::string filename, std::string filetype = "xml");

	int num_match_errors;// number of times we are no longer able to match 
	//a tracker point once it is already associated to a cluster.
	int num_search_errors_nonfatal; //same but inside internal search

private:
	enum SEARCHTYPE { normal_search, root_search, forward_only_search };
	inline size_t getNumChildren(int i);
	inline int getTransits(int i); //return number of transits of that node
	inline void updateTransitCount(int i); //increments cluster transit counts from leaf to root.

	inline float getClusterAnomalousness(int i);
	inline void markForDeletion(int i);
	/*inline*/ int getTreeAge(int i); //recursive
	inline void updateIDsFromOldClusters(int oldidx, int newidx, int newpt); //change link of each tracker id from old cluster to new one
	inline void incrementChildFreqs(int parent, int child);
	inline bool updateSingleClusterPoint(cv::Point2f pt, int currentCluster, int currentStage, int id, int type);
	int getTreeLength(int i); //return length of an entire tree, when given a leaf node. recursive.
	bool isParentTypeDifferent(int id, int newtype); //return true if anything in tree ending with [id] is different from [type]

	void split(int stage, int cluster, std::list<ObjectTracker>& kfs);

	inline bool forwardSearch(cv::Point2f pt, int sc, int tl, int id, int type, int* bestCluster, int* bestStage,
		float* minDist, enum SEARCHTYPE searchType = normal_search, int sp = 0);
	void internalForwardSearch(cv::Point2f pt, int pos_lower, int pos_upper, int sc, int* bc, int* bp, float* md, int type);
	inline void prune();
	bool concatenate(int i);
	void deleteZeroLengthClusters(int i);
	inline void forgetTracker(int objectID, std::list<ObjectTracker>::iterator i_kf, int cc);

};

#endif //CB_CLUSTERING
