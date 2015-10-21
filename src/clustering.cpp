/*my reimplementation of my matlab reimplementation of "on-line trajectory
clustering for anomalous events detection", piciarelli and foresti, 2006.
calum blair, june 2013*/
#define __STDC_LIMIT_MACROS //used for INT32_MAX
#include "stdafx.h"
#include "acceleratedAlgorithm.h"
#include "tracking.h"
#include "display.h"
#include "unsupervisedHeatmap.h"
#include "bayesianAnomalyOracle.h"
#include "clustering.h"
#include "trackingApp.h"


using namespace std;
using namespace cv;


//#define CLUSTER_DEBUG 1
#define CLUSTER_DEBUG 0

#ifdef LINUX
//gives loads of warnings but compiles OK
#define debug_printf
#else
//following line needs C99
#define debug_printf(fmt, ...) \
	do { if (CLUSTER_DEBUG) fprintf(stderr, fmt, __VA_ARGS__); } while (0)
#endif

ClusteringApp::ClusteringApp(){
	win_width = 0.4f;
	learning_alpha = 0.05f;

	root_cluster_search_depth = 15;
	anom_transit_thresh_root_node = 2;
	anom_transit_thresh_child_node = 6;
	anom_freq_thresh = 0.35f;
	anom_overall_thresh = 15;

	lost_thresh = 10;
	max_cluster_age = 5;
	cluster_age_scaler = 0.5f;

	min_tracker_age = 5;

	num_search_errors_nonfatal = 0;
	num_match_errors = 0;
}

//These write and read functions must be defined for the serialization in FileStorage to work
static void write(FileStorage& fs, const std::string&, const Cluster& x)
{
	x.write(fs);
}
static void read(const FileNode& node, Cluster& x, const Cluster& default_value = Cluster()){
	if (node.empty())
		x = default_value;
	else
		x.read(node);
}


void Cluster::write(FileStorage& fs) const{
	fs << "{" << "pt" << pt;
	fs << "sigma" << sigma << "parent" << parent;

	//FileStorage doesnt know how to deal with a vector<pair<>>
	//so unroll it
	vector<int>unrolledChildren(children.size() * 2, -99);
	for (size_t j = 0; j < children.size(); j++){
		unrolledChildren.at(2 * j) = children.at(j).first;
		unrolledChildren.at(2 * j + 1) = children.at(j).second;
	}
	fs << "unrolledChildren" << unrolledChildren;
	fs << "type" << type << "lastseen" << lastseen << "transits" << transits
		<< "}";
}

void Cluster::read(const FileNode& node) {
	node["pt"] >> pt;
	node["sigma"] >> sigma;
	parent = (int)node["parent"];
	vector<int> unrolledChildren;
	node["unrolledChildren"] >> unrolledChildren;

	assert(unrolledChildren.size() % 2 == 0); //must be even number
	//FileStorage doesnt know how to deal with a vector<pair<>>
	//so reconstruct from unrolled vector<int>
	for (size_t j = 0; j < unrolledChildren.size() - 1; j += 2){
		children.push_back(pair<int, int>(
			unrolledChildren.at(j), unrolledChildren.at(j + 1)));
	}
	type = (int)node["type"];
	lastseen = (int)node["lastseen"];
	transits = (int)node["transits"];
}


void ClusteringApp::writeClusters(string filename, string filetype){
	FileStorage fs(filename + "." + filetype, FileStorage::WRITE);
	cout << "Created cluster file " << filename << "." << filetype << endl;
	int nC = clusters.size();

	fs << "nClusters" << nC;
	fs << "SceneClusters" << "[";
	for (int j = 0; j < nC; j++){
		fs << clusters[j];
	}
	fs << "]";
	fs.release();
	cout << "Write Done, " << nC << " clusters written" << endl;
}

void ClusteringApp::readClusters(string filename, string filetype){
	clusters.clear();
	id2cl.clear();
	FileStorage fs(filename + "." + filetype, FileStorage::READ);
	if (!fs.isOpened())	{
		cerr << "Failed to open " << filename << endl;
		return;
	}
	cout << "Opened cluster file " << filename << "." << filetype << endl;
	int  nC = -1;

	fs["nClusters"] >> nC;
	assert(nC != -1);

	clusters.resize(nC);
	FileNode allClusters = fs["SceneClusters"];
	FileNodeIterator it = allClusters.begin(), it_end = allClusters.end();
	size_t j = 0;
	for (it; it != it_end; it++, j++){
		read((*it), clusters[j]);
	}
	assert(j == nC);
	fs.release();
	cout << "Read Done, " << nC << " clusters read" << endl;
}


//return number of children of a cluster. if 1 child and it's the root or
//disabled node return 0
inline size_t ClusteringApp::getNumChildren(int i){
	size_t n = clusters[i].children.size();
	if (n == 1 && clusters[i].children[0].first <= ROOT_NODE){
		n = 0;
	}
	return n;
}

inline int ClusteringApp::getTransits(int i){
	if (i > ROOT_NODE)
		return clusters[i].transits;
	else
		return 0;
}

int ClusteringApp::makeCluster(cv::Point2f pt, float sigma, int parent, int objType){
	if (sigma == -1) sigma = SIGMA_DEFAULT;
	//skipping type and parent defaults
	int parent_sz = 0;
	int new_id = clusters.size(); // possible off-by-1 error

	clusters.push_back(Cluster(pt, sigma, parent, pair<int, int>(ROOT_NODE, 0), objType, 0, 0));

	if (parent > ROOT_NODE){ //if this cluster is not root node, update parent
		assert(!isParentTypeDifferent(new_id, objType));
		if (getNumChildren(parent) == 0){
			clusters[parent].children[0] = pair<int, int>(new_id, 1);
		}
		else {
			clusters[parent].children.push_back(pair<int, int>(new_id, 1));
		}
		clusters[parent].lastseen = 0;
		parent_sz = clusters[parent].pt.size();
	}

	debug_printf("created child C%d from C%d point %d with [x %d y %d s%f]\n",
		new_id, parent, parent_sz, cvRound(pt.x), cvRound(pt.y), sigma);
	return new_id;
}


void ClusteringApp::updateTransitCount(int i){
	while (i > ROOT_NODE){
		clusters[i].transits++;
		i = clusters[i].parent;
	}
}
//matlab version had both flag and weight, but we will only use weights here
float ClusteringApp::getClusterAnomalousness(int i){
	if (i == ROOT_NODE) //is nonexistent
		return 0;
	else {
		if (clusters[i].parent > ROOT_NODE){ //is child
			int p = clusters[i].parent;
			int p2c = ROOT_NODE, sum = 0;
			for (size_t j = 0; j < clusters[p].children.size(); j++){
				sum += clusters[p].children[j].second;
				if (clusters[p].children[j].first == i){
					p2c = j;
				}
			}
			assert(p2c > ROOT_NODE);
			return 10.f*(1 - (clusters[p].children[p2c].second / (sum + 1.0f)));
		}
		else {
			return 10.f* (1 / (clusters[i].transits + 1.f));
		}
	}
}


void ClusteringApp::markForDeletion(int i)
{
	clusters[i] = Cluster(cv::Point2f(-1, -1), 0, DISABLED_NODE,
		std::pair<int, int>(ROOT_NODE, 0), OBJECT_UNDETERMINED, 0, 0);
	debug_printf("C%d deleted.\n", i);
}


/*if not a leaf node, call recursively. if leaf node, return current age.
overall, returns minimum age of all children of a node*/
int ClusteringApp::getTreeAge(int i){
	int n = getNumChildren(i);
	if (n){
		int age = INT32_MAX;
		for (int j = 0; j < n; j++){
			age = MIN(age, getTreeAge(clusters[i].children[j].first));
		}
		clusters[i].lastseen = age;//update age on the way out
		age = 0;//only a leaf node can be pruned
		return age;
	}
	else {
		//consider pruning leaf node here if very old
		return clusters[i].lastseen;
	}
}

//used when a cluster is set to be one type when an object of another type is also linked to it.
//forget the cluster completely and try again in a few frames
void ClusteringApp::forgetTracker(int objectID, list<ObjectTracker>::iterator i_kf, int cc){
	i_kf->age = -1;
	i_kf->anom = 0;
	i_kf->lastseen = 0;
	i_kf->unseen = true;
	id2cl[objectID] = pair<int, int>(ROOT_NODE, -1);
	debug_printf("forgetting object %d (type %d) as it has incompatible type with C%d (t%d) or parent\n",
		objectID, i_kf->type, cc, clusters[cc].type);
}

//offset refers to how far to shift the points wrt the stored value.
//for zero-length clusters this is parent length.
//for child  concat this is parent length.
void ClusteringApp::updateIDsFromOldClusters(int oldidx, int newidx, int offset){
	//re-link old clusters to new ones
	int newpt = 0;
	for (size_t j = 0; j < id2cl.size(); j++){
		if (id2cl[j].first == oldidx) {
			if (newidx == ROOT_NODE){
				newpt = 0;
			}
			else {
				newpt = offset + id2cl[j].second;
				//assert(newpt < (int)clusters[newidx].pt.size() || (clusters[newidx].pt.size() == newpt && newpt == 0));
			}
			debug_printf("id2cl at %d moved from C%d(%d) to C%d(%d)\n", j,
				oldidx, id2cl[j].second, newidx, newpt);
			id2cl[j] = pair<int, int>(newidx, newpt);

		}
	}
}


//prune dead clusters
void ClusteringApp::prune(){
	for (size_t j = 0; j < clusters.size(); j++){
		clusters[j].lastseen++;
		size_t age = clusters[j].lastseen;
		/*at each stage, also check if this is too old and kill it if
		so. scale by number of transits*/
		if (age > (max_cluster_age + clusters[j].pt.size()*clusters[j].pt.size()*0.1f +
			cluster_age_scaler * (max(1, clusters[j].transits)*max(1, clusters[j].transits))))
		{
			age = getTreeAge(j); //it's old so check children, but dont need to do this every step
			if (age > (max_cluster_age + clusters[j].pt.size()*clusters[j].pt.size()*0.1f + cluster_age_scaler
				* (max(1, clusters[j].transits)*max(1, clusters[j].transits))))
			{
				assert(getNumChildren(j) == 0 && clusters[j].children[0].first == ROOT_NODE);
				debug_printf("pruning C%d (length %d) as not seen for >%d steps. ",
					j, clusters[j].pt.size(), clusters[j].lastseen);

				if (clusters[j].parent > ROOT_NODE){//remove refs to child
					int p = clusters[j].parent;
					//using iterator here so we can erase. 
					if (getNumChildren(p)){
						for (vector<pair<int, int> >::iterator k = clusters[p].children.begin();
							k != clusters[p].children.end();)
						{
							if (k->first == j){
								k = clusters[p].children.erase(k);
							}
							else
								k++;
						}
					}
					if (getNumChildren(p) == 0){
						clusters[p].children.clear();
						clusters[p].children.push_back(pair<int, int>(ROOT_NODE, 0));
					}
					updateIDsFromOldClusters(j, p, clusters[p].pt.size() - 1); //point existing tracks to parent. 
				}
				else {
					updateIDsFromOldClusters(j, ROOT_NODE, DISABLED_NODE);
				}
				markForDeletion(j);
			}
		}
	}
}


//if given a parent ID, look recursively at all children of that ID
//to see if any of them can be concatenated
bool ClusteringApp::concatenate(int j){
	int n = getNumChildren(j);
	if (n > 1){
		return false;
	}
	else {
		if (n == 0) {
			return true;
		}
		else { //only 1 child, not empty
			int childID = clusters[j].children[0].first;
			bool flag = concatenate(childID);
			if (flag){
				int oldLen = clusters[j].pt.size(); //only for printouts
				int childLen = clusters[childID].pt.size();
				clusters[j].pt.insert(clusters[j].pt.end(),
					clusters[childID].pt.begin(), clusters[childID].pt.end());
				clusters[j].sigma.insert(clusters[j].sigma.end(),
					clusters[childID].sigma.begin(), clusters[childID].sigma.end());
				debug_printf("join: child C%d (len %d) to parent C%d at %d. C%d now %d long. ",
					childID, childLen, j, oldLen, j, clusters[j].pt.size());
				//remove ref to child
				clusters[j].children[0].first = ROOT_NODE;
				//still not sure about this
				clusters[j].children[0].second = MAX(clusters[j].children[0].second - 1, 0);
				//touch parent cluster
				clusters[j].lastseen = MIN(clusters[j].lastseen, clusters[childID].lastseen);
				//update transits
				clusters[j].transits = MIN(clusters[j].transits, clusters[childID].transits);
				//delete child
				assert(clusters[childID].children.size() == 1 &&
					clusters[childID].children[0].first == ROOT_NODE);
				markForDeletion(childID);
				updateIDsFromOldClusters(childID, j, oldLen);
			}
			else {
				//only 1 child, not empty, that child has >1 child itself which has failed to concatenate
				int oldLen = clusters[j].pt.size(); //only for printouts
				//move child points to parent
				int childLen = clusters[childID].pt.size();
				clusters[j].pt.insert(clusters[j].pt.end(),
					clusters[childID].pt.begin(), clusters[childID].pt.end());
				clusters[j].sigma.insert(clusters[j].sigma.end(),
					clusters[childID].sigma.begin(), clusters[childID].sigma.end());
				int grandchildren = getNumChildren(childID);
				//double check
				assert(getNumChildren(j) == 1 && clusters[j].children[0].first == childID);
				//remove ref to child
				clusters[j].children.clear();
				for (int k = 0; k < grandchildren; k++){
					clusters[clusters[childID].children[k].first].parent = j; //set parent of grandchildren to j
					clusters[j].children.push_back(clusters[childID].children[k]); //take freq data from grandchildren
				}
				assert(getNumChildren(j) == getNumChildren(childID));

				debug_printf("messy grandchild merge: child C%d (len %d) joined to parent C%d at %d. C%d now %d long. %d grandchildren moved",
					childID, childLen, j, oldLen, j, clusters[j].pt.size(), grandchildren);
				//update transits
				clusters[j].transits = MIN(clusters[j].transits, clusters[childID].transits);
				//touch parent cluster
				clusters[j].lastseen = MIN(clusters[j].lastseen, clusters[childID].lastseen);
				//delete child
				markForDeletion(childID);
				updateIDsFromOldClusters(childID, j, oldLen);
			}
			return flag;
		}
	}
}

//for any zero length clusters c, make c's children g children of c's parent p and delete c
void ClusteringApp::deleteZeroLengthClusters(int c){
	if (clusters[c].pt.size() == 0 && (clusters[c].parent > ROOT_NODE) && getNumChildren(c))
	{
		int p = clusters[c].parent;
		int grandchildren = getNumChildren(c);
		int orig_children = getNumChildren(p);

		//p may have more than 1 child
		bool child_related_to_parent = false;
		size_t child_pos = 0; //find c's position in p
		vector<pair<int, int> >::iterator p_it = clusters[p].children.begin();

		for (child_pos = 0; child_pos < getNumChildren(p); child_pos++, p_it++){
			if (clusters[p].children[child_pos].first == c){
				child_related_to_parent = true;
				break;
			}
		}
		assert(clusters[c].sigma.size() == 0);
		assert(child_related_to_parent);
		//remove ref to child
		clusters[p].children.erase(p_it);

		for (int k = 0; k < grandchildren; k++){
			clusters[clusters[c].children[k].first].parent = p; //set parent of grandchildren to j
			clusters[p].children.push_back(clusters[c].children[k]); //take freq data from grandchildren
		}
		int all_children = getNumChildren(p);
		assert(all_children == grandchildren + orig_children - 1); //-1 for self

		debug_printf("zero-length cluster removal: zero-length node C%d's %d children moved to parent C%d, which has %d (prev %d) children",
			c, grandchildren, p, all_children, orig_children);
		//update transits
		int max_tr = clusters[p].transits, min_ls = clusters[p].lastseen;
		for (int k = 0; k < all_children; k++){
			max_tr = MAX(max_tr, clusters[clusters[p].children[k].first].transits);
			min_ls = MIN(min_ls, clusters[clusters[p].children[k].first].lastseen);
		}
		//update transits and lastseen
		clusters[p].transits = max_tr;
		clusters[p].lastseen = min_ls;

		//delete child
		markForDeletion(c);
		updateIDsFromOldClusters(c, p, clusters[p].pt.size() - 1);
	}
}


//this does NOT update trajectories
void ClusteringApp::incrementChildFreqs(int parentID, int childID){
	if (parentID == ROOT_NODE){ //%its a new point, where childID may be a root
		//itself (dont care) or the root is very short.
		if (clusters[childID].parent > ROOT_NODE){ //then we have a very short root
			int p = clusters[childID].parent; //set p to actual parent
			/*if (clusters[p].parent > ROOT_NODE)
			throw("grandchild, need to recurse"); ///this has never happened
			//in opencv turnsout it does! FIXME: replace with recursion up the tree*/
			//find and increment freq
			for (size_t j = 0; j < clusters[p].children.size(); j++){
				if (clusters[p].children[j].first == childID){
					clusters[p].children[j].second++;
					break;
				}
			}
		}
	}
	else {
		if (parentID != childID) {//walk over tree with while loop
			if (clusters[childID].parent != parentID){
				int p, i = childID;
				while (clusters[i].parent != parentID) {
					p = clusters[i].parent; //get parent of i
					for (size_t j = 0; j < clusters[p].children.size(); j++){
						if (clusters[p].children[j].first == childID){
							clusters[p].children[j].second++;
							break;
						}
					}
					i = p; //walk up the tree
				}
			}
			else { //update frequencies of child within parent
				for (size_t j = 0; j < clusters[parentID].children.size(); j++){
					if (clusters[parentID].children[j].first == childID){
						clusters[parentID].children[j].second++;
						break;
					}
				}
			}
		} //else parentID == childID, do nothing
	}
}

//walk up tree and return true if object types are different
bool ClusteringApp::isParentTypeDifferent(int id, int newtype){
	if (clusters[id].parent <= ROOT_NODE){
		return (are_different_object_types(clusters[id].type, newtype));
	}
	else {
		return (are_different_object_types(clusters[id].type, newtype)) ||
			isParentTypeDifferent(clusters[id].parent, newtype);
	}
}

//walk up tree and return true if object types are different
bool ClusteringApp::updateSingleClusterPoint(Point2f pt, int cc, int cs, int id, int type){
	assert(!are_different_object_types(type, clusters[cc].type));//check D&T arent incompatible
	if (clusters[cc].type == OBJECT_UNDETERMINED &&
		(type == OBJECT_CAR || type == OBJECT_PED))
	{
		if (isParentTypeDifferent(cc, type)){
			return true; //bail out
		}
		debug_printf("resolving C%d froum undet to type %d using id %d\n", cc, type, id);
		clusters[cc].type = type; //only set if object type is unknown
	}

	//save old point
	Point2f oldpt = clusters[cc].pt[cs];
	clusters[cc].pt[cs].x = (1 - learning_alpha)*clusters[cc].pt[cs].x + learning_alpha * pt.x;
	clusters[cc].pt[cs].y = (1 - learning_alpha)*clusters[cc].pt[cs].y + learning_alpha * pt.y;
	//TODO check this:sqrt & squared?
	float de = euclideanDistance(clusters[cc].pt[cs], pt);
	/*try and stop this running away from itself for large de.
	caps at 1.447 of de. */
	float new_sigma = (1.f - learning_alpha)*clusters[cc].sigma[cs] +
		learning_alpha * MIN(de, 1 + expf(-.5f*de));

	debug_printf("updated C%d pt %d from [x%d y%d s%f] to [%d %d %f] based on %d %d (%d type %d)\n",
		cc, cs, cvRound(oldpt.x), cvRound(oldpt.y), clusters[cc].sigma[cs],
		cvRound(clusters[cc].pt[cs].x), cvRound(clusters[cc].pt[cs].y), new_sigma,
		cvRound(pt.x), cvRound(pt.y), id, type);

	clusters[cc].sigma[cs] = new_sigma;
	clusters[cc].lastseen = 0;
	return false;
}


//return length of an entire tree, when given a leaf node (ie. what is the
//distance from the end of the given cluster to the root node?). recursive.
int ClusteringApp::getTreeLength(int i){
	if (i <= ROOT_NODE)
		return 0;
	else {
		if (clusters[i].parent == ROOT_NODE)
			return clusters[i].pt.size();
		else
			return getTreeLength(clusters[i].parent) + clusters[i].pt.size();
	}
}


//split cluster [parent] at point[stage]. thus the last point in [parent]
//will be [stage] and the first point in [child] is [stage+1]
void ClusteringApp::split(int stage, int parentID, list<ObjectTracker>& kfs){
	assert((int)clusters[parentID].pt.size() >= stage);
	//start off with default function
	int childID = makeCluster(Point2f(-1, -1), SIGMA_DEFAULT, ROOT_NODE, clusters[parentID].type);
	clusters[childID].transits = clusters[parentID].transits;
	//%get remaining length
	//copy points at end of parent to child

	clusters[childID].pt.clear(); clusters[childID].sigma.clear();
	vector<Point2f>::iterator p_iter = clusters[parentID].pt.begin();
	p_iter += (stage + 1);
	vector<float>::iterator f_iter = clusters[parentID].sigma.begin();
	f_iter += (stage + 1);
	//explicit resize of output elems
	int numel_child = distance(p_iter, clusters[parentID].pt.end());
	clusters[childID].pt.clear();
	clusters[childID].pt.resize(numel_child);

	clusters[childID].sigma.clear();
	clusters[childID].sigma.resize(numel_child);

	copy(p_iter, clusters[parentID].pt.end(), clusters[childID].pt.begin());
	copy(f_iter, clusters[parentID].sigma.end(), clusters[childID].sigma.begin());

	/*%deal with children. because child nodes are only attached at the
	%end of a parent, all existing child nodes of the parent should be
	%reattached to the new child.*/
	int n = getNumChildren(parentID); //%NOT including the new childID
	int sum = 0;
	if (n) {
		/*%childID is now parent of existing children, and inherits its
		%frequencies*/
		clusters[childID].children = clusters[parentID].children;
		for (size_t j = 0; j < clusters[parentID].children.size(); j++){
			//existing children now have childID as parent
			clusters[clusters[parentID].children[j].first].parent = childID;
			sum += clusters[parentID].children[j].second;
		}
	}
	else
		sum = 1;

	//clobber any existing children, no matter n
	clusters[parentID].children.clear(); //parent now only has 1 child
	clusters[parentID].children.push_back(pair<int, int>(childID, sum));

	//dont need to set freq for case n==0?
	clusters[parentID].sigma.resize(stage + 1);
	clusters[parentID].pt.resize(stage + 1);

	//now update parent of child
	clusters[childID].parent = parentID;

	//find all trajectories associated with this cluster. if they are over the length of 
	//parentID then reassign to childID.
	//easiest way to do this is to iterate over *running* trackers
	int pl = getTreeLength(parentID);
	for (list<ObjectTracker>::iterator i_kf = kfs.begin(); i_kf != kfs.end(); i_kf++){
		if (i_kf->age > pl && id2cl[i_kf->id].first == parentID){
			int np = MAX(0, id2cl[i_kf->id].second - (stage + 2)); //-2 for size and -1 for autoincrement
			assert((np < (int)clusters[childID].pt.size()) && np >= 0);
			assert(parentID == id2cl[i_kf->id].first);//may not hold
			debug_printf("split:id2cl at %d moved from C%d(%d) to C%d(%d) [in split]\n",
				i_kf->id, parentID, id2cl[i_kf->id].second, childID, np);
			id2cl[i_kf->id] = pair<int, int>(childID, np);
		}
	}

	stringstream s;
	for (size_t j = 0; j < clusters[childID].children.size(); j++){
		//existing children now have childID as parent
		s << (clusters[childID].children[j].first) << " ";
	}
	debug_printf("split cluster C%d after point %d to form C%d. Existing children C%s changed\n",
		parentID, stage, childID, (s.str()).c_str());
}


/*recursive forward cluster search. looks for best match within a window range.
%[pt]: point to search for. [pos_lower pos_upper]: tl range to search over.
%sc: search cluster(cluster to search). [bc bp]: existing best
%cluster and point indices. md: min distance to best point*/
void ClusteringApp::internalForwardSearch(
	Point2f pt, int pos_lower, int pos_upper, int sc, int* bc, int* bp, float* md, int type)
{
	if (are_different_object_types(type, clusters[sc].type))
		return;
	int ts_this = getTreeLength(sc); //tree length of this (for searches in children)
	size_t n = getNumChildren(sc); //does this have children or not?

	for (int i = pos_lower; i < pos_upper; i++){
		int sp = i;
		//handle special case of just after a cluster split, where age is too high to match
		if (i == pos_lower && n == 0 && sp == clusters[sc].pt.size())
		{
			sp--;
			num_search_errors_nonfatal++;
			//this is ok because only the normal case uses sp and if selected point
			//is outside the cluster window anyway it doesnt matter
		}
		//normal case
		if (sp < (int)clusters[sc].pt.size()) { //we are in this cluster
			float d = sqrtf(((pt.x - clusters[sc].pt[sp].x)*(pt.x - clusters[sc].pt[sp].x) +
				(pt.y - clusters[sc].pt[sp].y)*(pt.y - clusters[sc].pt[sp].y)) /
				clusters[sc].sigma[sp]);
			if (d < *md){
				*md = d; *bp = sp; *bc = sc; //store details of new closest point
			}
		}
		else { //start looking at children
			for (size_t j = 0; j < n; j++){ //search within each child
				internalForwardSearch(pt, MAX(0, i - (ts_this)), pos_upper, //search point. look at (i-ts_parent) for match, pos_upper, 
					clusters[sc].children[j].first, bc, bp, md, type);
			}
			break; //jump out of for loop, we've already looked at all depths
		}
	}
}


/*Search all children of cluster [sc] for matches to [pt] to an appropriate
%depth based on trajectory length [tl].
%should return [matched(bool) best_cluster best_stage distance]
%everything else is unscoped*/

bool ClusteringApp::forwardSearch(Point2f pt, int sc, int tl, int id, int type,
	int* bestCluster, int* bestStage, float* minDist, enum SEARCHTYPE searchtype, int sp)
{
	//sp is search point
	/*%work out number of steps to search over (now no harm in setting
	%this >tl because we wont look past the end of a tree)*/
	int pos_lower;// 
	int pos_upper;// 

	int startpoint = 0, parentLength = 0;
	switch (searchtype){
	case normal_search:
		//start with search point and go from there.
		//usual back/forward within-cluster search
		parentLength = getTreeLength(clusters[sc].parent);
		startpoint = MAX(parentLength + sp, tl) - parentLength;
		pos_lower = MAX(0, startpoint - cvCeil((tl - 1)*win_width));
		assert(pos_lower >= 0);
		pos_upper = sp + cvCeil(tl*+win_width); // refers to current cluster anyway
		//assert(pos_upper > pos_lower);
		break;
	case root_search: //when matching unmatched, points to a root node
		pos_upper = root_cluster_search_depth;
		pos_lower = cvFloor((tl - 1)*(1 - win_width));
		debug_printf("searching from %d to root search depth %d\n",
			pos_lower, root_cluster_search_depth);
		break;
	case forward_only_search:
		pos_lower = sp;
		pos_upper = sp + cvCeil(tl*+win_width);
		break;
	default:
		throw("search type: wtf");
	}

	/*%TODO change this for root nodes and new points, make it much
	%easier to get a match
	%recursively search this cluster and all its children over given range*/
	int bc = -1, bp = -1;
	float md = numeric_limits<float>::infinity();

	internalForwardSearch(pt, pos_lower, pos_upper, sc, &bc, &bp, &md, type);

	//now we have the closest point within this range, check if it matches
	bool m = false;

	if (numeric_limits<float>::infinity() != md)
		m = (md < 2 * sqrtf(clusters[bc].sigma[bp]));

	if (m && (bc == sc))
		debug_printf("matched point %d %d% (%d type%d) to same cluster C%d pos %d\n",
		cvRound(pt.x), cvRound(pt.y), id, type, bc, bp);
	else {
		if (m)
			debug_printf("matched point %d %d (%d type%d) to child C%d pos %d of C%d\n",
			cvRound(pt.x), cvRound(pt.y), id, type, bc, bp, sc);
		else {
			float th = numeric_limits<float>::infinity();
			if (md != numeric_limits<float>::infinity())
				th = 2 * sqrtf(clusters[bc].sigma[bp]);
			debug_printf("didn`t match %d %d (%d type%d) to C%d or any children."
				" closest was C%d pos %d d%f (thresh %f)",
				cvRound(pt.x), cvRound(pt.y), id, type, sc, bc, bp, md, th);
		}
	}
	*bestCluster = bc;
	*bestStage = bp;
	*minDist = md;
	return m;
}


void ClusteringApp::update(std::list<ObjectTracker>& kfs){
	debug_printf("Updating clusters\n");
	nIDs = TrackerIDList::getCount();

	if (nIDs > id2cl.size()){
		//extend id2cl to fit nIDs and set all new ids to -1
		id2cl.resize(nIDs, pair<int, int>(ROOT_NODE, 0));
	}

	if (clusters.size() == 0){
		makeCluster(Point2f(-1, -1), SIGMA_DEFAULT_ROOT_NODE, ROOT_NODE, OBJECT_UNDETERMINED);
		if (id2cl.size() == 0) //link cluster to id
			id2cl.push_back(pair<int, int>(0, 0));
		else
			id2cl[0] = pair<int, int>(0, 0);
	}

	//early stuff: if lost, update transit count of associated cluster
	//update anomalousness of object
	int nTracks = kfs.size();
	int nClusters = clusters.size();

	//corner cases
	if (nClusters == 1 && clusters[0].pt[0] == Point2f(-1, -1) &&
		kfs.size() > 0 && kfs.begin()->age >= min_tracker_age)
	{
		clusters.clear();
		makeCluster(Point2f(kfs.begin()->statePost.at<float>(0),
			kfs.begin()->statePost.at<float>(1)),
			SIGMA_DEFAULT_ROOT_NODE, ROOT_NODE, kfs.begin()->type);
		id2cl[0] = pair<int, int>(ROOT_NODE, 0);
		debug_printf("initialising C0 to [%d %d] from id %d\n",
			cvRound(clusters[0].pt[0].x), cvRound(clusters[0].pt[0].y), kfs.begin()->id);
	}


	size_t j = 0; //keep scope of j sensible
	for (list<ObjectTracker>::iterator i_kf = kfs.begin(); i_kf != kfs.end(); i_kf++, j++){
		if (i_kf->lost){
			updateTransitCount(id2cl[i_kf->id].first);
			id2cl[i_kf->id] = pair<int, int>(ROOT_NODE, 0);
		}
		else if (i_kf->age >= min_tracker_age){
			float tracker_score, cluster_score, dummy_theta = 0, dummy_rho = 0;

			//start main loop to assign track to a cluster
			int id = i_kf->id;
			int type = i_kf->type;

			int points_to_cluster, tl;
			Point2f track_pt;
			if (i_kf->unseen){ //may have to assign several points
				points_to_cluster = i_kf->record.size();
			}
			else {
				points_to_cluster = 1;
			}

			for (int k = 0; k < points_to_cluster; k++){
				enum SEARCHTYPE this_search = normal_search;
				if (i_kf->unseen){
					tl = k + 1; //for unmatched points, set track length 
					//as we go along. start off at 1 (always see at least 1 point)
					track_pt = i_kf->record[k];
					debug_printf("clustering %d of %d points in id %d( type %d)\n",
						k, points_to_cluster, id, type);
					this_search = forward_only_search;
				}
				else { //normal behaviour, should only go run this once
					tl = i_kf->age;
					track_pt = Point2f(i_kf->statePost.at<float>(0),
						i_kf->statePost.at<float>(1));
				}
				/*if two objects of undetermined type were
				previously matched to a tracker and one of those objects
				then resolves to a (car/ped) AND the other then resolves to
				a <ped/car> then the track-to-cluster match will then be
				invalid. work around this by checing and unassigning matches
				befor the main switch so we can treat the new thing as unassigned.
				*/
				if (id2cl[id].first > ROOT_NODE && are_different_object_types(
					type, clusters[id2cl[id].first].type))
				{
					debug_printf("unassigning C%d from object (%d, t%d) as"
						" different types\n", id2cl[id].first, id, type);
					forgetTracker(id, i_kf, id2cl[id].first);
					continue;
				}

				//on with the main loop
				// has this object been seen before?
				if (id2cl[id].first == ROOT_NODE) {
					//not seen before so search root nodes
					bool matched = false; float md = numeric_limits<float>::infinity();
					int bc = -1, bp = -1;
					int bc_temp = -1, bp_temp = -1; //set to invalid values
					float md_temp = md;
					bool flag = false;
					for (int ci = 0; ci < nClusters; ci++){ //if it matches objType
						if ((clusters[ci].parent == ROOT_NODE) &&
							!are_different_object_types(clusters[ci].type, type))
						{
							//find closest point in first few points, in all root clusters
							flag = forwardSearch(track_pt, ci, tl, id, type,
								&bc_temp, &bp_temp, &md_temp, root_search); //ALWAYS do a root search
							if (md_temp < md) { //new set of best points found
								bc = bc_temp; bp = bp_temp;
								matched = flag; md = md_temp;
							}
						}
					}
					if (matched) { //update
						if (bp != 0) { //assume we always go to first point?
							debug_printf("new traj from id %d type %d matched to pos>0: C%d pos %d.\n",
								id, type, bc, bp);

						}
						updateSingleClusterPoint(track_pt, bc, bp, id, type);
						incrementChildFreqs(ROOT_NODE, bc);
					}
					else { //create cluster (as root node)
						bc = makeCluster(track_pt, SIGMA_DEFAULT_ROOT_NODE, ROOT_NODE, type);
						bp = 0;
					}
					//now update all our record keeping methods
					assert(id2cl[id].first == ROOT_NODE);
					id2cl[id] = pair<int, int>(bc, bp);
					debug_printf("new track id %d type %d linked to C%d [%d %d]\n",
						id, type, bc, cvRound(track_pt.x), cvRound(track_pt.y));
				}
				else { //has clusters assoc'ed already bc its an existing track.
					//%watch: this branch uses cc instead of bc
					int cc = id2cl[id].first; //get current cluster
					int sp = ++id2cl[id].second; //get current search point and increment
					//if changing the above then change the +2 assertion in  sploit()
					assert(cc != ROOT_NODE);
					int cc_old = cc;
					//search that cluster and its children, return best point & cluster
					bool matched = false; float md = numeric_limits<float>::infinity();
					int bc = -1, bp = 0; //initialise to zero, maybe bad idea

					if (isParentTypeDifferent(cc, type)){
						//take some  action to prevent matching to existing clusters.
						//in this case it's much simpler to reset the tracker.
						forgetTracker(id, i_kf, cc);  //check this works
						continue;
					}
					matched = forwardSearch(track_pt, cc, tl, id, type, &bc, &bp, &md, this_search, sp);
					if (!matched && md == numeric_limits<float>::infinity()){
						if (!are_different_object_types(clusters[cc].type, type))
							bp = MAX(0, MIN((sp - 1), ((int)clusters[cc].pt.size()) - 1)); //take a guess at previous match and hopefully split
						else
							printf("what\n");

						if (bc != cc && bc != -1){
							debug_printf("best match to id %d type %d was C%d, is now C%d pos%d.\n",
								id, type, cc, bc, bp);
							cc = bc;
						}
						int ts_leaf = getTreeLength(cc) - 1; //<note fragile change from matlab here
						int ts_branch = getTreeLength(clusters[cc].parent);

						//if closest point is NOT at end of tree
						if ((ts_branch + bp) < ts_leaf){
							if (matched) {/* %matches cluster but not beyond end of cluster
										  %ok if tl>tree_size as thats the purpose of the window*/
								if (updateSingleClusterPoint(track_pt, cc, bp, id, type)){
									cc = -1; bp = 0;
									debug_printf("bailout/1: type mismatch for %d (%d) somewhere in tree. resetting id2cl from C%d(%d) to C%d(%d)\n",
										id, type, id2cl[id].first, id2cl[id].second, cc, bp);
								}
								else {
									debug_printf("update/1: change id2cl at %d from C%d(%d) to C%d(%d)\n",
										id, id2cl[id].first, id2cl[id].second, cc, bp);
									incrementChildFreqs(cc_old, cc);
								}
								id2cl[id] = pair<int, int>(cc, bp);
							}
							else { //split current cluster(at mdpi) to create 2 children
								debug_printf("closest point of id %d type %d is C%d "
									"point %d (not at end)\n", id, type, cc, bp);
								split(bp, cc, kfs);
								int childID = makeCluster(track_pt, SIGMA_DEFAULT, cc, type);
								debug_printf("update/2: post split change id2cl at %d to C%d(%d)\n",
									id, childID, 0);
								id2cl[id] = pair<int, int>(childID, 0); //update link
								//incrementChildFreqs(cc_old,childID); this is done in make_cluster 
							}
						}
						else {
							if (((ts_branch + MAX(sp - 1, bp)/*bp*/) == ts_leaf) && (tl <= ts_leaf) && matched){
								/*elseif closest point is at end of tree AND tl<=tree size AND
								matched: update closest point*/
								if (updateSingleClusterPoint(track_pt, cc, bp, id, type)){
									cc = -1; bp = 0;
									debug_printf("bailout/2: type mismatch for %d (%d) somewhere in tree. resetting id2cl from C%d(%d) to C%d(%d)\n",
										id, type, id2cl[id].first, id2cl[id].second, cc, bp);
								}
								else {
									debug_printf("update/3: change id2cl at %d from C%d(%d) to C%d(%d)\n",
										id, id2cl[id].first, id2cl[id].second, cc, bp);

									incrementChildFreqs(cc_old, cc);
								}
								id2cl[id] = pair<int, int>(cc, bp);
							}
							else { //(closest point is at end of tree AND tl>tree_size OR ~matched
								int childID = makeCluster(track_pt, SIGMA_DEFAULT, cc, type);
								debug_printf("update/4: change id2cl at %d from C%d(%d) to C%d(%d)\n",
									id, id2cl[id].first, id2cl[id].second, childID, 0);
								id2cl[id] = pair<int, int>(childID, 0);
								/*dont do incrementChildFreqs(cc_old,childID);
								%this gets done in make_cluster*/
							}
						}
					}
				}
				if (i_kf->unseen && i_kf->age >= 0){ //all unseen points now processed
					i_kf->unseen = false;
					i_kf->record.clear();
					debug_printf("end clustering of id %d type %d\n", id, type);
				}
			}
		}
	}

	prune(); //prune dead clusters

	//concatenate all clusters with only 1 child, and try to delete anything which is zero length
	for (size_t j = 0; j < clusters.size(); j++){
		concatenate(j);
		deleteZeroLengthClusters(j);
	}

	//tidy cluster array and use the chance to do some assertions
	for (size_t j = 0; j<clusters.size(); j++){
		if (clusters[j].parent == DISABLED_NODE) {
			size_t k = clusters.size() - 1;
			while (clusters[k].parent == DISABLED_NODE && k>j) //psossible off-by-1 error
				k--;
			if (k > j && clusters[k].parent != DISABLED_NODE){
				debug_printf("tidy: replacing C%d with C%d. ", j, k);
				clusters[j] = clusters[k];//k now assigned to j, can delete k
				markForDeletion(k);
				for (size_t kk = 0; kk < getNumChildren(j); kk++){
					//change any/all children of k to j
					clusters[clusters[j].children[kk].first].parent = j;
				}
				//change "children" field of parent of k, if it has one
				if (clusters[j].parent > ROOT_NODE) {
					for (size_t kk = 0; kk < getNumChildren(clusters[j].parent); kk++){
						if (clusters[clusters[j].parent].children[kk].first == k){
							clusters[clusters[j].parent].children[kk].first = j;
							break;
						}
					}
				}
				updateIDsFromOldClusters(k, j, 0/*-1*/);//-1 so no overall change in length offset
			}
		}
		//assertions
		assert(clusters[j].pt.size() == clusters[j].sigma.size());
		for (size_t k = 0; k < clusters[j].children.size(); k++){
			assert(clusters[j].children[k].first != j);
		}
		if (clusters[j].parent > ROOT_NODE && clusters[clusters[j].parent].pt.size()){
			assert(euclideanDistance(clusters[j].pt.front(), clusters[clusters[j].parent].pt.back()) < 200);
		}
	}

	{//get rid of disabled clusters at end of list. watch for off by one errors
		size_t j = clusters.size() - 1;
		if (j > 0){
			while (clusters[j].parent == DISABLED_NODE && j > 0)
				j--;
			if (j < clusters.size() - 1){
				debug_printf("shrinking cluster array from %d to %d entries\n", clusters.size(), j + 1);
				clusters.resize(j + 1);
			}
		}
	}
	//assertions
	nClusters = clusters.size();
	for (list<ObjectTracker>::iterator i_kf = kfs.begin(); i_kf != kfs.end(); i_kf++){
		assert(id2cl[i_kf->id].first <= nClusters);
	}
	//return. evaluate cluster anomaly and plot later
}


float ClusteringApp::rateAnomalyAndPlotClusters(Mat& base_img, DisplayColourMap* colours,
	std::list<ObjectTracker>& kfs, BayesianAnomalyDetector* bayes, bool draw, bool drawClusters){
	//plot all the clusters
	if (drawClusters && draw){
		char s[100];
		for (size_t j = 0; j < clusters.size(); j++){
			sprintf(s, "%2.2f/%d #%d", getClusterAnomalousness(j), clusters[j].transits, j);
			if (clusters[j].pt.size()){
				putText(base_img, s, clusters[j].pt[0] + Point2f(20, 20),
					FONT_HERSHEY_COMPLEX_SMALL, .8, colours->getColoursFromID(clusters[j].type));
			}
			for (size_t k = 1; k < clusters[j].pt.size(); k++){
				line(base_img, clusters[j].pt[k - 1], clusters[j].pt[k],
					colours->getColoursFromID(clusters[j].type));
			}
			if (clusters[j].parent > ROOT_NODE && clusters[clusters[j].parent].pt.size()){
				line(base_img, clusters[j].pt[0], clusters[clusters[j].parent].pt.back(),
					colours->getColoursFromID(clusters[j].type));
			}
			else if (clusters[j].parent == ROOT_NODE && clusters[j].pt.size()){
				circle(base_img, clusters[j].pt[0], 5, colours->getColoursFromID(clusters[j].type), 5);
			}
		}
	}
	//get and plot anomaly level for all points. include cluster anomaly level only for active clusters
	float overall_anom = 0, tot_anom = 0, c_anom = 0;
	char msg[30];
	int count = 0;
	float obj_anom = 0;
	for (list<ObjectTracker>::iterator i_kf = kfs.begin(); i_kf != kfs.end(); i_kf++){
		if (i_kf->age >= min_tracker_age && !i_kf->lost){ //only do this for valid long-lived trackers 
			c_anom = getClusterAnomalousness(id2cl[i_kf->id].first);
			float anom_score_x = 0, anom_score_y = 0;
			bayes->calcAnomalyScores(i_kf->bb, i_kf->statePost.at<float>(2), i_kf->statePost.at<float>(3), i_kf->type,
				&anom_score_x, &anom_score_y);
			i_kf->anom += (anom_score_x * 10 + anom_score_y * 10);
			obj_anom = i_kf->anom / (i_kf->age - (min_tracker_age - 1));
			overall_anom = MAX(overall_anom, obj_anom + c_anom);
			tot_anom += (obj_anom + c_anom);
			count++;
			if ((obj_anom + c_anom) >= anom_overall_thresh){
				i_kf->anom_counter++;
			}
			else {
				i_kf->anom_counter = 0;
			}

			if (draw){
				int thisID = id2cl[i_kf->id].first;
				float dummy_tracker_score, dummy_cluster_score, theta_mean = 0, rho_mean = 0, xu_m = 0, yu_m = 0, yu_v, xu_v;
				bayes->hm->getRegionMean(i_kf->bb, i_kf->type, &xu_m, &yu_m, &xu_v, &yu_v);
				sprintf(msg, "%2.1f/%2.1f/%d/%2.1f,%2.1f", obj_anom, c_anom, getTransits(thisID), anom_score_x, anom_score_y); //object anomaly level (set in last frame)
				debug_printf("%d: %f %s\n", count, obj_anom, msg);
				putText(base_img, msg, Point2f(i_kf->statePost.at<float>(0) - 10, i_kf->statePost.at<float>(1) + 20),
					FONT_HERSHEY_SIMPLEX, 0.6, colours->getColoursFromID(i_kf->type + 3), 2);
				if ((obj_anom + c_anom) >= anom_overall_thresh){
					rectangle(base_img, Point2f(i_kf->statePost.at<float>(0) - 20, i_kf->statePost.at<float>(1) - 20),
						Point2f(i_kf->statePost.at<float>(0) + 30, i_kf->statePost.at<float>(1) + 30), CV_RGB(255, 255, 255), 5);
				}
				//taken from tracker app
				circle(base_img, Point2f(i_kf->statePost.at<float>(0) - 5, i_kf->statePost.at<float>(1) - 5),
					10, colours->getColoursFromID(i_kf->type), 2);
				sprintf(msg, "%d|%2.3f,%2.3f|%3.fr%2.1f", i_kf->id, i_kf->statePost.at<float>(2),
					i_kf->statePost.at<float>(3), i_kf->motionBearingDegrees(), i_kf->motionMagnitude()); //object tracker centre
				putText(base_img, msg, Point2f(i_kf->statePost.at<float>(0) - 10, i_kf->statePost.at<float>(1) - 10),
					FONT_HERSHEY_SIMPLEX, 0.6, colours->getColoursFromID(i_kf->type + 3), 2);
			}
		}
	}
	if (draw){
		sprintf(msg, "anom:%3.1f /%3.1f", overall_anom, tot_anom / MAX(count, 1));
		putText(base_img, msg, Point(10, 100), FONT_HERSHEY_SIMPLEX, 0.8, CV_RGB(255, 0, 0), 2);
	}
	return overall_anom;
}

//rederaw cluster outputs on camera plane
void ClusteringApp::transformClusterstoCam(vector<vector<Point> >* camClusters, vector<int>*clusterTypes, Mat H_base2cam){
	vector<Point2f> camPoints;
	vector<Point> labelledCluster;

	for (size_t i = 0; i < clusters.size(); i++){
		camPoints.clear();
		int isChild = 0;
		if (clusters[i].pt.size()){ //note that these might not match up as some empty clusters
			perspectiveTransform((clusters[i].pt), camPoints, H_base2cam); //transform the points
			//place these into a Point vector
			if (clusters[i].parent > ROOT_NODE && clusters[clusters[i].parent].pt.size()){//parent is nonzero
				isChild = 1;
			}
			labelledCluster.resize(camPoints.size() + isChild); //leave room for parent/child link

			for (size_t j = 0; j < camPoints.size(); j++){
				labelledCluster[j + isChild] = Point((int)camPoints[j].x, (int)camPoints[j].y);
			}
			//put the parent->child link in there too
			if (isChild){
				vector<Point2f> parentpt(1, clusters[clusters[i].parent].pt.back());
				camPoints.clear();
				perspectiveTransform(parentpt, camPoints, H_base2cam); //transform last point in parent
				labelledCluster[0] = Point((int)camPoints[0].x, (int)camPoints[0].y);
			}

			//push that into a vector
			camClusters->push_back(labelledCluster); //put the vector into new list
			clusterTypes->push_back(clusters[i].type);
		}
	}
}
