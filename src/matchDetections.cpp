#include "stdafx.h"
#include "acceleratedAlgorithm.h"
#include "tracking.h"
#include "display.h"
#include "unsupervisedHeatmap.h"

#include "trackingApp.h"
#include "fpgaImageSize.h"


using namespace cv;
using namespace std;


//utility function. uses Platt scaling to generate detection probability based on svm score
//and pre-learned coefficients
inline double convertConfidenceToProb(double conf, bool isCar){
	double A, B;
	if (isCar){
		A = -4.394326000996488;
		B = -0.963527793325675;
	}
	else {
		A = -5.213255101313579;
		B = 6.529957414487241;
	}
	return  1.0 / exp(A*conf + B);
}

bool compareTrackableDetections(TrackableDetection i, TrackableDetection j){
	return (i.source > j.source); //depends what algorithm_motiononly is set as (+ or - everything else)
}


TrackerApp::TrackerApp(Mat H_cam2base_, std::map<int, int> impl2obj_, std::map<int, int> impl2algo_,
	PatchProcessor hog_patch_, Point validTL, Point validBR)
{
	//set parameters
	thresh_det2track_ = 55.f; //was 50 -> 60 -> 50
	loose_thresh_det2track = 2 * thresh_det2track_;
	eccentricity = sqrt(pow(thresh_det2track_, 2) - pow(0.4f*thresh_det2track_, 2)) / thresh_det2track_;

	H_cam2base = H_cam2base_;
	impl2obj = impl2obj_;
	impl2algo = impl2algo_;
	hog_patch = hog_patch_;

	max_obj_area = 0;
	motionBorder = 10.f; //50px round an object - is this really necessary?
	motionScale = 1.5f;
#ifdef BANK_ST 
	motionScale = 2.f;
#endif
	track_lost_hard_thresh = 15; //kill after this unless stationary
	track_lost_soft_thresh = 5; //kill after this unless it's being used
	track_lost_speed_thresh = 2; //ersatz speed. if above this then kill

	lost_xmin = validTL.x; //allowed range on the base plane
	lost_xmax = validBR.x;
	lost_ymin = validTL.y;
	lost_ymax = validBR.y;

	//set up inverse homography
	H_base2cam = H_cam2base.inv();
	/*H_base2cam.at<float>(2,0) = 0;
	H_base2cam.at<float>(2,1) = 0;
	H_base2cam.at<float>(2,2) = 1;*/

	merge_tracks_age_thresh = 50;
	merge_tracks_dist_thresh = 40;
	merge_tracks_theta_thresh = (float)CV_PI / 6.f; //30deg

	intersection_thresh = 0.45f; // final detection overlap threshold. ilids: 0.45f
	motion_detection_overlap_thresh = 0.9f; //(ilids 0.9f)compare motion-generated detection cands against existing full-frame dets
	motion_motion_overlap_thresh = 0.9f; //(ilids=0.9f) motion vs motion bbox comparison. for bankst this is v low
}


//late/separated deletion of lost trackers
int TrackerApp::deleteLostTrackers(){
	int count = 0;
	for (list<ObjectTracker>::iterator i_kf = kfs.begin(); i_kf != kfs.end();){
		if (i_kf->lost){
			i_kf = kfs.erase(i_kf);
			count++;
		}
		else
			i_kf++;
	}
	if (count)
		cout << "deleted " << count << "lost trackers\n";
	return count;
}

//second chance tracker matching. erase in place rather than setting lost flag,
//so cluster transit counts dont become incorrect
void inline TrackerApp::mergeNearbyTracks(){
	list<ObjectTracker>::iterator i_kf, j_kf;
	bool kill_i = false, kill_j = false;
	for (i_kf = kfs.begin(); i_kf != kfs.end();){
		for (j_kf = kfs.begin(); j_kf != kfs.end();){
			kill_i = false; kill_j = false;
			if (i_kf != j_kf &&
				!i_kf->lost &&
				!j_kf->lost &&
				!are_different_object_types(i_kf->type, j_kf->type) &&
				(min(i_kf->age, j_kf->age) < merge_tracks_age_thresh))
			{
				//just checked sameness, lost, type, and age.
				//now check distance, angle (and rho?)
				float dx = i_kf->statePost.at<float>(0) - j_kf->statePost.at<float>(0);
				float dy = i_kf->statePost.at<float>(1) - j_kf->statePost.at<float>(1);
				if (sqrt(dx * dx + dy * dy) < merge_tracks_dist_thresh){
					float theta_i = atan2f(i_kf->statePost.at<float>(3), i_kf->statePost.at<float>(2)) + CV_PI_F / 2.f;
					theta_i = theta_i < 0 ? theta_i + 2 * CV_PI_F : theta_i;
					float theta_j = atan2f(j_kf->statePost.at<float>(3), j_kf->statePost.at<float>(2)) + CV_PI_F / 2.f;
					theta_j = theta_j < 0 ? theta_j + 2 * CV_PI_F : theta_j;
					float absdiff = abs(theta_i - theta_j);
					if (MIN(absdiff, 2.f*CV_PI_F - absdiff) < merge_tracks_theta_thresh){
						if (i_kf->age < j_kf->age) {//kill i
							kill_i = true;
						}
						else { //kill j
							kill_j = true;
						}
					}
				}
			}
			//assuming kill_i and kill_j can never be triggered at once
			if (kill_j){
				printf("replacing tracker %d with %d\n", j_kf->id, i_kf->id);
				i_kf->correct(*(Mat_<float>(2, 1) << j_kf->statePost.at<float>(0), j_kf->statePost.at<float>(1)));
				j_kf->lost = true;
				j_kf++;
			}
			else {
				if (kill_i){
					printf("replacing tracker %d with %d\n", i_kf->id, j_kf->id);
					j_kf->correct(*(Mat_<float>(2, 1) << i_kf->statePost.at<float>(0), i_kf->statePost.at<float>(1)));
					i_kf->lost = true;
					break; //out of inner for loop
				}
				else {
					j_kf++;
				}
			}
		}
		i_kf++;
	}
}


void TrackerApp::improveDetections(
	vector<Detection>* initialDetections, vector<Detection>*improvedDetections, HeisenFrame* hf)
{
	size_t numDets = initialDetections->size();
	Size imsz = hf->getFrame(CV_8UC3).size();

	improvedDetections->clear();//copy at end
	//work on the motion candidates a bit
	list<MotionBB> motionCands;
	list<Detection> detectedCands;
	bool all_motion_invalid = false;
	float minPedX = 64, minPedY = 128, minCarX = 104, minCarY = 56;
	float minBothY = min(minCarY, minPedY), minBothX = min(minPedX, minCarX);
	int max_obj_area = (int)(imsz.area()*0.25);

	//preprocess: throw away any objects over .25*framesize
	//unfortunately, one vector contains the motion and object detections.
	//separate the motion stuff and filter it
	for (vector<Detection>::iterator i_det = initialDetections->begin();
		i_det != initialDetections->end(); i_det++)
	{
		int pl, pr, pt, pb, pw, ph, candType; //initialising these inside a switch is awkward
		float pH, pW;
		int source_algo = impl2algo[i_det->source];
		Rect bb;
		//get algorithm by looking up originating detection's implementation
		//then taking that through a map. this might be a bit slow.
		switch (source_algo){
		case ALGORITHM_HOG:
		case ALGORITHM_HOG_CAR:
			if (i_det->bb.area() < max_obj_area){
				/*	improvedDetections->push_back(*i_det); dont add directly to output vec yet*/
				detectedCands.push_back(*i_det);
			}
			break;
		case ALGORITHM_MOG:
			//get info for an image patch
			bb = i_det->bb;
			pl = (int)max(ceil(bb.x - min(0.5f*bb.width, motionBorder * 2)), 0.f);
			pr = (int)min(floor(bb.x + min(1.5f*bb.width, bb.width + motionBorder * 2)), imsz.width - 1.0f);
			//assuming (0,0) is top left and bboxes start at top left and go down (ie they work the same way as matlab)
			pt = (int)max(ceil(bb.y - min(0.5f*bb.height, motionBorder)), 0.f);
			pb = (int)min(floor(bb.y + min(1.5f*bb.height, bb.height + motionBorder)), imsz.height - 1.0f);
			pw = pr - pl + 1;
			ph = pb - pt + 1;
			//what about the expanded image?
			pW = min(pw*motionScale, FPGA_COLS - pl - 1.f);
			pw = cvFloor((pW / motionScale));
			pH = min(ph*motionScale, FPGA_ROWS - pt - 1.f);
			ph = cvFloor((pH / motionScale));


			//test against possible object types
#ifndef BANK_ST
			if (ph*motionScale < minBothY || pw*motionScale <minBothX)
				continue; //too small, throw away detection/ useful for noisy bgsubs
#endif

			candType = 0; //this will be a bitmask
			if (ph*motionScale >= minPedY && pw*motionScale >= minPedX)
				candType += OBJECT_PED; //person
			//its a viable pedestrian candidate
			if (ph*motionScale >= minCarY && pw*motionScale >= minCarX)
				candType += OBJECT_CAR;
			//store the motion
			motionCands.push_back(MotionBB(pl, pr, pt, pb, pw, ph, candType, *i_det));

			if (i_det->bb.area() > max_obj_area)
				all_motion_invalid = true;
			break;
		default:
			std::cout << "unknown object type\n";
		}
	}

	if (all_motion_invalid){
		cout << " very noisy , throwing away all motion detections\n";
		motionCands.clear(); //v noisy so throw away all motion objs
	}

	//now filter out duplicate motion candidates
	int nremoved = 0;
	for (list<MotionBB>::iterator i_mot = motionCands.begin(); i_mot != motionCands.end(); i_mot++){
		list<MotionBB>::iterator j_mot;
		Rect rect_i = Rect(i_mot->l, i_mot->t, i_mot->w, i_mot->h); //cache one object
		Rect rect_j;
		for (j_mot = motionCands.begin(); j_mot != motionCands.end();){
			if (j_mot == i_mot){ j_mot++; continue; } //dont compare anything to itself
			rect_j = Rect(j_mot->l, j_mot->t, j_mot->w, j_mot->h);
			float inter = BBprops::intersection(rect_i, rect_j);
			if (inter > 0.9f && rect_i.area() > rect_j.area()){
				nremoved++;
				//if using erase, it iterates j_mot automatically so cant have a continuation-cindition in the for loop
				j_mot = motionCands.erase(j_mot);
			}
			else
				j_mot++;
		}
	}

	//now filter remaining motion candidates against existing (full-frame) detections, do overlap this time
	//slightly awkward while loop because we're erasing the thing in the outer loop
	{
		nremoved = 0;
		list<MotionBB>::iterator i_mot = motionCands.begin();
		list<Detection>::iterator imp_iter;
		while (i_mot != motionCands.end()) {
			bool erase = false;
			Rect rect_i = Rect(i_mot->l, i_mot->t, i_mot->w, i_mot->h); //cache one object
			for (imp_iter = detectedCands.begin(); imp_iter != detectedCands.end(); imp_iter++){
				float inter = BBprops::overlap(rect_i, imp_iter->bb);
				if (inter > intersection_thresh/*0.9f*/){
					nremoved++;
					erase = true;
					break; //out of for loop
				}
			}
			if (erase)
				i_mot = motionCands.erase(i_mot);
			else
				i_mot++;
		}
	}

	//if we had more time/were doing this properly, would extract this , separate generation of list of patches, 
	//and patch processing, and farm out patch processing to various elements.
	double car_prob_thresh = 0.0, ped_prob_thresh = 0.2;
	//Mat itmp = hf->getFrame(CV_8UC3);
	std::cout << "start patches ";
	for (list<MotionBB>::iterator i_mot = motionCands.begin(); i_mot != motionCands.end(); i_mot++){
		//either extract a patch or copy the region wholesale
		//lets copy wholesale at the mo
		//todo use detectROI if it works properly
		vector<Detection> cars, peds;

		if (i_mot->candidates > 0){
			Rect r = Rect(i_mot->l, i_mot->t, i_mot->w, i_mot->h);
			hf->setPatch(r, motionScale);
			//rectangle(itmp,r,Scalar(255,255,0),3);
			/*
			Mat roi = img(Range(i_mot->t,i_mot->b+1), Range(i_mot->l,i_mot->r+1));
			Mat impatch;
			resize(roi,impatch,Size(0,0),motionScale,motionScale);*/

			//maybe copy ROI to another image here
			if (i_mot->candidates & OBJECT_CAR){
				hog_patch.car->detect(/*impatch*/ hf, cars);
			}
			if (i_mot->candidates & OBJECT_PED){
				hog_patch.ped->detect(/*impatch*/ hf, peds);
			}
		}
		bool added = false;
		if (cars.size() || peds.size()){ //if anything detected
			for (size_t j = 0; j < cars.size(); j++){ //check score
				double prob = convertConfidenceToProb(cars[j].score, true);
				if (prob >= car_prob_thresh){
					cars[j].score = prob; //update score
					//clobber bb with bb rescaled to source img
					cars[j].bb = Rect(i_mot->l + cvRound(cars[j].bb.x / motionScale),
						i_mot->t + cvRound(cars[j].bb.y / motionScale),
						cvRound(cars[j].bb.width / motionScale),
						cvRound(cars[j].bb.height / motionScale));
					detectedCands.push_back(cars[j]); //add
					added = true;
				}
			}
			for (size_t j = 0; j < peds.size(); j++){ //repeat for peds
				double prob = convertConfidenceToProb(peds[j].score, false);
				if (prob >= ped_prob_thresh){
					peds[j].score = prob; //update score
					peds[j].bb = Rect(i_mot->l + cvRound(peds[j].bb.x / motionScale),
						i_mot->t + cvRound(peds[j].bb.y / motionScale),
						cvRound(peds[j].bb.width / motionScale),
						cvRound(peds[j].bb.height / motionScale));
					detectedCands.push_back(peds[j]); //add
					added = true;
				}
			}
		}
		if (!added){
			i_mot->det.source = ALGORITHM_MOTION_ONLY; //ugh bit of a hack
			detectedCands.push_back(i_mot->det); //put the original motion-based det back into the candidate list
		}
	}
	hf->usePatch = false; //turn off patch processing once done
	std::cout << "end patches \n";

	//check that none of the final tracking candidates overlap
	/*vector<Detection>::iterator imp_iter;*/
	nremoved = 0;
	for (list<Detection>::iterator idL = detectedCands.begin(), jdL; idL != detectedCands.end(); idL++){
		/*for (imp_iter = improvedDetections->begin(); imp_iter!=improvedDetections->end(); imp_iter++){*/
		for (jdL = detectedCands.begin(); jdL != detectedCands.end(); /*jdL++*/){
			if (idL == jdL) { jdL++; continue; }//dont compare anything to itself
			float inter = BBprops::overlap(idL->bb, jdL->bb);
			if (inter > intersection_thresh && idL->bb.area() > jdL->bb.area()){
				nremoved++;
				jdL = detectedCands.erase(jdL);
			}
			else
				jdL++;
		}
	}
	std::copy(detectedCands.begin(), detectedCands.end(), back_inserter(*improvedDetections));
}


void TrackerApp::runTrackersOnDetections(vector<Detection>detectedCands, Mat& base_img, int frames2skip, bool draw)
{
	//looser tracker matching if running with frameskip (only do this when matching to new points obviously)
	thresh_det2track = (1 + sqrtf(frames2skip))*thresh_det2track_;
	//thresh_det2track = thresh_det2track_;
	//loose_thresh_det2track = 2*thresh_det2track_; //use the scaled threshold
	//eccentricity = sqrt( pow(thresh_det2track,2) - pow(0.4f*thresh_det2track,2))/thresh_det2track;  //use the scaled threshold

	//convert source labels to object types
	static DisplayColourMap colours;
	for (vector<Detection>::iterator idL = detectedCands.begin(); idL != detectedCands.end(); idL++){
		idL->source = impl2obj[idL->source];
	}
	//transform to base plane	
	vector<TrackableDetection> baseCands(detectedCands.size());
	//surprised we dont need to use 3-element floats btut ok
	vector<Point2f> topleftpoints(detectedCands.size()), bottomrightpoints(detectedCands.size());
	size_t pt = 0;//putting this here but should go in loop
	for (vector<Detection>::iterator idL = detectedCands.begin(); idL != detectedCands.end(); pt++, idL++){
		//fill 2 vecs with camera plane points
#ifdef BANK_ST
		topleftpoints[pt] = Point2f((float)idL->bb.x, (float)idL->bb.y + idL->bb.height); //now bottom left
		bottomrightpoints[pt] = Point2f((float)idL->bb.x + idL->bb.width, (float)idL->bb.y); //now top right
#else
		topleftpoints[pt] = Point2f((float)idL->bb.x, (float)idL->bb.y);
		bottomrightpoints[pt]	= Point2f((float) idL->bb.x+idL->bb.width,	(float) idL->bb.y+idL->bb.height);
#endif
		//preserve these details
		baseCands[pt].score = idL->score;
		baseCands[pt].source = idL->source;
	}
	if (topleftpoints.size()){ //only run if points exist
		perspectiveTransform(topleftpoints, topleftpoints, H_cam2base);
		perspectiveTransform(bottomrightpoints, bottomrightpoints, H_cam2base);
	}

	//#define CLIP_RECTS_TO_FRAME
	//really not sure about this, leaving it turned off for now. not tested.
	//think the test for disacrding a detection should be that the centroid is no longer within the frame
	//but leaving these dodgy detections in place allows existing trackers to match to dets outside the 
	//window?
#ifdef CLIP_RECTS_TO_FRAME
	Size ground_plane_sz = base_img.size();
	vector<TrackableDetection> clippedBaseCands;// no initialisation/preallocation here
	TrackableDetection this_det; //
	for (pt=0;pt<topleftpoints.size();pt++){
		this_det.centre.x = topleftpoints[pt].x + max(1.f, (bottomrightpoints[pt].x-topleftpoints[pt].x)/2.f);
		this_det.centre.x = topleftpoints[pt].y + max(1.f, (bottomrightpoints[pt].y-topleftpoints[pt].y)/2.f);
		this_det.score = baseCands[pt].score;
		this_det.source= baseCands[pt].source;
		//fill in rest of fields.
		//clip this rect
		Rect this_rect = clipRect(Rect(cvRound(topleftpoints[pt].x),  cvRound(topleftpoints[pt].y),
			cvRound(bottomrightpoints[pt].x-topleftpoints[pt].x),cvRound(bottomrightpoints[pt].y-topleftpoints[pt].y)),
			ground_plane_sz);
		if (isClippedRectWithinWindow(this_rect,ground_plane_sz)){
			//finalise this_det
			this_det.bb = this_rect;
			//push the new TrackableDetection into a new vector			
			clippedBaseCands.push_back(this_det);
		}
	}
	if (clippedBaseCands.size() != baseCands.size()){
		baseCands.clear();
		baseCands = clippedBaseCands; //can we do this?
	}

#else 
	//finalise everything in baseCands
	for (pt = 0; pt < topleftpoints.size(); pt++){
#ifdef BANK_ST
		baseCands[pt].centre.x = topleftpoints[pt].x + max(1.f, (bottomrightpoints[pt].x - topleftpoints[pt].x) / 2.f);
		baseCands[pt].centre.y = bottomrightpoints[pt].y + max(1.f, (topleftpoints[pt].y - bottomrightpoints[pt].y) / 2.f);
#else
		baseCands[pt].centre.x = topleftpoints[pt].x + max(1.f, (bottomrightpoints[pt].x - topleftpoints[pt].x) / 2.f);
		baseCands[pt].centre.y = topleftpoints[pt].y + max(1.f, (bottomrightpoints[pt].y-topleftpoints[pt].y)/2.f);
#endif
		//fill in rest of fields. not clipping these?
		/*baseCands[pt].bb = Rect(cvRound(topleftpoints[pt].x),  cvRound(topleftpoints[pt].y),
			cvRound(bottomrightpoints[pt].x-topleftpoints[pt].x),cvRound(bottomrightpoints[pt].y-topleftpoints[pt].y));*/
		//piss. make sensible rects from these
		baseCands[pt].bb = Rect(
			cvRound(min(topleftpoints[pt].x, bottomrightpoints[pt].x)),
			cvRound(min(topleftpoints[pt].y, bottomrightpoints[pt].y)),
			abs(cvCeil(bottomrightpoints[pt].x - topleftpoints[pt].x)),
			abs(cvCeil(bottomrightpoints[pt].y - topleftpoints[pt].y))
			);
	}
#endif

	//sort these so motion-only points come last
	sort(baseCands.begin(), baseCands.end(), &compareTrackableDetections);
	//now take centroids and present to kalman filter
	if (draw){
		for (vector<TrackableDetection>::iterator idL = baseCands.begin(); idL != baseCands.end(); idL++){
			rectangle(base_img, idL->bb, colours.getColoursFromID(idL->source), 1);
			//draw cross at centre
			line(base_img, Point2f(idL->centre.x, idL->centre.y), Point2f(idL->centre.x + 10, idL->centre.y), CV_RGB(255, 0, 0));
			line(base_img, Point2f(idL->centre.x, idL->centre.y - 10), Point2f(idL->centre.x, idL->centre.y + 10), CV_RGB(255, 0, 0));
		}
	}

	//can now run kalman filter.
	size_t nDets = baseCands.size();	size_t nTracks = kfs.size();

	if (!nTracks){
		if (nDets){ //initialise one track for every det
			for (size_t j = 0; j < baseCands.size(); j++){
				kfs.push_back(ObjectTracker(baseCands[j].bb, baseCands[j].source));
			}
		}
		else {
			std::cout << "no tracks and no detections\n";
		}
	}
	else { //have tracks so update all the tracks
		for (list<ObjectTracker>::iterator i_kf = kfs.begin(); i_kf != kfs.end(); i_kf++){
			i_kf->predict();
			if (draw){
				circle(base_img, Point2f(i_kf->statePost.at<float>(0) - 10, i_kf->statePost.at<float>(1) - 10), 10,
					colours.getColoursFromID(i_kf->type));
				/*circle(base_img,Point2f(i_kf->statePost.at<float>(0),i_kf->statePost.at<float>(1)),loose_thresh_det2track,
					colours.getColoursFromID(i_kf->type),1);
					circle(base_img,Point2f(i_kf->statePost.at<float>(0),i_kf->statePost.at<float>(1)),thresh_det2track,
					colours.getColoursFromID(i_kf->type),1);*/
			}
		}
		if (nDets){
			vector<TrackableDetection> unassignedDets;
			// try and match detections to tracks. detections are rows, tracks are columns
			Mat dist_det2track = Mat_<float>(nDets, nTracks);
			//build matrix of euclidean distances between det and track centres
			size_t i = 0;
			for (vector<TrackableDetection>::iterator i_bc = baseCands.begin(); i_bc != baseCands.end(); i_bc++, i++){
				float* row_ptr = dist_det2track.ptr<float>(i);
				size_t j_dist = 0;
				for (list<ObjectTracker>::iterator j_kf = kfs.begin(); j_kf != kfs.end(); j_kf++, j_dist++){
					if (are_different_object_types(j_kf->type, i_bc->source)){
						row_ptr[j_dist] = numeric_limits<float>::infinity(); //invalid comparison ,set distance to sth really big
					}
					else { //valid so get euclidean distance
						row_ptr[j_dist] = sqrt(
							(j_kf->statePost.at<float>(0) - i_bc->centre.x) * (j_kf->statePost.at<float>(0) - i_bc->centre.x) +
							(j_kf->statePost.at<float>(1) - i_bc->centre.y)   *(j_kf->statePost.at<float>(1) - i_bc->centre.y));
					}
				}
			}

			float ecc_y_factor = 1.f / (1 - eccentricity*eccentricity);
			//loop over detections. can only match detction to 1 tracker
			vector<int> trackFlags(nTracks, -1);

			for (size_t i = 0; i < nDets; i++){
				bool matched = false;
				while (!matched){
					list<ObjectTracker>::iterator  bt_kf = kfs.end(); //best tracker pointer iterator
					int idx = -1; float minval = numeric_limits<float>::infinity();
					float* row_ptr = dist_det2track.ptr<float>(i);
					if (doSimpleD2TMatching){
						for (size_t j = 0; j < nTracks; j++){
							if (row_ptr[j] < minval){
								idx = j; minval = row_ptr[j];
							}
						}
					}
					else { //iterate over nTracks, carry a pointer to kfs with us too
						size_t j = 0;
						for (list<ObjectTracker>::iterator j_kf = kfs.begin(); j < nTracks; j++, j_kf++){
							if (row_ptr[j] < loose_thresh_det2track){ //pre-filter
								float t_angle = atan2f(j_kf->statePost.at<float>(3), j_kf->statePost.at<float>(2)) + CV_PI_F / 2.f;
								//clamp to 0-180deg
								if (t_angle < 0) {
									t_angle += CV_PI_F;
								}
								else {
									if (t_angle > CV_PI_F)
										t_angle -= CV_PI_F;
								}
								float dx = baseCands[i].centre.x - j_kf->statePost.at<float>(0);
								float dy = baseCands[i].centre.y - j_kf->statePost.at<float>(1);
								//we already have rho, its euclidean distance from row_ptr[j]
								float d2t_angle = atan2f(dy, dx) + CV_PI_F / 2.f;
								//clamp to 0-180deg
								if (d2t_angle < 0) {
									d2t_angle += CV_PI_F;
								}
								else {
									if (d2t_angle > CV_PI_F)
										d2t_angle -= CV_PI_F;
								}
								float diff_angle = abs(d2t_angle - t_angle);
								//now do pol2cart
								float x2 = row_ptr[j] * cos(diff_angle);
								float y2 = row_ptr[j] * sin(diff_angle);
								float dist = sqrt(x2 * x2 + ecc_y_factor * y2 * y2);
								//j_kf->statePost.at<float>(0), j_kf->statePost.at<float>(1),1.f);

								//done
								if (dist < minval){
									minval = dist;
									idx = j;
									bt_kf = j_kf;
								}
							}
						}
					}
					if (minval > thresh_det2track){
						unassignedDets.push_back(baseCands[i]);
						//nothing worked here, no tracker matched det.
						break;//out of while loop
					}
					else {
						if (frames2skip > 0){
							//hack something into existence to make tracks stay attached to stationary objects
							float factor;
							if (bt_kf->age < 20)
								factor = 1;
							else
								factor = max(0.2f, min(1.f, 2 * bt_kf->motionMagnitude()));
							float newthresh = thresh_det2track*factor;
							/*circle(base_img,Point2f(bt_kf->statePost.at<float>(0),bt_kf->statePost.at<float>(1)),newthresh,
								CV_RGB(0,0,0),1);*/
							if (minval > newthresh){ //throw away this match
								row_ptr[idx] = numeric_limits<float>::infinity();
							}

						}
						if ((baseCands[i].source == OBJECT_PED) || (baseCands[i].source == OBJECT_CAR)){
							//known objects get first crack at matching
							if (trackFlags[idx] == -1 || trackFlags[idx] == OBJECT_UNDETERMINED ||
								trackFlags[idx] == baseCands[i].source)
							{
								//set flags
								trackFlags[idx] = baseCands[i].source;
								matched = true;
								//update kf
								if (doSimpleD2TMatching){ //list walking expensive so only do it when we have to
									bt_kf = kfs.begin();
									for (int bt_i = 0; bt_i < idx; bt_i++)
										bt_kf++;
								}

								assert(bt_kf != kfs.end());
								bt_kf->correct(baseCands[i]);
							}
							else {//tracker now matched to object of different type. try again
								row_ptr[idx] = numeric_limits<float>::infinity();
								cout << "throwing away closest track (it was wrong type)\n";
							}

						}
						else { //det type is undetermined
							if (trackFlags[idx] == -1 || trackFlags[idx] == OBJECT_UNDETERMINED){
								//tracker not yet matched
								//set flags
								trackFlags[idx] = baseCands[i].source;
								matched = true;
								//update kf
								if (doSimpleD2TMatching){ //list walking expensive so only do it when we have to
									bt_kf = kfs.begin();
									for (int bt_i = 0; bt_i < idx; bt_i++)
										bt_kf++;
								}
								assert(bt_kf != kfs.end());
								bt_kf->correct(baseCands[i]);
							}
							else { //track has already been matched to a known object type this frame.
								//dont mix it with unknown patches.
								row_ptr[idx] = numeric_limits<float>::infinity();
								cout << "throwing away closest track (detection type undetermined)\n";
							}
						}
					}
				}
			}
			for (size_t j = 0; j < unassignedDets.size(); j++){//initialise unassigned detections
				kfs.push_back(ObjectTracker(unassignedDets[j].bb, unassignedDets[j].source));
			}
			//should really draw lines between dets and updated trackers here
		}
		mergeNearbyTracks(); //second chance tracker matching
	}
	//bin old or lost trackers and draw updated ones
	char msg[30];
	for (list<ObjectTracker>::iterator i_kf = kfs.begin(); i_kf != kfs.end();){
		if (i_kf->lastseen > track_lost_hard_thresh /* TODO and going too fast to be stationary*/ ||
			i_kf->lastseen > (track_lost_soft_thresh + i_kf->hits / 3) ||
			(i_kf->statePost.at<float>(0) < lost_xmin || i_kf->statePost.at<float>(0) > lost_xmax ||
			i_kf->statePost.at<float>(1) < lost_ymin || i_kf->statePost.at<float>(1) > lost_ymax) && i_kf->age > 10){
			//last line above gives new trackers outside window a few frames grace to move into frame
			//dont delete it yet as the clustering system needs to know that a transit has been completed.
			//i_kf = kfs.erase(i_kf);
			i_kf->lost = true;
			i_kf++; // if switch back to erasing here REMEBMER AND UNDERSTAND WHAT THIS LINE DOES
		}
		else {
			/////////point plotting////////////////////////////////////////////
			if (draw){
				/*circle(base_img,Point2f(i_kf->statePost.at<float>(0)-5,i_kf->statePost.at<float>(1)-5),
					10,colours.getColoursFromID(i_kf->type),2);
					//sprintf(msg,"%d",i_kf->id); //object id only
					//sprintf(msg,"%d/%2.1f",i_kf->id,i_kf->anom/i_kf->age); //object anomaly level (set in last frame)
					//sprintf(msg,"%d/%2.1f/%2.3f,%2.3f",i_kf->id,i_kf->anom/(i_kf->age -4),
					//i_kf->statePost.at<float>(2),i_kf->statePost.at<float>(3)); //object anomaly and tracker centre
					sprintf(msg,"%d|%2.3f,%2.3f|%3.fr%2.1f",i_kf->id,i_kf->statePost.at<float>(2),
					i_kf->statePost.at<float>(3), i_kf->motionBearingDegrees(), i_kf->motionMagnitude()); //object tracker centre
					putText(base_img,msg,Point2f(i_kf->statePost.at<float>(0)-10,i_kf->statePost.at<float>(1)-10),
					FONT_HERSHEY_SIMPLEX, 0.6,colours.getColoursFromID(i_kf->type+3),2);*/
				rectangle(base_img, i_kf->bb, CV_RGB(255, 0, 0));
				/*putText(base_img,msg,Point2f(i_kf->bb.x-10,i_kf->bb.y-10),
				FONT_HERSHEY_SIMPLEX, 0.8,colours.getColoursFromID(i_kf->type),1);*/
			}
			/////////end point plotting////////////////////////////////////////
			//if unseen by cluster, add seen points to record 
			if (i_kf->unseen){
				i_kf->record.push_back(Point2f(i_kf->statePost.at<float>(0), i_kf->statePost.at<float>(1)));
			}
			i_kf++;
		}
	}
}

void TrackerApp::transformTracksToCam(vector<Point3i>* trackedPoints){
	trackedPoints->clear();
	vector<Point2f> basePoints;
	vector<int> detTypes;
	//transform tracked points back to camera plane
	for (list<ObjectTracker>::iterator i_kf = kfs.begin(); i_kf != kfs.end(); i_kf++){
		if (!i_kf->lost){
			basePoints.push_back(Point2f(i_kf->statePost.at<float>(0), i_kf->statePost.at<float>(1)));
			detTypes.push_back(i_kf->type);
		}
	}
	vector<Point2f> camPoints;
	if (basePoints.size()) //only run if basepoints exists
		perspectiveTransform(basePoints, camPoints, H_base2cam);
	//shove these into a point3i and return
	trackedPoints->resize(camPoints.size());
	for (size_t j = 0; j < camPoints.size(); j++){
		trackedPoints->at(j) = Point3i((int)camPoints[j].x, (int)camPoints[j].y, detTypes[j]);
	}
}



int TrackerApp::transformAnomTracksToCam(vector<AnomalousDetection>* anomDets){
	anomDets->clear();
	vector<Point2f> basePoints;
	vector<int> anomAge, anomIDs;
	//transform anomalous tracked points back to camera plane
	for (list<ObjectTracker>::iterator i_kf = kfs.begin(); i_kf != kfs.end(); i_kf++){
		if (i_kf->anom_counter > 0){
			basePoints.push_back(Point2f(i_kf->statePost.at<float>(0), i_kf->statePost.at<float>(1)));
			anomAge.push_back(i_kf->anom_counter);
			anomIDs.push_back(i_kf->id);
		}
	}
	vector<Point2f> camPoints;
	if (basePoints.size()) //only run if basepoints exists
		perspectiveTransform(basePoints, camPoints, H_base2cam);
	//put these in an a vector of AnomalousDetection and return.
	anomDets->resize(camPoints.size());

	for (size_t j = 0; j < camPoints.size(); j++){
		anomDets->at(j).x = (int)camPoints[j].x;
		anomDets->at(j).y = (int)camPoints[j].y;
		anomDets->at(j).id = anomIDs[j];
		anomDets->at(j).duration = anomAge[j];
	}
	return anomDets->size();
}
