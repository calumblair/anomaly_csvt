#include "stdafx.h"
#include "acceleratedAlgorithm.h"
#include "tracking.h"
#include "display.h"
#include "unsupervisedHeatmap.h"
#include "trackingApp.h"

using namespace std;
using namespace cv;

//Kalman filter implementation; see thesis documentation and OpenCV example
ObjectTracker::ObjectTracker(Rect bb_, int objectType_){
	bb = bb_;
	type = objectType_;
	id = TrackerIDList::getNew();

	age = 0;
	lastseen = 0;
	lost = false;
	hits = 0;

	//clustering-specific stuff
	anom = 0.f;
	unseen = true;
	anom_counter = 0;

	//initialise KF for 2D tracking
	init(4, 2); //controlParams = 0, type = CV_32F
	//set all the things the KF needs
	/*A or F*/
	transitionMatrix = *(Mat_<float>(4, 4) << 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1);
	/* H */
	measurementMatrix = *(Mat_<float>(2, 4) << 1, 0, 0, 0, 0, 1, 0, 0);
	//these numbers come from and were tested in MATLAB
	/* Q */
	setIdentity(processNoiseCov, Scalar::all(1));
	/* R */

	setIdentity(measurementNoiseCov, Scalar::all(noise));

	/* P */ //(again, from Matlab): (H\R)/H'
	errorCovPost = *(Mat_<float>(4, 4) << noise, 0, 0, 0, 0, noise, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
	//alternatively
	//setIdentity(errorCovPost, Scalar::all(1)); //set to 1 for now
	//update the KF with the initial point
	statePost.at<float>(0) = (bb.x + bb.width / 2.f);
	statePost.at<float>(1) = (bb.y + bb.height / 2.f);
	statePost.at<float>(2) = 0;
	statePost.at<float>(3) = 0.1f;
}


const Mat& ObjectTracker::predict(const Mat& control){
	/*statePre =*/ KalmanFilter::predict(control);

	//update bb. assume size doesnt change
	bb.x = cvRound(statePost.at<float>(0) - bb.width / 2.f);
	bb.y = cvRound(statePost.at<float>(1) - bb.height / 2.f);

	age++;
	lastseen++;
	return statePre;
}


const Mat& ObjectTracker::correct(const Mat& measurement){
	/*statePost =*/ KalmanFilter::correct(measurement);
	hits++;
	lastseen = 0;
	return statePost;
}

const Mat& ObjectTracker::correct(const TrackableDetection& det){
	assert(!are_different_object_types(type, det.source));//check D&T arent incompatible
	if (type == OBJECT_UNDETERMINED)
		type = det.source; //only set if object type is unknown
	//unroll the detection and pass on
	KalmanFilter::correct(*(Mat_<float>(2, 1) << det.centre.x, det.centre.y));

	//update the bb
	bb.width = det.bb.width;
	bb.height = det.bb.height;

	bb.x = cvRound(statePost.at<float>(0) - bb.width / 2.f);
	bb.y = cvRound(statePost.at<float>(1) - bb.height / 2.f);

	/*statePost =*/
	hits++;
	lastseen = 0;
	return statePost;
}
