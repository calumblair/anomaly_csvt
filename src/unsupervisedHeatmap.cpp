#include "stdafx.h"
#include <time.h>
#include "acceleratedAlgorithm.h"
#include "tracking.h"
#include "display.h"
#include "unsupervisedHeatmap.h"
#include "trackingApp.h"


using namespace cv;
using namespace std;


UnsupervisedHeatmap::UnsupervisedHeatmap(void){
	scale = 1;
	init(Size(0, 0), scale);
	//	learningRate = 0.005f;
	learningRate = 0.002f;

};

inline Rect UnsupervisedHeatmap::downScaleRect(Rect r_, int scale){
	return Rect(r_.x / scale, r_.y / scale, r_.width / scale, r_.height / scale);
}

void UnsupervisedHeatmap::init(Size mapSize_, int scale_){
	scale = scale_;
	origMapSize = mapSize_;
	mapSize = Size(mapSize_.width / scale, mapSize_.height / scale); //?
	x_ped_map = Mat::zeros(mapSize, CV_32FC1);
	y_ped_map = Mat::zeros(mapSize, CV_32FC1);
	x_car_map = Mat::zeros(mapSize, CV_32FC1);
	y_car_map = Mat::zeros(mapSize, CV_32FC1);
}

//this works on an unclipped, unscaled rect
void UnsupervisedHeatmap::updateRegion(cv::Rect roi_, float xval, float yval, int type){
	Rect roi = downScaleRect(clipRect(roi_, origMapSize), scale);
	if (type == OBJECT_PED){
		addWeighted(x_ped_map(roi), 1 - learningRate, Scalar(xval), learningRate, 0, x_ped_map(roi));
		addWeighted(y_ped_map(roi), 1 - learningRate, Scalar(yval), learningRate, 0, y_ped_map(roi));
	}
	else{
		float lrate = learningRate;
		if (type == OBJECT_UNDETERMINED)
#ifdef BANK_ST
			return;
#endif
		lrate *= 0.5f;//half weight to undet objects
		addWeighted(x_car_map(roi), 1 - lrate, Scalar(xval), lrate, 0, x_car_map(roi));
		addWeighted(y_car_map(roi), 1 - lrate, Scalar(yval), lrate, 0, y_car_map(roi));
	}
}

//do all trackers in one step.
void UnsupervisedHeatmap::updateAll(std::list<ObjectTracker>* trackers, int age_thresh, bool alsoUpdateTrackers){
	for (list<ObjectTracker>::iterator i_kf = trackers->begin(); i_kf != trackers->end(); i_kf++)
	{
		if ((i_kf->age > age_thresh) && !i_kf->lost){
			if (alsoUpdateTrackers){ //update tracker BEFORE we take into account the contribution of this object to heatmap
				float xm = 0, ym = 0, obj_current_anom = 0;
				float xv = 0, yv = 0;
				int id = i_kf->id;
				getRegionMean(i_kf->bb, i_kf->type, &xm, &ym, &xv, &yv);
				//anomaly calculation goes here
				float rho_m = sqrt(xm*xm + ym*ym);
				float theta_m = atan2f(ym, xm) * 180 / CV_PI_F + 90; theta_m = theta_m < 0 ? theta_m + 360 : theta_m;

				float theta_o = i_kf->motionBearingDegrees();
				float rho_o = i_kf->motionMagnitude();

				float theta_diff = MIN(abs(theta_o - theta_m), 360.f - abs(theta_o - theta_m));
				float rho_diff = 0;//???

				obj_current_anom = sqrt((xm - i_kf->statePost.at<float>(2))*(xm - i_kf->statePost.at<float>(2)) / xv +
					(ym - i_kf->statePost.at<float>(3))*(ym - i_kf->statePost.at<float>(3)) / yv);
				i_kf->anom += obj_current_anom;
			}
			updateRegion(i_kf->bb, i_kf->statePost.at<float>(2), i_kf->statePost.at<float>(3), i_kf->type);
		}
	}
}

//get region results: mean and/or stddev. works on unclipped rect
void UnsupervisedHeatmap::getRegionMean(cv::Rect roi_, int type, float* x_mean, float* y_mean,
	float* x_var, float* y_var)
{
	Rect roi = downScaleRect(clipRect(roi_, origMapSize), scale);
	Scalar mean_m, stddev_m;
	if (type == OBJECT_PED){
		meanStdDev(x_ped_map(roi), mean_m, stddev_m);
		*x_mean = (float)mean_m[0];
		*x_var = (float)stddev_m[0];
		meanStdDev(y_ped_map(roi), mean_m, stddev_m);
		*y_mean = (float)mean_m[0];
		*y_var = (float)stddev_m[0];
	}
	else {
		meanStdDev(x_car_map(roi), mean_m, stddev_m);
		*x_mean = (float)mean_m[0];
		*x_var = (float)stddev_m[0];
		meanStdDev(y_car_map(roi), mean_m, stddev_m);
		*y_mean = (float)mean_m[0];
		*y_var = (float)stddev_m[0];
	}
}

void UnsupervisedHeatmap::getRegionData(cv::Rect clipped_roi, int type, char direction, Mat& outmat)
{
	Rect roi = downScaleRect(clipped_roi, scale);
	if (type == OBJECT_PED){
		if (direction == 'y')
			outmat = y_ped_map(roi);
		else
			outmat = x_ped_map(roi);
	}
	else {
		if (direction == 'y')
			outmat = y_car_map(roi);
		else
			outmat = x_car_map(roi);
	}
}

//read in a pre-learned heatmap.
//'../data/', 'ilids_pv3','unsupervised_x.tiff' should work
//todo do a scaling thing to read in and multiply to convert from whatever to 32FC1
//TODO change these to draw car maps
void UnsupervisedHeatmap::read(string heatmap_path, string setname, string mapname){
	try{
		string date, filename;
		int w = 0, h = 0, scale = 0;
		Size sz;
		filename = heatmap_path + setname + mapname + ".xml";
		FileStorage heatmap_fs(filename, FileStorage::READ);
		heatmap_fs["collateDate"] >> date;

		//cv::Size has no overload so reconstruct from width and height
		heatmap_fs["origMapSize_w"] >> w;
		heatmap_fs["origMapSize_h"] >> h;
		heatmap_fs["scale"] >> scale;
		sz = Size(w, h);
		init(sz, scale);

		//check reconstructed mapsize matches stored one
		heatmap_fs["mapSize_w"] >> w;
		heatmap_fs["mapSize_h"] >> h;
		sz = Size(w, h);
		assert(sz == mapSize);

		heatmap_fs["x_car"] >> x_car_map;
		heatmap_fs["y_car"] >> y_car_map;
		heatmap_fs["x_ped"] >> x_ped_map;
		heatmap_fs["y_ped"] >> y_ped_map;

		assert(x_car_map.size() == sz);
		cout << "heatmaps built on " << date << "and of " <<
			mapSize.width << "wx" << mapSize.height << "h read from " << filename << endl;
		heatmap_fs.release();
	}
	catch (exception e){
		cout << "failed to load unsupervised context heatmaps" << e.what() << endl;
	}
}

//write out to file
void UnsupervisedHeatmap::write(string heatmap_path, string setname, string mapname){
	try{
		string filename = heatmap_path + setname + mapname + ".xml";

		FileStorage heatmap_fs(filename, FileStorage::WRITE);
		time_t rawtime; time(&rawtime);
		heatmap_fs << "collateDate" << asctime(localtime(&rawtime));
		//can't write size directly...
		heatmap_fs << "origMapSize_w" << origMapSize.width;
		heatmap_fs << "origMapSize_h" << origMapSize.height;
		heatmap_fs << "scale" << scale;
		heatmap_fs << "mapSize_w" << mapSize.width;
		heatmap_fs << "mapSize_h" << mapSize.height;
		heatmap_fs << "x_car" << x_car_map;
		heatmap_fs << "y_car" << y_car_map;
		heatmap_fs << "x_ped" << x_ped_map;
		heatmap_fs << "y_ped" << y_ped_map;
		cout << "heatmaps written to file on collate date " << asctime(localtime(&rawtime)) << endl;
		heatmap_fs.release();
	}
	catch (exception e){
		cout << "failed to write unsupervised context heatmaps" << e.what() << endl;
	}
}


void LocationLogger::updateRegion(cv::Rect unclipped_roi, float xval, float yval, int type){
	Rect roi = downScaleRect(clipRect(unclipped_roi, origMapSize), scale);
#ifdef BANK_ST
	if (type == OBJECT_UNDETERMINED)
		return;
#endif
	if (type == OBJECT_PED){
		add(x_ped_map(roi), 1, x_ped_map(roi));
	}
	else{
		//give half weight to undet objects
		float lrate = (type == OBJECT_UNDETERMINED) ? 0.5f : 1.f;
		add(x_car_map(roi), Scalar(lrate), x_car_map(roi));
	}
}

//do all trackers in one step.
void LocationLogger::updateAll(std::list<ObjectTracker>* trackers, int age_thresh, bool alsoUpdateTrackers){
	for (list<ObjectTracker>::iterator i_kf = trackers->begin(); i_kf != trackers->end(); i_kf++)
	{
		if ((i_kf->age > age_thresh) && !i_kf->lost){
			if (alsoUpdateTrackers){
				;//no functionality to update trackers
			}
			LocationLogger::updateRegion(i_kf->bb, i_kf->statePost.at<float>(2), i_kf->statePost.at<float>(3), i_kf->type);
		}
	}
}
