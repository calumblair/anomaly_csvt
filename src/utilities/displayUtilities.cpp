//display utilities or helper functions and classes for display of images etc.
#include "stdafx.h"
#include "display.h"
#include "cb_opencv.h"


using namespace cv;
using namespace boost::filesystem;
using namespace std;
//getColoursFromID returns a 3-colour scalar consistent with the source ID of each detection
CvScalar DisplayColourMap::getColoursFromID(int id){
	int paint_red, paint_blue, paint_green;
	switch (id){ //blue green red  purple  yellow again
	case 1: //blue
		paint_red = 000;	paint_green = 000;	paint_blue = 255;		break;
	case 2: //green
		paint_red = 000;	paint_green = 255;	paint_blue = 000;		break;
	case 3://red
		paint_red = 255;	paint_green = 000;	paint_blue = 000;		break;
	case 4://yellow
		paint_red = 255;	paint_green = 255;	paint_blue = 000;		break;
	case 5: //sort of darker cyan
		paint_red = 000;	paint_green = 191;	paint_blue = 191;		break;
	case 6: //cyan
		paint_red = 000;	paint_green = 255;	paint_blue = 255;		break;
	case 7: //purple
		paint_red = 255;	paint_green = 000;	paint_blue = 255;		break;
	case 8: //orange
		paint_red = 255;	paint_green = 128;	paint_blue = 000;		break;
	case 9: //white
		paint_red = 255;	paint_green = 255;	paint_blue = 255;		break;
	case 99://white again
		paint_red = 255;	paint_green = 255;	paint_blue = 255;		break;
	default://orange again
		paint_red = 255;	paint_green = 128;	paint_blue = 0;
	}
	return CV_RGB(paint_red, paint_green, paint_blue);
}

//YetanotherFrameReader - wrapper for imread(), reads all (image) files in a directory
YetAnotherFrameReader::YetAnotherFrameReader(){
	setFolder(path("."));
}
YetAnotherFrameReader::YetAnotherFrameReader(std::string folder){
	setFolder(folder);
}

void YetAnotherFrameReader::setFolder(path folder){
	file_iter = directory_iterator(folder);
}

//inherit this class and overload this function to use different image reading methods
void YetAnotherFrameReader::internalFrameReader(std::string filename, cv::Mat* framePtr){
	*framePtr = imread(filename);
}

boost::filesystem::path YetAnotherFrameReader::getFilename(int skip){
	path current_file = file_iter->path();
	if (skip > 1){
		//we need to do something special here as 
		//boost's directory_iterator doesn't allow us to jump multiple files
		//what we should do is rewrite this class to build a list of image files in a folder, keep an index into them,
		//and increment the index. alternatively:
		try{
			for (skip--; skip > 0; skip--)
				file_iter++;
			current_file = file_iter->path();
		}
		catch (const exception& e)
		{	//if we run out of files, return 0. (alternatively, loop)
			cout << e.what() << endl;
#ifdef LINUX
			return "";
#else
			return false;
#endif
		}
	}
	else
		file_iter++;
	return current_file;
}

bool YetAnotherFrameReader::getFrame(Mat* frame, int skip){ //skip is frameskip
	path current_file = getFilename(skip);
	internalFrameReader(current_file.string(), frame);
	bool success = (frame->rows != 0);
	if (!success)
		cout << "couldn't read any more files from folder " << file_iter->path() << endl;
	return success;
}

ProcessingTimer::ProcessingTimer() { work_fps = 0; work_begin = 0; };
void ProcessingTimer::workBegin() { work_begin = cv::getTickCount(); }

double ProcessingTimer::workEnd()
{
	//looks like it's best to use getTickCount rather than getCPUTickCount here
	//more consistent although possibly less accurate
	int64 delta = cv::getTickCount() - work_begin;
	double freq = cv::getTickFrequency();
	// timer works differently on different machines (sigh)
	//#ifdef GLASD470077 // winXP
	work_fps = delta * 1000 / freq;//freq / delta;
	//#else
	//	work_fps = delta/freq;//freq / delta;
	//#endif
	return work_fps;
}

std::string ProcessingTimer::workFps() const
{
	std::stringstream ss;
	ss << work_fps;
	return ss.str();
}
