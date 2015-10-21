//display.h
//header for image display helper functions
#ifndef CB_DISPLAY_FUNCTIONS
#define CB_DISPLAY_FUNCTIONS 1
//DisplayColourMap: given an implementation ID, it returns a consistent set of colours to be used with that ID
class DisplayColourMap{
public:
	CvScalar getColoursFromID(int id);
};

//frame reader which reads images from a folder set by setfolder(). this one uses cv::imread() in internalFrameReader
class YetAnotherFrameReader{
public: void setFolder(boost::filesystem::path folder);
		YetAnotherFrameReader();
		YetAnotherFrameReader(std::string folder);
		bool getFrame(cv::Mat* frame, int skip = 1);
protected:
	boost::filesystem::directory_iterator file_iter, iter_end_of_dir;
	void internalFrameReader(std::string, cv::Mat* newframe);
	boost::filesystem::path getFilename(int skip);
};


class ProcessingTimer{
public:
	ProcessingTimer();
	void workBegin();
	double workEnd();
	std::string workFps() const;
protected:
	double work_fps;
	int64 work_begin;
};

#endif
