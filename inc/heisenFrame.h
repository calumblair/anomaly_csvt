#ifndef CB_HEISENFRAME_H
#define CB_HEISENFRAME_H
//store frame both on cpu and gpu, lazy evaluation and caching of same.
class HeisenFrame{
public:
	HeisenFrame(void);
	void setFrame(cv::Mat frame_);
	void setPatch(cv::Rect patch_loc, float zoom);

	cv::Mat getFrame(int mat_type);
	cv::gpu::GpuMat getGpuFrame(int mat_type);

	cv::gpu::GpuMat getGpuPatch(int mat_type);
	cv::Mat getPatch(int mat_type);

	bool usePatch;//flag for roi-based detectors to use patch instead of whole frame

private:
	std::vector<cv::Mat> frameCache;
	std::vector<cv::gpu::GpuMat> gpuFrameCache;

	std::vector<cv::Mat> patchCache;
	std::vector<cv::gpu::GpuMat> gpuPatchCache;

	cv::Rect patch_loc;
	float zoom;
};
#endif
