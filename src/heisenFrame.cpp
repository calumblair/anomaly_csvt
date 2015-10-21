#include "stdafx.h"
#include "acceleratedAlgorithm.h"
#include "heisenFrame.h"
#include "fpgaImageSize.h"

using namespace cv;
using namespace std;

//abstraction class to allow the rest of the program to not care whether an image exists in CPU or GPU memory (ie it's uncertain) and 
//location only gets evaluated when the image needs to be processed. 

HeisenFrame::HeisenFrame(void){
	frameCache.clear();
	gpuFrameCache.clear();
	patchCache.clear();
	gpuPatchCache.clear();
	zoom = 0;
	patch_loc = Rect(0, 0, 0, 0);
	usePatch = false;
}

void HeisenFrame::setFrame(cv::Mat frame_){
	frameCache.clear();
	frameCache.push_back(frame_);
	//invalidate everything else
	gpuFrameCache.clear();
	patchCache.clear();
	gpuPatchCache.clear();

	zoom = 0;
	patch_loc = Rect(0, 0, 0, 0);
	usePatch = false;
}

cv::Mat HeisenFrame::getFrame(int mat_type){
	for (size_t i = 0; i<frameCache.size(); i++){
		if (frameCache[i].type() == mat_type){
			return frameCache[i];
		}
	}

	//if no exact match, generate from existing if possible
	if (frameCache.size() && CV_MAT_CN(mat_type) == 1 &&
		CV_MAT_CN(frameCache[0].type()) > CV_MAT_CN(mat_type))
	{
		if (CV_MAT_CN(frameCache[0].type()) == 3){
			frameCache.push_back(Mat(0, 0, 0));
			cvtColor(frameCache[0], frameCache.back(), CV_BGR2GRAY);
			//cout << "hf cvtcol to gray"<<endl;
			assert(frameCache.back().type() == mat_type);
			return frameCache.back();
		}
		else if (CV_MAT_CN(frameCache[0].type()) == 4){
			frameCache.push_back(Mat(0, 0, 0));
			cvtColor(frameCache[0], frameCache.back(), CV_BGRA2GRAY);
			//cout << "hf cvtcol 4 to gray"<<endl;
			assert(frameCache.back().type() == mat_type);
			return frameCache.back();
		}
	}
	throw("unknown matrix type");
}

gpu::GpuMat HeisenFrame::getGpuFrame(int mat_type){
	//look for desired type in vector of processed types. if found, return it
	for (size_t i = 0; i<gpuFrameCache.size(); i++){
		if (gpuFrameCache[i].type() == mat_type){
			return gpuFrameCache[i];
		}
	}

	//if no exact match, generate from existing if possible
	//generate grey
	if (gpuFrameCache.size() && CV_MAT_CN(mat_type) == 1 &&
		CV_MAT_CN(gpuFrameCache[0].type()) > CV_MAT_CN(mat_type))
	{
		if (CV_MAT_CN(gpuFrameCache[0].type()) == 3){
			gpuFrameCache.push_back(gpu::GpuMat(0, 0, 0));
			gpu::cvtColor(gpuFrameCache[0], gpuFrameCache.back(), CV_BGR2GRAY);
			//cout << "hf gpu cvtcol 3 to gray only"<<endl;
			assert(gpuFrameCache.back().type() == mat_type);
			return gpuFrameCache.back();
		}
		else if (CV_MAT_CN(gpuFrameCache[0].type()) == 4){
			gpuFrameCache.push_back(gpu::GpuMat(0, 0, 0));
			gpu::cvtColor(gpuFrameCache[0], gpuFrameCache.back(), CV_BGRA2GRAY);
			//cout << "hf gpu cvtcol 4 to gray only"<<endl;
			assert(gpuFrameCache.back().type() == mat_type);
			return gpuFrameCache.back();
		}
	}

	if (gpuFrameCache.size() && (CV_MAT_CN(mat_type) == 4)){//convert from 3
		for (size_t i = 0; i < gpuFrameCache.size(); i++){
			if (CV_MAT_CN(gpuFrameCache[i].type()) == 3){
				gpuFrameCache.push_back(gpu::GpuMat(0, 0, 0));
				//cout << "hf gpu cvtcol 3to4 only"<<endl;
				gpu::cvtColor(gpuFrameCache[i], gpuFrameCache.back(), CV_BGR2BGRA);
				assert(gpuFrameCache.back().type() == mat_type);
				return gpuFrameCache.back();
			}
		}
	}
	if (gpuFrameCache.size() && (CV_MAT_CN(mat_type) == 3)){//convert from 4
		for (size_t i = 0; i < gpuFrameCache.size(); i++){
			if (CV_MAT_CN(gpuFrameCache[i].type()) == 4){
				gpuFrameCache.push_back(gpu::GpuMat(0, 0, 0));
				//cout << "hf gpu cvtcol 4to3 only"<<endl;
				gpu::cvtColor(gpuFrameCache[i], gpuFrameCache.back(), CV_BGRA2BGR);
				assert(gpuFrameCache.back().type() == mat_type);
				return gpuFrameCache.back();
			}
		}
	}

	//that didnt work, get from host
	for (size_t i = 0; i < frameCache.size(); i++){
		if (frameCache[i].type() == mat_type){
			gpuFrameCache.push_back(gpu::GpuMat(0, 0, 0));
			gpuFrameCache.back().upload(frameCache[i]);
			//cout << "hf gpu upload direct match"<<endl;
			assert(gpuFrameCache.back().type() == mat_type);
			return gpuFrameCache.back();
		}
	}

	//if no match, remake from scratch
	//cout << "hf gpu optimistic upload	"<<endl;
	gpuFrameCache.push_back(gpu::GpuMat(0, 0, 0));
	gpuFrameCache.back().upload(frameCache[0]);

	return getGpuFrame(mat_type); //severe potential for infinite recursion here
}

void HeisenFrame::setPatch(cv::Rect patch_loc_, float zoom_){
	zoom = zoom_;
	patch_loc = patch_loc_;

	usePatch = true; //set flag for detectors that use patches
	//clear existing patches
	patchCache.clear();
	gpuPatchCache.clear();
}

cv::Mat HeisenFrame::getPatch(int mat_type){
	Mat tmp_patch;
	for (size_t i = 0; i < patchCache.size(); i++){
		if (patchCache[i].type() == mat_type){
			return patchCache[i];
		}
	}
	bool found = false;
	//find one of same type to take a patch from
	size_t j = 0;
	for (; j < frameCache.size(); j++){
		if (frameCache[j].type() == mat_type){
			found = true;
			break;
		}
	}
	if (found){
		//printf("frameCahce[j].type() is %d and mat_type is %d. Depth of each is %d and %d and channels are %d and %d\n",
		//frameCache[j].type(),mat_type, frameCache[j].depth(),CV_MAT_DEPTH(mat_type), frameCache[j].channels(), CV_MAT_CN(mat_type));
		tmp_patch = frameCache[j](Range(patch_loc.y, patch_loc.y + patch_loc.height),
			Range(patch_loc.x, patch_loc.x + patch_loc.width));
		patchCache.push_back(Mat(0, 0, 0));
		resize(tmp_patch, patchCache.back(), Size(0, 0), zoom, zoom);
	}
	else{
		getFrame(mat_type);
		tmp_patch = getPatch(mat_type); //do like this so we can hit the assertion
	}
	//cout << "hf resize"<<endl;
	assert(patchCache.back().type() == mat_type);
	assert(patchCache.back().rows <= FPGA_ROWS && patchCache.back().cols <= FPGA_COLS);
	return patchCache.back();
	//?
	throw("unknown matrix type for patch");
}

gpu::GpuMat HeisenFrame::getGpuPatch(int mat_type){
	for (size_t i = 0; i < gpuPatchCache.size(); i++){
		if (gpuPatchCache[i].type() == mat_type){
			return gpuPatchCache[i];
		}
	}
	gpuPatchCache.push_back(gpu::GpuMat(0, 0, 0));
	gpu::GpuMat tmp = gpu::GpuMat::GpuMat(getGpuFrame(mat_type), patch_loc);

	gpu::resize(tmp, gpuPatchCache.back(), Size(0, 0), zoom, zoom);
	//cout << "hf resize in gpu"<<endl;
	assert(gpuPatchCache.back().type() == mat_type);
	return gpuPatchCache.back();

	throw("unknown gpu matrix type for patch");
}
