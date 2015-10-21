#include "stdafx.h"
#include <cuda_profiler_api.h>
#include <nvToolsExt.h>
#include "profiling.h"

//generate information for the NVIDIA graphical profiler
void profilerNullA(const char*)
{
	;
}

nvtxRangeId_t profilerStartNull(const char*)
{
	return 0;
}

void profilerEndNull(nvtxRangeId_t)
{
	;
}

void profilerMarkTimerA(const char* id)
{
	printf("timer %s: \t\t%lu\n", id, cv::getCPUTickCount());
}
