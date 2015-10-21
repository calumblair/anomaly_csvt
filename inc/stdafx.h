// stdafx.h : include file for standard system include files,
// or project specific include files that are used frequently, but
// are changed infrequently
//

#pragma once


#include <stdio.h>
#if defined _WIN32 || defined _WIN64
#include "targetver.h"
#include <tchar.h>
#endif

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <iomanip>
#include <stdexcept>
#include <limits>

//will need to link against bmd_lib.c/.h

#if defined _WIN32 || defined _WIN64
#include <WinSock2.h>
#include <WinBase.h>
#elif defined (__linux__) || defined (LINUX)
#include <unistd.h>
#ifndef LINUX
#define LINUX 1
#endif
#else
#error "unknown platform - not windows or linux"
#endif

#include <opencv2/core/core.hpp>
#include <opencv2/core/internal.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/video/tracking.hpp>

#define HAVE_BOOST 1
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <boost/bimap.hpp>
//threaded exception handling
#include <boost/exception/info.hpp>
#include <boost/exception/errinfo_errno.hpp>
#include <boost/exception_ptr.hpp>

#include <cuda_runtime_api.h> //for cudaDeviceReset
