//#define CB_GPU_PROFILING 1
//#define CB_SIMPLE_PROFILING 1
#ifdef  CB_GPU_PROFILING
#ifndef CB_GPU_PROFILING_DEFS
#define profilerMarkA(x) nvtxMarkA(x)
#define profilerRangeStartA(x) nvtxRangeStartA(x)
#define profilerRangeEnd(x) nvtxRangeEnd(x)
#endif
#else 
#ifdef  CB_SIMPLE_PROFILING
#ifndef CB_GPU_PROFILING_DEFS
#define profilerMarkA(x) profilerMarkTimerA(x)
#define profilerRangeStartA(x) profilerStartNull(x)
#define profilerRangeEnd(x) profilerEndNull(x)
#endif

#ifdef __cplusplus
extern "C" {
#endif

	void profilerMarkTimerA(const char*);

#ifdef __cplusplus
}
extern "C" {
#endif

	nvtxRangeId_t profilerStartNull(const char*);

#ifdef __cplusplus
}
extern "C" {
#endif

	void profilerEndNull(nvtxRangeId_t);

#ifdef __cplusplus
}
#endif

#else
#ifndef CB_GPU_PROFILING_DEFS
#define CB_GPU_PROFILING_DEFS
#define profilerMarkA(x) profilerNullA(x)
#define profilerRangeStartA(x) profilerStartNull(x)
#define profilerRangeEnd(x) profilerEndNull(x) 
#endif

#ifdef __cplusplus
extern "C" {
#endif

	void profilerNullA(const char*);

#ifdef __cplusplus
}
extern "C" {
#endif

	nvtxRangeId_t profilerStartNull(const char*);

#ifdef __cplusplus
}
extern "C" {
#endif

	void profilerEndNull(nvtxRangeId_t);

#ifdef __cplusplus
}
#endif
#endif //CB_GPU_PROFILING_DEFS
#endif //CB_GPU_PROFILING
