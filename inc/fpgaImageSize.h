/*Image size: This is the important one*/
#define FPGA_IMAGE_IS_PAL 1
//#define FPGA_IMAGE_IS_CUSTOM 1
#ifndef FPGA_IMAGE_SIZE
#ifdef FPGA_IMAGE_IS_PAL 
#define FPGA_ROWS 578
#define FPGA_COLS 770
#else
#ifdef FPGA_IMAGE_IS_CUSTOM
#define FPGA_ROWS 58		
#define FPGA_COLS 450
#else
#define FPGA_ROWS 770
#define FPGA_COLS 1026
#endif
#endif

#define FPGA_PED_HOG_NBINS 9
#define FPGA_CAR_HOG_NBINS 18

#endif

#ifndef FPGA_TASKS
#define FPGA_TASKS
//object detection types
#define FPGA_HOG_GET_HISTS	(  1)
#define FPGA_HOG_GET_SCORES	(  2)
#define FPGA_USE_CAR_HOG	(  4)
#define FPGA_USE_PED_HOG	(  8)

#endif //FPGA_TASKS
