//fpgaInterface.cpp
//make sure fpgawrapper etc is compiled with C compiler & linked to c++ stuff
//CB 2/4/12

#include "stdafx.h"

#ifdef USE_WINDRIVER_FPGA_INTERFACE
#include "wdc_lib.h"
#include "../bmd_lib.h"


//debug option: don't do any work on FPGA and loopback data
//#define USE_PASSTHRU 1
using namespace std;
using namespace cv;
#include "fpgaInterface.h"
#include "fpga.h"

#define INTERFACE_DEBUG (0)
#define debug_printf(fmt, ...) \
	do { if (INTERFACE_DEBUG) fprintf(stderr, fmt, __VA_ARGS__); } while (0)


///////////////////////////////////////////////////////////////////////////////////////////////////
//size calculations for various fpga algorithms
UINT32 hogTransferSizeCalc(CvMat* img, int rrqGroupSize, UINT32* uIngressSizeB, UINT32* egressHistssz,
	UINT32* egressScoresz, UINT32* uEgressHistsTxSizeB, UINT32* uEgressScoreTxSizeB)
{
	assert(CV_MAT_TYPE(img->type) == CV_8UC1); //assert 8bit pixels
	//find ingress (host to fpga) transfer size : rows* cols * sizeof(pixel depth), rounded up to nearest
	//RRQ group size
	int imgsize = img->rows * img->cols;

	//flush application pipeline - appears to need 15 * 4096 bytes worth of data, about 7 cells worth.
	//if ( imgsize%rrqGroupSize) //if remainder
	*uIngressSizeB = (imgsize / rrqGroupSize + FPGA_PIPELINE_DEPTH_GROUPS) * rrqGroupSize;
	//has to be +>5 blocks to make sure the egress write completes
	//else
	//*uIngressSizeB = imgsize;
	//extra bytes: need a clear 4K to get proper output.
	printf(" Image is %d x %d and 8BPP. Size %d bytes (0x%08X) (Step is %d). Padded to nearest RRQ group (= %d) this is %d (0x%08X) bytes\n",
		img->rows, img->cols, img->rows * img->cols, img->rows * img->cols, img->step,
		rrqGroupSize, *uIngressSizeB, *uIngressSizeB);

	assert((*uIngressSizeB % FPGA_DMA_SIZE_INCREMENT) == 0);
	assert((*uIngressSizeB % rrqGroupSize) == 0); //this is the most efficient transfer size, and other sizes are not tested

	//calculate size of output histograms, egressHistSz
	*egressHistssz = ((img->rows - 2) / 8 * (img->cols - 2) / 8 * FPGA_PED_HOG_NBINS * 4);
	//calculate overhead needed for histograms, egressHistsTxSize(bytes)
	if (*egressHistssz % FPGA_EGRESS_AE_THRESH)
		*uEgressHistsTxSizeB = (*egressHistssz / FPGA_EGRESS_AE_THRESH + 1) * FPGA_EGRESS_AE_THRESH;
	else
		*uEgressHistsTxSizeB = *egressHistssz;
#if USE_PASSTHRU
	*uEgressHistsTxSizeB = *uIngressSizeB - 4096 *4; //if doing an interface test and streaming data straight through then use this value
#endif
	assert((*uEgressHistsTxSizeB % FPGA_DMA_SIZE_INCREMENT) == 0);
	assert((*uEgressHistsTxSizeB % FPGA_EGRESS_AE_THRESH) == 0); //if this is false, some hists will be left in the egress fifo at EOF

	//calculate size of output scores, egressScoresz
	//score bytes = (96 - upto 15) *(128+7)*4
	//actual scores are only 80*120*4
	//varying bottom porch varies the amount of junk we leave in the buffer
	*egressScoresz = (((img->cols - 2) / 8 + 7) * ((img->rows - 2) / 8 - FPGA_PED_HOG_VBLOCKS) * 4);
	//calculate overhead needed for scores, egressScoreTxSize(bytes)
	if (*egressScoresz % FPGA_EGRESS_AE_THRESH)
		*uEgressScoreTxSizeB = (*egressScoresz / FPGA_EGRESS_AE_THRESH + 1) * FPGA_EGRESS_AE_THRESH;
	else
		*uEgressScoreTxSizeB = *egressScoresz;
#if USE_PASSTHRU
	*uEgressScoreTxSizeB = *uIngressSizeB - 4096 *4; //if doing an interface test and streaming data straight through then use this value
#endif
	assert((*uEgressScoreTxSizeB % FPGA_DMA_SIZE_INCREMENT) == 0);
	assert((*uEgressScoreTxSizeB % FPGA_EGRESS_AE_THRESH) == 0); //if this is false, some hists will be left in the egress fifo at EOF

	printf("Egress from FPGA: Expecting %d cells with %d bins @4 bytes/float (should be 442368B for 1026*770)."
		" Need %d bytes or %d (0x%08X) bytes when rounded up\n", (img->rows - 2) / 8 * (img->cols - 2) / 8, FPGA_PED_HOG_NBINS, *egressHistssz,
		*uEgressHistsTxSizeB, *uEgressHistsTxSizeB);
	printf("Scores from FPGA: Expecting %d window scores @4 bytes/float (should be 43740B). Need %d bytes or %d (0x%08X) bytes when rounded up\n",
		((img->rows - 2) / 8 + 7) * ((img->cols - 2) / 8 - 15), *egressScoresz, *uEgressScoreTxSizeB, *uEgressScoreTxSizeB);

	//return the biggest buffer that we need
	return *uIngressSizeB + MAX(*uEgressScoreTxSizeB, *uEgressHistsTxSizeB);
}

UINT32 allHogTransferSizeCalc(CvMat* img, int algorithmType, int rrqGroupSize, UINT32* uIngressSizeB, UINT32* uEgressSizeB, bool display = 0)
//UINT32* egressHistssz,UINT32* egressScoresz, UINT32* uEgressHistsTxSizeB, UINT32* uEgressScoreTxSizeB)
{
	UINT32 uEgressDatasz = 1;
	assert(CV_MAT_TYPE(img->type) == CV_8UC1); //assert 8bit pixels
	//find ingress (host to fpga) transfer size : rows* cols * sizeof(pixel depth), rounded up to nearest
	//RRQ group size
	//flush application pipeline - appears to need 15 * 4096 bytes worth of data, about 7 cells worth.
	*uIngressSizeB = ((img->rows * img->cols) / rrqGroupSize + FPGA_PIPELINE_DEPTH_GROUPS) * rrqGroupSize;
	//has to be +>5 blocks to make sure the egress write completes
	//experimentally, need a clear 4K to get proper output.
	assert((*uIngressSizeB % FPGA_DMA_SIZE_INCREMENT) == 0);
	assert((*uIngressSizeB % rrqGroupSize) == 0); //this is the most efficient transfer size, and other sizes are not tested
	if (display){
		debug_printf("Image is %d x %d and 8BPP. Size %d bytes (0x%08X) (Step is %d). Padded to nearest RRQ group (= %d) this is %d (0x%08X) bytes\n",
			img->rows, img->cols, img->rows * img->cols, img->rows * img->cols, img->step,
			rrqGroupSize, *uIngressSizeB, *uIngressSizeB);
	}

	//output
	if (algorithmType & FPGA_HOG_GET_SCORES){
		//calculate size of output scores
		//score bytes = (96 - upto 15) *(128+7)*4
		//actual scores are only 80*120*4
		//varying bottom porch varies the amount of garbage we leave in the buffer
		if (algorithmType & FPGA_USE_CAR_HOG)
			uEgressDatasz = (((img->cols - 2) / 8 + 7) * ((img->rows - 2) / 8 - FPGA_CAR_HOG_VBLOCKS) * 4);
		else if (algorithmType & FPGA_USE_PED_HOG)
			uEgressDatasz = (((img->cols - 2) / 8 + 7) * ((img->rows - 2) / 8 - FPGA_PED_HOG_VBLOCKS) * 4);
	}
	else {//calculate size of output histograms
		if (algorithmType & FPGA_HOG_GET_HISTS){
			if (algorithmType & FPGA_USE_CAR_HOG)
				uEgressDatasz = ((img->rows - 2) / 8 * (img->cols - 2) / 8 * FPGA_CAR_HOG_NBINS * 4);
			else if (algorithmType & FPGA_USE_PED_HOG)
				uEgressDatasz = ((img->rows - 2) / 8 * (img->cols - 2) / 8 * FPGA_PED_HOG_NBINS * 4);
		}
	}//else whatever algorithms are needed

	//calculate overhead needed for scores/hists
	if (uEgressDatasz % FPGA_EGRESS_AE_THRESH)
		*uEgressSizeB = (uEgressDatasz / FPGA_EGRESS_AE_THRESH + 1) * FPGA_EGRESS_AE_THRESH;
	else
		*uEgressSizeB = uEgressDatasz;
#if USE_PASSTHRU
	*uEgressSizeB = *uIngressSizeB - 4096 *4; //if doing an interface test and streaming data straight through then use this value
#endif
	assert((*uEgressSizeB % FPGA_DMA_SIZE_INCREMENT) == 0);
	assert((*uEgressSizeB % FPGA_EGRESS_AE_THRESH) == 0);
	//if this is false, some hists will be left in the egress fifo at EOF
	if (display){
		printf("Data from FPGA: Expecting %d %s @4 bytes/float (should be 442368B for 1026*770). Need %d bytes or %d (0x%08X) bytes when rounded up\n",
			uEgressDatasz, (algorithmType & FPGA_HOG_GET_SCORES) ? "window scores" : "cell histograms",
			uEgressDatasz, *uEgressSizeB, *uEgressSizeB);
	}
	assert(uEgressDatasz != 1); //make sure it actually got set
	//return the biggest buffer that we need
	return *uIngressSizeB + *uEgressSizeB;
}

UINT32 salTransferSizeCalc(CvMat* img, int rrqGroupSize, UINT32* uIngressSizeB, UINT32* uEgressImgTxSizeB){
	int outImgSize;

	assert(CV_MAT_TYPE(img->type) == CV_8UC1); //assert 8bit pixels
	//find ingress (host to fpga) transfer size : rows* cols * sizeof(pixel depth), rounded up to nearest
	//RRQ group size
	//flush application pipeline - need a clear 4K to get proper output.
	*uIngressSizeB = (img->rows * img->cols / rrqGroupSize + 1/*FPGA_PIPELINE_DEPTH_GROUPS*/) * rrqGroupSize;

	printf("Image is %d x %d and 8BPP. Size %d bytes (0x%08X) (Step is %d). Padded to nearest RRQ group (= %d) this is %d (0x%08X) bytes\n",
		img->rows, img->cols, img->rows * img->cols, img->rows * img->cols, img->step,
		rrqGroupSize, *uIngressSizeB, *uIngressSizeB);
	assert((*uIngressSizeB % FPGA_DMA_SIZE_INCREMENT) == 0);
	assert((*uIngressSizeB % rrqGroupSize) == 0); //this is the most efficient transfer size, and other sizes are not tested

	//calculate size of output image
	outImgSize = img->rows * img->cols * sizeof(float);
	//calculate overhead needed for output image, uEgressImgTxSizeB(bytes)
	//might need to output this back to the calling class
	if (outImgSize % FPGA_EGRESS_AE_THRESH)
		*uEgressImgTxSizeB = (outImgSize / FPGA_EGRESS_AE_THRESH + 1) * FPGA_EGRESS_AE_THRESH;
	else
		*uEgressImgTxSizeB = outImgSize;
#if USE_PASSTHRU
	*uEgressImgTxSizeB = *uIngressSizeB - 4096 *4; //if doing an interface test and streaming data straight through then use this value
#endif
	assert((*uEgressImgTxSizeB % FPGA_DMA_SIZE_INCREMENT) == 0);
	assert((*uEgressImgTxSizeB % FPGA_EGRESS_AE_THRESH) == 0); //if this is false, some hists will be left in the egress fifo at EOF

	printf("Egress from FPGA: Expecting image of %d x %d at %d bytes/pixel. Need %d bytes or %d (0x%08X) bytes when rounded up\n",
		img->rows, img->cols, sizeof(float), outImgSize, *uEgressImgTxSizeB, *uEgressImgTxSizeB);

	//return the biggest buffer that we need
	return *uIngressSizeB + *uEgressImgTxSizeB;
}


///////////////////////////////////////////////////////////////////////////////////////////////////

/*Internal interface to FPGA - only access this through fpgaHOGProcessor as it needs the Windriver libraries to build*/
/*This needs to be reinitialised each time the input image dimensions change */
class FPGAInterface
{
private:
	//constructors
	FPGAInterface(const Mat& img);
	FPGAInterface(); //default constructor should never be called
	//taken from Thinking in C++ Vol 2 page623 (Meyers pattern)
	void operator=(FPGAInterface&);// not allowed
	FPGAInterface(const FPGAInterface&); //not allowed

	bool isInitialised;
	float* cellHistsfp; // pointer to base of cell histograms in
	//DMA buffer.

	//Physical addresses for DMA buffer & transfers
	UINT32 IngressLowerAddr; // 64bit input addr, constant
	UINT32 IngressUpperAddr; //set in fpgaDMABufferSetup, constant
	UINT32 EgressLowerAddr;// 64bit output addr, constant
	UINT32 EgressUpperAddr; //set in fpgaDMABufferSetup, constant

	UINT32 rrqGroupSizeB; //read request group size, set in fpgaGetPayloadSizes, constant
	UINT32 maxEgressTxSizeB; //set in constructor

	UINT32 uIngressSizeB; //all the following are algorithm-specific

	UINT32 egressDatasz;
	UINT32 uEgressSizeB; //egress size after padding

	DWORD option;
	static WDC_DEVICE_HANDLE hDev;
	static DIAG_DMA dma;

	// declare the C-specific Windriver functions as friends.
	friend DWORD fpgaInterfaceInit_c(WDC_DEVICE_HANDLE hDev, DIAG_DMA* dma, WDC_DEVICE_HANDLE* tmp_hDev);
	friend DWORD fpgaInterfaceDestructor_c(WDC_DEVICE_HANDLE* hDev, DIAG_DMA* dma);
	friend int fpgaProcessImage(WDC_DEVICE_HANDLE hDev, PDIAG_DMA pDma, CvMat* img, int operationType,
		UINT32 IngressLowerAddr, UINT32 IngressUpperAddr, UINT32 EgressLowerAddr, UINT32 EgressUpperAddr,
		int egressHistssz, UINT32 uIngressSizeB, UINT32 uEgressSizeB);

public:
	~FPGAInterface();
	//Core getData function returns a pointer to the floats in the base of the DMA hist buffer
	//if isScores is false, return the cell hists
	//else return the hog window scores
	int getData(const Mat& img, float** cellHistsfp, int operationDetails = FPGA_USE_CAR_HOG | FPGA_HOG_GET_HISTS);
	static FPGAInterface& getInstance(const Mat& img){
		static FPGAInterface fpga(img); //single instance
		return fpga;
	}
	cv::Size init_imgsz; //Size that the buffer was set up for: determined at initialisation
	int cellhist_sz;
};

//Initialise & define class variables referenced in FPGAInterface class
WDC_DEVICE_HANDLE FPGAInterface::hDev = (void *)0xdeadbeef;
DIAG_DMA FPGAInterface::dma;

///////////////////////////////////////////////////////////////////////////////////////////////
// FPGAInterface class rewritten as a singleton, as seen here
// http://stackoverflow.com/questions/1008019/c-singleton-design-pattern
// so that we can easily call from HOGProcessor and anything else.


//FPGAInterface constructor
//takes a pointer to the image size & type it should be constructed for
//and a function pointer to calculate buffer sizes.

//make sure we initialise the application with the 
//biggest buffer first.
//this is currently the car_hog implementation(with 18 bins per cell)
FPGAInterface::FPGAInterface(const Mat& img)
{
	WDC_DEVICE_HANDLE tmp;
	DWORD status;
	float* tmp_float_ptr = NULL;

	cout << "Initialising windriver FPGA interface" << endl;
	status = fpgaInterfaceInit_c(&hDev, &dma, &tmp);
	//hDev doesn't get set properly so this returns a tmp hDev,
	//which we assign to the static hDev
	hDev = (WDC_DEVICE_HANDLE)tmp;
	if (status) {
		cout << "FPGAInterface failed to initialise, windriver error " << status <<
			". Image buffer not created." << endl;
		//throw back to main thread
		throw boost::enable_current_exception(fpga_not_found_error("FPGA not found."));
	}
	else {
		cout << "fpga found & device handle opened OK" << endl;
		Mat img_grey;
		assert(img.channels() == 1);
		if (img.channels() != 1) { //shouldnt happen
			img_grey = Mat(img.rows, img.cols, CV_8UC1);
			cvtColor(img, img_grey, CV_BGR2GRAY);
		}
		else
			img_grey = img;

		//Now set up the image buffers
		cout << "Calculating buffer sizes using greyscale version of source image and CAR_HOG algorithm" << endl;
		init_imgsz = img_grey.size();
		cellhist_sz = (img_grey.rows - 2) / 8 * (img_grey.cols - 2) / 8 * FPGA_CAR_HOG_NBINS;
		printf("Input image has dimensions %d r x %d c\n", img_grey.rows, img_grey.cols);
		printf("creating (largest possible: car) output histogram matrix with dimensions %d x %d\n", cellhist_sz, 1);
		CvMat grey_mat = img_grey;

		//initialising the interface is now in 3 stages
		//first we try a PIO read from it, and get the payload sizes for RX and TX
		//read request size should be 512
		//note that calling this will close any existing open buffers
		rrqGroupSizeB = fpgaGetPayloadSizes(hDev, &dma);

		//knowing the read request size, we now calculate the buffer size we need
		//and how to split that up (ie what the ingress and egress addresses should be.
		//this is specific to each application (eg HOG)
		//we modfiy this slightly by initialising with the largest application 
		//and re-run each function below as needed
		allHogTransferSizeCalc(&grey_mat, (FPGA_USE_CAR_HOG | FPGA_HOG_GET_HISTS), rrqGroupSizeB, &uIngressSizeB, &uEgressSizeB, true);
		maxEgressTxSizeB = uEgressSizeB;

		//finally we open a DMA buffer of the required size, and obtain 
		//pointers to physical memory 
		status = fpgaOpenDMABuffer(hDev, &dma, uIngressSizeB, uEgressSizeB,
			&IngressLowerAddr, &IngressUpperAddr, &EgressLowerAddr, &EgressUpperAddr);
		if (!status)
			cout << "FPGA DMA buffer setup OK" << endl;
		else
			cout << "Error " << status << " while setting up DMA buffer " << endl;

		//Assign pointer to start of cell histogram
		tmp_float_ptr = (float*)((char*)dma.pBuf + uIngressSizeB);
		cellHistsfp = tmp_float_ptr;
		isInitialised = true;
	}
}

//FPGAInterface default constructor: has no parameters, should never be called
FPGAInterface::FPGAInterface()
{
	cout << "Warning: FPGAInterface default constructor called. This takes no parameters " <<
		"and the fpga link has not been set up " << endl;
}

//FPGAInterface destructor
FPGAInterface::~FPGAInterface()
{
	cout << "FPGAInterface destructor called " << endl;
	DWORD status = fpgaInterfaceDestructor_c(&hDev, &dma);
	if (!status)
		isInitialised = false;
}

//getCells core interface
int FPGAInterface::getData(const Mat& img, float** cellHistsfp_out, int operationDetails)
{
	int status;
	//TODO Calculation/assertion of image size

	static Mat cellhists(1, cellhist_sz, CV_32FC1);
	static CvMat grey_cvmat;

	if (!isInitialised)
		assert(0);

	//if img_rgb is RGB, convert to greyscale
	assert(img.channels() == 1);
	if (img.channels() != 1) {
		static Mat img_grey_tmp = Mat(img.rows, img.cols, CV_8UC1);
		cvtColor(img, img_grey_tmp, CV_BGR2GRAY);
		grey_cvmat = img_grey_tmp;
	}
	else
		grey_cvmat = img;

	allHogTransferSizeCalc(&grey_cvmat, operationDetails, rrqGroupSizeB, &uIngressSizeB, &uEgressSizeB);

	assert(grey_cvmat.cols == init_imgsz.width); //size of this frame must be the same size as we
	assert(grey_cvmat.rows <= init_imgsz.height); //initialised with
	status = (int)fpgaProcessImage(hDev, &dma, &(grey_cvmat), operationDetails,
		IngressLowerAddr, IngressUpperAddr, EgressLowerAddr, EgressUpperAddr,
		egressDatasz, uIngressSizeB, uEgressSizeB);
	//block here until the cell histogram data transfer is completed

	*cellHistsfp_out = cellHistsfp;
	return status;
}

//core of fpgaHOGProcessor: keep a static instance of FPGAInterface and call it without knowing from
//what direction the calls to hogcore come from (ie GPUMat or Mat)
int fpgaHOGProcessor::hogcore(const Mat& img, float** hists_buf_ptr, int operationDetails)
{
	static bool isInitialised = false;
	if (!useFPGA) {
		//either we are on the laptop or the fpga is not connected
		//at this point we can either return zeros or load previously calculated data
		//for that fpga method for that frame in that video or file, or 
		//just return a matrix of zeros.
		//assume we are passing in 32bit floating-point data
		//read in cell hist floats from binary file or matlab file

		bool useFPGAHistData = false; //switch to TRUE to use
		//use histogram dump from a previous run rather than a matlab reference
		int exp_cellhist_sz = (img.rows - 2) / 8 * (img.cols - 2) / 8 * FPGA_PED_HOG_NBINS;
		static int status;
		static bool dataLoaded = false; //applies if we are processing a single frame
		static float* dummy_fpga_results_ptr_;
		if (!dataLoaded) {
#define HOSTNAMELEN 50
			DWORD hostNameLen = HOSTNAMELEN;
			bool isLaptop = false;

			//work out hostname - this isn't very important as the useFPGA flag should already have been set elsewhere
			//if msvc errors here, turn off unicode in project settings-> general->character set. 
#ifdef WIN32
			TCHAR thisHost[HOSTNAMELEN];
			if (GetComputerName(thisHost, &hostNameLen)) //if no errors - returns nonzero if succeeded
#else
			char thisHost[HOSTNAMELEN];
			// non-windows/cross-platform version:
			if (!gethostname(thisHost, hostNameLen)) //returns 0 if success
#endif
				isLaptop = !STRING_COMPARE(thisHost, laptopName.c_str());

			*hists_buf_ptr = (float*)malloc(exp_cellhist_sz * sizeof(float)); //only do this once otherwise lots of memory used
			dummy_fpga_results_ptr_ = *hists_buf_ptr;
			cell_hist_sz = exp_cellhist_sz;
			cout << "No FPGA connected, using " << cell_hist_sz << " bytes or " << cell_hist_sz / 4 << " entries of zeros" << endl;
			memset(dummy_fpga_results_ptr_, 0, cell_hist_sz);
			status = 0;
			dataLoaded = true;
		}
		else
			*hists_buf_ptr = dummy_fpga_results_ptr_;
		return status;
	}
	else { //actually use FPGA
		static FPGAInterface& fpga = FPGAInterface::getInstance(img); //construct only with image type
		cell_hist_sz = fpga.cellhist_sz;
		return fpga.getData(img, hists_buf_ptr, operationDetails); //FPGA_CAR_HOG | FPGA_GET_SCORES
	}
}
#endif
