#include <stdio.h>
#include <math.h>
#ifdef ENABLE_WINDRIVER_FPGA_INTERFACE
#include "wdc_lib.h"
#include "utils.h"
#ifdef LINUX
//include definition for min ()
//because stdlibs dont have a definition for it 
#include <sys/param.h>
#endif

//include headers etc from Windriver toolkit
#include "status_strings.h"
#include "samples/shared/diag_lib.h"
#include "samples/shared/wdc_diag_lib.h"
#include "samples/shared/pci_regs.h"
#include <bmd_lib.h>


#include <opencv2/core/core_c.h>
#include <opencv2/highgui/highgui_c.h>
#include "fpgaInterface.h"
extern  BOOL IsValidDmaHandle(BMD_DMA_HANDLE, CHAR *);

//if profiling
#include <nvToolsExt.h>
#include "profiling.h"
extern void profilerNullA(const char*);
extern nvtxRangeId_t profilerStartNull(const char*);
extern void profilerEndNull(nvtxRangeId_t);
//end if profiling

/*************************************************************
General definitions
*************************************************************/

#include <stdarg.h>
void dbg_printf(const char *fmt, ...)
{
	va_list args;
	va_start(args, fmt);
	vfprintf(stderr, fmt, args);
	va_end(args);
}

/* Error messages display */
#define BMD_ERR printf
#define USE_PASSTHRU 0
#define DEBUG 0
#define READ_DMA_STATUS BMD_ReadReg32(hDev, BMD_DMACST_OFFSET)
#define TRACE(x) do { if (DEBUG) dbg_printf x; } while (0)


/* -----------------------------------------------
Device find, open and close
----------------------------------------------- */
static WDC_DEVICE_HANDLE DeviceFindAndOpen(DWORD dwVendorId, DWORD dwDeviceId);
static BOOL DeviceFind(DWORD dwVendorId, DWORD dwDeviceId, WD_PCI_SLOT *pSlot);
static WDC_DEVICE_HANDLE DeviceOpen(const WD_PCI_SLOT *pSlot);
static void DeviceClose(WDC_DEVICE_HANDLE hDev, PDIAG_DMA pDma);
static void DIAG_DMAClose(WDC_DEVICE_HANDLE hDev, PDIAG_DMA pDma);
static void DiagDmaIntHandler(WDC_DEVICE_HANDLE hDev,
	BMD_INT_RESULT *pIntResult);

static void DIAG_DMAClose(WDC_DEVICE_HANDLE hDev, PDIAG_DMA pDma)
{
	DWORD dwStatus;
	printf("in diagdmaclose\n");
	if (!pDma)
		return;

	if (BMD_IntIsEnabled(hDev))
	{
		dwStatus = BMD_IntDisable(hDev);
		printf("DMA interrupts disable%s\n",
			(WD_STATUS_SUCCESS == dwStatus) ? "d" : " failed");
	}

	if (pDma->hDma)
	{
		BMD_DmaClose(pDma->hDma);
		printf("DMA closed (handle 0x%p)\n", pDma->hDma);
	}
	printf("bzero diagdmaclose\n");
	BZERO(*pDma);
	printf("leaving diagdmaclose\n");
}


/* Find and open a BMD device */
static WDC_DEVICE_HANDLE DeviceFindAndOpen(DWORD dwVendorId, DWORD dwDeviceId)
{
	WD_PCI_SLOT slot;

	if (!DeviceFind(dwVendorId, dwDeviceId, &slot))
		return NULL;

	return DeviceOpen(&slot);
}

/* Find a BMD device */
static BOOL DeviceFind(DWORD dwVendorId, DWORD dwDeviceId, WD_PCI_SLOT *pSlot)
{
	DWORD dwStatus;
	DWORD i, dwNumDevices;
	WDC_PCI_SCAN_RESULT scanResult;

	BZERO(scanResult);
	dwStatus = WDC_PciScanDevices(dwVendorId, dwDeviceId, &scanResult);
	if (WD_STATUS_SUCCESS != dwStatus)
	{
		BMD_ERR("DeviceFind: Failed scanning the PCI bus.\n"
			"Error: 0x%lx - %s\n", dwStatus, Stat2Str(dwStatus));
		return FALSE;
	}

	dwNumDevices = scanResult.dwNumDevices;
	if (!dwNumDevices)
	{
		BMD_ERR("No matching device was found for search criteria "
			"(Vendor ID 0x%lX, Device ID 0x%lX)\n", dwVendorId, dwDeviceId);
		return FALSE;
	}

	printf("\nFound %ld matching device(s) "
		"[Vendor ID 0x%lX%s, Device ID 0x%lX%s]:\n",
		dwNumDevices, dwVendorId, dwVendorId ? "" : " (ALL)",
		dwDeviceId, dwDeviceId ? "" : " (ALL)");

	for (i = 0; i < dwNumDevices; i++)
	{
		printf("\n%2ld. Vendor ID: 0x%lX, Device ID: 0x%lX\n", i + 1,
			scanResult.deviceId[i].dwVendorId,
			scanResult.deviceId[i].dwDeviceId);

		WDC_DIAG_PciDeviceInfoPrint(&scanResult.deviceSlot[i], FALSE);
	}
	printf("\n");

	*pSlot = scanResult.deviceSlot[i - 1];

	return TRUE;
}

/* Open a handle to a BMD device */
static WDC_DEVICE_HANDLE DeviceOpen(const WD_PCI_SLOT *pSlot)
{
	WDC_DEVICE_HANDLE hDev;
	DWORD dwStatus;
	WD_PCI_CARD_INFO deviceInfo;

	/* Retrieve the device's resources information */
	BZERO(deviceInfo);
	deviceInfo.pciSlot = *pSlot;
	dwStatus = WDC_PciGetDeviceInfo(&deviceInfo);
	if (WD_STATUS_SUCCESS != dwStatus)
	{
		BMD_ERR("DeviceOpen: Failed retrieving the device's resources "
			"information.\nError 0x%lx - %s\n", dwStatus, Stat2Str(dwStatus));
		return NULL;
	}

	/* NOTE: You can modify the device's resources information here, if
	necessary (mainly the deviceInfo.Card.Items array or the items number -
	deviceInfo.Card.dwItems) in order to register only some of the resources
	or register only a portion of a specific address space, for example. */

	/* Open a handle to the device */
	hDev = BMD_DeviceOpen(&deviceInfo, NULL);
	if (!hDev)
	{
		BMD_ERR("DeviceOpen: Failed opening a handle to the device: %s",
			BMD_GetLastErr());
		return NULL;
	}

	return hDev;
}

/* Close handle to a BMD device */
static void DeviceClose(WDC_DEVICE_HANDLE hDev, PDIAG_DMA pDma)
{
	printf("in deviceclose\n");
	if (!hDev)
		return;

	if (pDma)
		DIAG_DMAClose(hDev, pDma);

	printf("skipping closing device cause it always crashes. bad hack.\n");
	/*	if (!BMD_DeviceClose(hDev))
	{
	BMD_ERR("DeviceClose: Failed closing BMD device: %s",
	BMD_GetLastErr());
	}*/
	printf("leaving deviceclose\n");
}

/* Diagnostics interrupt handler routine */
static void DiagDmaIntHandler(WDC_DEVICE_HANDLE hDev,
	BMD_INT_RESULT *pIntResult)
{
	printf("\n###\nDMA %s based interrupt, recieved #%ld\n",
		pIntResult->fIsMessageBased ? "message" : "line",
		pIntResult->dwCounter);
	if (pIntResult->fIsMessageBased)
		printf("Message data 0x%lx\n", pIntResult->dwLastMessage);
	printf("###\n\n");
	printf("Not verifying transfer\n");
	//DmaTransferVerify(hDev, pIntResult->pBuf, pIntResult->dwTotalCount,
	//pIntResult->u32Pattern, pIntResult->fIsRead);
}

//C core of fpgaInterface class initialisation function
DWORD fpgaInterfaceInit_c(WDC_DEVICE_HANDLE hDev, DIAG_DMA* dma, WDC_DEVICE_HANDLE* tmp_hDev){
	//BOOL status = 1;
	DWORD dwStatus;
	//hDev = NULL; //initialise


	/* Initialize the BMD library */
	dwStatus = BMD_LibInit();
	if (WD_STATUS_SUCCESS != dwStatus){
		BMD_ERR("bmd_diag: Failed to initialize the BMD library: %s",
			BMD_GetLastErr());
		return dwStatus;
	}
	/* Find and open a BMD device (by default ID) */
	hDev = DeviceFindAndOpen(FPGA_VENDOR_ID, FPGA_DEVICE_ID);
	if (hDev == NULL)
		return WD_WINDRIVER_STATUS_ERROR; //fake a windriver error

	BZERO(dma);
	/*Pointer confusion wrt tmp_hDev: return the tmp pointer and we re-set it in the constructor */
	*tmp_hDev = hDev;

	return dwStatus;
}

DWORD fpgaInterfaceDestructor_c(WDC_DEVICE_HANDLE* hDev, PDIAG_DMA pDma){
	DWORD dwStatus;
	/* Perform necessary cleanup before exiting the program */
	printf("in fpgaInterfaceDestructor_c\n");
	if (hDev)
		DeviceClose(hDev, NULL);

	dwStatus = BMD_LibUninit();
	if (WD_STATUS_SUCCESS != dwStatus)
	{
		BMD_ERR("bmd_diag: Failed to uninit the BMD library: %s",
			BMD_GetLastErr());
	}
	printf("leaving fpgaInterfaceDestructor_c\n");
	return dwStatus;
}

//see DMA status register document for what each of these bits do
void fpgaPrintStatus(UINT32 dmacst, BOOL pad) {
	if (pad)
		TRACE(("\t\t"));
	TRACE(("Status %08X | Egress%s%s%s%s%s | Ingress%s%s%s%s%s | Stalls %d | %s |Frames %d  R/W %X\n ", \
		dmacst, \
		(dmacst >> 14 & 0x01 ? " full" : "     "), (dmacst >> 31 & 0x01 ? " almostfull" : "           "), \
		(dmacst >> 29 & 0x01 ? " almostempty" : "            "), (dmacst >> 30 & 0x01 ? " empty" : "      "), \
		(dmacst >> 24 & 0x01 ? " overflowed" : "           "), \
		(dmacst >> 27 & 0x01 ? " almostfull" : "           "), (dmacst >> 28 & 0x01 ? " full" : "     "), \
		(dmacst >> 26 & 0x01 ? " almostempty" : "            "), (dmacst >> 25 & 0x01 ? " empty" : "      "), \
		(dmacst >> 23 & 0x01 ? " overflowed" : "           "), \
		dmacst >> 18 & 0x1F, (dmacst >> 15 & 0x01 ? "TIMEOUT" : "       "), \
		dmacst >> 16 & 0x03, dmacst & 0x0F));

}
//appOptions:0: ped,get hists
// 1: ped,get scores
// 2: car, get hists
//3: car, get scores

int fpgaProcessImage(WDC_DEVICE_HANDLE hDev, PDIAG_DMA pDma, CvMat* img, int operationType,
	UINT32 IngressLowerAddr, UINT32 IngressUpperAddr, UINT32 EgressLowerAddr, UINT32 EgressUpperAddr,
	int egressDataSz, UINT32 uIngressSizeB, UINT32 uEgressSizeB)
{
	//This does the actual procesing; however, we want to remove as much of the buffer init
	//stuff as possible & move it to the constructor
	//requested DMA transfer size must be a multiple of this FPGA_DMA_SIZE_INCREMENT
	//FPGA_HOG_NBINS must always be 9
	//egressDataSz is how much valid data will be transferred (ie not how much data is moved but how much of that is valid) - used for
	//dumps of floats at this point
	BOOL fIsDmaOpen;
	BOOL fPolling = TRUE;	//we're not using FPGA interrupts
	DWORD dwOptions;
	BOOL dumpHists = FALSE;
	BOOL dumpHistsAsBin = FALSE;

	//decode operationType
	BOOL isCar = (operationType & FPGA_USE_CAR_HOG) ? TRUE : FALSE;
	BOOL getScores = (operationType & FPGA_HOG_GET_SCORES) ? TRUE : FALSE;

	nvtxRangeId_t id1;

	int iImgIterA, iImgIterB;

	BOOL isDone;
	int watchdog;

	UINT32 dmacst;

	//Profiling
#if DEBUG
	DWORD write_start_counter, write_stop_counter;
	DWORD read_start_counter, read_stop_counter;
	double read_perf, write_perf;
#endif

	unsigned char* cp;
	float* fp, *fptop;
	UINT32* up;

	void *memcpy_dst; //pointers for memcpy
	unsigned char *memcpy_src;
	size_t memcpy_sz;

#if USE_PASSTHRU
	CvMat passthru_img;
#endif
	fIsDmaOpen = pDma->hDma ? TRUE : FALSE;

	if (!fIsDmaOpen){
		printf("Error: DMA handle to FPGA should be initialised by this point\n");
		return -1;
	}
	if (!IsValidDmaHandle(/*(BMD_DMA_HANDLE)*/(pDma->hDma), "BMD_DMADevicePrepare_unroll"))
		return -1;

	/*The read direction denotes ingress TO the DDR2 memory.
	The write direction denotes egress FROM the Endpoint DDR2 memory. */
	dwOptions = DMA_TO_DEVICE | DMA_FROM_DEVICE; //need to transfer both ways
	/* The BMD reference design does not support s/g DMA, so we use contiguous	*/
	dwOptions |= DMA_KERNEL_BUFFER_ALLOC;

	//copy image into buffer - assumes img->step == img->rows - must do writes through userspace?

	memcpy_dst = (pDma->pBuf);
	memcpy_src = (img->data.ptr);
#ifdef LINUX
	memcpy_sz=MIN(img->cols*img->rows, (int)uIngressSizeB);
#else
	memcpy_sz = min(img->cols*img->rows, (int)uIngressSizeB);
#endif

	memcpy(memcpy_dst, memcpy_src, memcpy_sz);

	TRACE(("Buffer from \t0x%016llX to 0x%016llX phys filled with source image\n",
		pDma->hDma->pDma->Page->pPhysicalAddr, (long long int)pDma->hDma->pDma->Page->pPhysicalAddr
		+ (int)(memcpy_sz)));
	//zero rest of buffer
	//lazy, reuse pointer
	iImgIterA = (int)EgressLowerAddr - IngressLowerAddr; //get difference between ingress and egress - not necessarily
	//the same as image size for scaled/smaller images
	//however, assumes contiguous buffer
	iImgIterB = (int)(uIngressSizeB);
	fptop = (float*)((char*)(pDma->pBuf) + iImgIterA + uEgressSizeB);
	if ((int)fptop > (int)((char*)(pDma->pBuf) + pDma->hDma->pDma->dwBytes))
		printf("warning - possibility of buffer overflow\n");
	fp = (float*)((char*)(pDma->pBuf) + iImgIterA);
	memset(fp, 0, uEgressSizeB);//set output to zero - may not actually need to do this

	TRACE(("Buffer from \t0x%016llX to 0x%016llX filled with 0.0f\n",
		(long long int)pDma->hDma->pDma->Page->pPhysicalAddr + (int)iImgIterA,
		(long long int)pDma->hDma->pDma->Page->pPhysicalAddr + (int)(iImgIterA + uEgressSizeB)));

	/* Assert Initiator Reset - don't need to deassert*/
	BMD_WriteReg32(hDev, BMD_DMACST_OFFSET, 0x2A); //reset, ack read and write complete
	/*Must now have >=2*FPGA_COLS delay between reset deassert and image data
	hitting hog logic inside fpga. this shouldnt be a problem*/
	//WDC_Sleep(200,0);

	TRACE(("DMA starting with ")); fpgaPrintStatus(READ_DMA_STATUS, FALSE);
	isDone = FALSE;
	//Read performance counters at start

#if DEBUG
	write_start_counter = BMD_ReadReg32(hDev, BMD_DMAWRP_OFFSET);
	read_start_counter = BMD_ReadReg32(hDev, BMD_DMARDP_OFFSET);
#endif

	/* Prepare for full-duplex transfer*/
	if (dwOptions & DMA_TO_DEVICE) {//is read
		/* Ingress to FPGA: READ from Main Memory */
		/* Set upper (32) 8bits  of DMA address (should be zero for 32bits anyway)*/
		BMD_WriteReg32(hDev, BMD_DMARAS_U_OFFSET, IngressUpperAddr);
		/* Set lower 32bit of DMA address */
		BMD_WriteReg32(hDev, BMD_DMARAS_L_OFFSET, IngressLowerAddr);
		/* Set destination address */
		BMD_WriteReg32(hDev, BMD_DMARAD_OFFSET, 0);
		/* Set read transfer (number of bytes) */
		BMD_WriteReg32(hDev, BMD_DMARXS_OFFSET, (DWORD)uIngressSizeB);
		TRACE(("FPGA programmed for ingress of %08X bytes FROM %08X %08X TO %08X\n",
			uIngressSizeB, IngressUpperAddr, IngressLowerAddr, 0));
		TRACE(("FPGA registers are  DMARAS_U %08X, DMARAS_L %08X, DMARAD %08X, DMARXS %08X\n",
			BMD_ReadReg32(hDev, BMD_DMARAS_U_OFFSET), BMD_ReadReg32(hDev, BMD_DMARAS_L_OFFSET),
			BMD_ReadReg32(hDev, BMD_DMARAD_OFFSET), BMD_ReadReg32(hDev, BMD_DMARXS_OFFSET)
			));
		if (!(dwOptions & DMA_FROM_DEVICE)) { //if ingress only
			BMD_WriteReg32(hDev, BMD_DMACST_OFFSET, 0x0004);
			TRACE(("Ingress/READ transfer only started\n"));
		}
	}

	if (dwOptions & DMA_FROM_DEVICE) {
		/* Egress from FPGA: WRITE to Main Memory */
		/* Set upper 32bits of DMA address  - should be zero*/
		BMD_WriteReg32(hDev, BMD_DMAWAD_U_OFFSET, EgressUpperAddr);
		/* Set lower 32bit of DMA address */
		BMD_WriteReg32(hDev, BMD_DMAWAD_L_OFFSET, EgressLowerAddr);
		/* Set source address - 0 */
		BMD_WriteReg32(hDev, BMD_DMAWAS_OFFSET, 0);
		/* Set write transfer #bytes */
		BMD_WriteReg32(hDev, BMD_DMAWXS_OFFSET, (DWORD)uEgressSizeB);
		TRACE(("FPGA programmed for egress  of %08X bytes TO %08X %08X FROM %08X\n",
			uEgressSizeB, EgressUpperAddr, EgressLowerAddr, 0));
		TRACE(("FPGA registers are  DMAWAD_U %08X, DMAWAD_L %08X, DMAWAS %08X, DMAWXS %08X\n",
			BMD_ReadReg32(hDev, BMD_DMAWAD_U_OFFSET), BMD_ReadReg32(hDev, BMD_DMAWAD_L_OFFSET),
			BMD_ReadReg32(hDev, BMD_DMAWAS_OFFSET), BMD_ReadReg32(hDev, BMD_DMAWXS_OFFSET)
			));
		if (!(dwOptions & DMA_TO_DEVICE)) { //if egress only
			BMD_WriteReg32(hDev, BMD_DMACST_OFFSET, 0x0001);
			TRACE(("Egress/WRITE transfer only started\n"));
		}
	}
	BMD_DmaSyncCpu(pDma->hDma);
	TRACE(("Before full-duplex /after simplex, ")); fpgaPrintStatus(READ_DMA_STATUS, FALSE);
	id1 = profilerRangeStartA("fpga:processing");
	if ((dwOptions & DMA_FROM_DEVICE) && (dwOptions & DMA_TO_DEVICE)){
		UINT32 cmd2send = 0x0005; //read & write command
		if (getScores)
			cmd2send |= 0x0040;
		if (isCar)
			cmd2send |= 0x0080;
		//this does for both using 0x0085 for car cells, 0x00C5 for car scores, and the two codes above for peds.
		TRACE(("Full-duplex transfer started - getting scores? %d cars? %d\n", getScores, isCar));
		BMD_WriteReg32(hDev, BMD_DMACST_OFFSET, cmd2send);

	}

	watchdog = 0;
	while (!isDone) {
		dmacst = READ_DMA_STATUS;
		fpgaPrintStatus(dmacst, TRUE);
		if ((dwOptions & DMA_FROM_DEVICE) && (dwOptions & DMA_TO_DEVICE)){ //full duplex
			if (dmacst & 0x000A){ //read and write done
				isDone = TRUE;
				break;
			}
		}
		else if ((dwOptions & DMA_FROM_DEVICE)){ //egress
			if (dmacst & 0x0002) //write done
				isDone = TRUE;
		}
		else if ((dwOptions & DMA_TO_DEVICE)){ //egress
			if (dmacst & 0x0008) //write done
				isDone = TRUE;
		}
		WDC_Sleep(200, 0);
		if (watchdog > 500) break;
		watchdog++;
	}
	profilerRangeEnd(id1);
	profilerMarkA("fpga:finished transfer");
	TRACE(("TODO: add a 'frame complete before stall' signal in HDL\n"));
	TRACE(("finished DMA\n"));
	fpgaPrintStatus(READ_DMA_STATUS, TRUE);

	//Performance info
#if DEBUG
	write_stop_counter = BMD_ReadReg32(hDev, BMD_DMAWRP_OFFSET);
	read_stop_counter = BMD_ReadReg32(hDev, BMD_DMARDP_OFFSET);
	read_perf = ((uIngressSizeB*1.0f) / (1.0f*pow(2.f, 20))) / (1.0*(read_stop_counter - read_start_counter) / (FPGA_DATA_CLOCK));
	write_perf = ((uEgressSizeB*1.0f)/(1.0f*pow(2.f,20))) / (1.0*(write_stop_counter - write_start_counter)/(FPGA_DATA_CLOCK)) ;
	TRACE(("WR cycles %d, RD cycles %d, Write speed %f MB/s, read speed %f MB/s based on %dMhz trn_clk. %fms write and %fms read, %fms total\n",
		write_stop_counter - write_start_counter, read_stop_counter - read_start_counter, write_perf, read_perf,(int)(FPGA_DATA_CLOCK/1000000),
		1000.0*(write_stop_counter - write_start_counter)/(FPGA_DATA_CLOCK), 1000.0*(read_stop_counter - read_start_counter)/(FPGA_DATA_CLOCK),
		1000.0*(write_stop_counter - read_start_counter)/(FPGA_DATA_CLOCK)
		));
#endif
	BMD_WriteReg32(hDev, BMD_DMACST_OFFSET, 0x000A); //clear done flags

	//zero-copy: point cellHistsfp to base of buffer
	up = (UINT32*)((UINT32)pDma->pBuf + uIngressSizeB); //get a pointer to a 32bit int @base of egress buffer
	cp = (unsigned char*)(pDma->pBuf) + uIngressSizeB; //used for passthru
	//fp = (float*) ((char*) (pDma->pBuf) + uIngressSizeB);
	fp = (float*)((char*)(pDma->pBuf) + iImgIterA);
	TRACE(("changed fp to go to start of egress instead of moving by ingress size. check this doesnt break hog\n"));
	//no copy here, do inside class

	if (dumpHists){
		FILE *fh;
		static int counter = 0;
		char filename[40];
		char tmpnum[5];
		UINT32 iCells;
		sprintf(tmpnum, "%04d", counter);
		strcpy(filename, "C:\\temp\\histsdump_");
		strcat(filename, tmpnum);
		strcat(filename, ".txt");
		if ((fh = fopen(filename, "w")) == NULL)
			printf("Failed opening %s file\n", filename);
		else {
			for (iCells = 0; iCells < /*egressDataSz*/uEgressSizeB / 4; iCells++)
				fprintf(fh, "%f\n", fp[iCells]);
			fclose(fh);
		}
		counter++;
	}
	if (dumpHistsAsBin){
		FILE *fh;
		char filename[] = "histsdump.bin";
		int count;
		if ((fh = fopen(filename, "wb")) == NULL)
			printf("Failed opening histsdump file %s\n", filename);
		else {
			count = fwrite(fp, 4, egressDataSz / 4, fh);
			printf("Wrote %d hists to %s\n", count, filename);
			fclose(fh);
		}
	}
	//Return a status: 0 if OK, else return dmacst
#if USE_PASSTHRU
	passthru_img = cvMat(img->rows, img->cols, CV_8UC1, cp);
	cvShowImage("passthru img", &passthru_img);
	cvWaitKey(5);
	{
		static int counter=0;
		FILE *dfh;
		char filename[40];
		char tmpnum[5];
		sprintf(tmpnum,"%04d",counter);
		strcpy(filename,"C:\\temp\\passthrudump");
		strcat(filename,tmpnum);
		strcat(filename,".txt");
		if (!(dfh= fopen(filename,"w")))
			printf("Failed opening %s file\n",filename);
		else {
			int i;
			for (i = 0; i < (int) uEgressSizeB; i ++)
				fprintf(dfh, "%d\n",cp[i]);
			fclose(dfh);
		}
		counter++;
	}
#endif
	dmacst = READ_DMA_STATUS;
	if ((dmacst & 0x000F) == 0x0000)
		return 0x0000;
	else
		return dmacst;
}

//first step in communcations with the FPGA is to find out its payload sizes in both directions
//this also has the benefit of us doing a test read to the chip before allocating buffers
int fpgaGetPayloadSizes(WDC_DEVICE_HANDLE hDev, PDIAG_DMA pDma){
	BOOL fIsDmaOpen;
	DWORD dwOptions;
	BOOL fIsRead;
	int rrqGroupSize;

	//Get Max Payload size & read request size in both bytes and dwords from device
	WORD wMaxPayloadSizeDW, wReadRequestSizeDW, wMaxPayloadSizeB, wReadRequestSizeB;

	fIsDmaOpen = pDma->hDma ? TRUE : FALSE;

	if (fIsDmaOpen)
		DIAG_DMAClose(hDev, pDma);

	//Now (re-)open DMA
	//Set options here as opposed to asking the user
	dwOptions = DMA_TO_DEVICE; //need to transfer both ways
	/*The read direction denotes ingress TO the FPGA FROM RAM.
	The write direction denotes egress FROM the FPGA TO RAM. */
	fIsRead = dwOptions & DMA_FROM_DEVICE ? FALSE : TRUE;
	dwOptions |= DMA_FROM_DEVICE;

	/* The BMD reference design does not support s/g DMA, so we use contiguous	*/
	dwOptions |= DMA_KERNEL_BUFFER_ALLOC;

	//Initialize FPGA
	/*reset DMA logic (doesn't affect certain parts of the app)
	//	and set ack read and write complete */
	//BMD_WriteReg32(hDev, BMD_DMACST_OFFSET, 0x2A); 
	//WDC_Sleep(200,0);

	/* Get the max payload size and read request sizes in bytes & dwords from the device */
	//max RRQ (should be 512 bytes = 16 dwords)
	//max payload( should be 256 bytes = 8 dwords) - for this Dell PC
	//However, the FPGA does not currently handle a MPS of 256 bytes properly
	//and was regenerated with MPS=128Bytes
	wReadRequestSizeB = BMD_DmaGetMaxPacketSize(hDev, fIsRead);
	wReadRequestSizeDW = wReadRequestSizeB / sizeof(UINT32);
	wMaxPayloadSizeB = BMD_DmaGetMaxPacketSize(hDev, !fIsRead);
	wMaxPayloadSizeDW = wMaxPayloadSizeB / sizeof(UINT32);
	printf("Max payload size (for egress) is %d (0x%X) bytes / %d dwords. \nRead request size (for ingress) is %d (0x%X) bytes / %d dwords\n",
		wMaxPayloadSizeB, wMaxPayloadSizeB, wMaxPayloadSizeDW, wReadRequestSizeB, wReadRequestSizeB, wReadRequestSizeDW);
	rrqGroupSize = wReadRequestSizeB * FPGA_LAUNCHED_RRQS;
	return rrqGroupSize;
}


//Open a single buffer for image ingress and egress. This function will open the buffer and return relevant 
//addresses for each section
int fpgaOpenDMABuffer(WDC_DEVICE_HANDLE hDev, PDIAG_DMA pDma, UINT32 uIngressSizeB, UINT32 uEgressTxSizeB,
	UINT32* IngressLowerAddr, UINT32* IngressUpperAddr, UINT32* EgressLowerAddr, UINT32* EgressUpperAddr)
{
	DWORD dwTotalCountB, dwOptions, dwStatus;
	BOOL fIsRead, fPolling;
	dwTotalCountB = (DWORD)uIngressSizeB + (DWORD)uEgressTxSizeB;
	dwOptions = DMA_TO_DEVICE; //need to transfer both ways
	/*The read direction denotes ingress TO the FPGA FROM RAM.
	The write direction denotes egress FROM the FPGA TO RAM. */
	fIsRead = dwOptions & DMA_FROM_DEVICE ? FALSE : TRUE;
	//TODO PCIe interrupts
	fPolling = TRUE;
	//requested DMA transfer size must be a multiple of FPGA_DMA_SIZE_INCREMENT
	//FPGA_HOG_NBINS must always be 9
	//TODO pass out base of buffer and construct an image matrix around it.
	/* Open DMA handle */
	dwStatus = BMD_DmaOpen(hDev, &pDma->pBuf, dwOptions, dwTotalCountB, &pDma->hDma);

	if (WD_STATUS_SUCCESS != dwStatus)
	{
		printf("\nFailed to open DMA handle. Error 0x%lx - %s\n", dwStatus,
			Stat2Str(dwStatus));
		return dwStatus;
	}
	printf("\nDMA handle was opened successfully (handle 0x%lx)\n", pDma->hDma);

	if (!fPolling) /* Enable DMA interrupts (if not polling) */
	{
		BMD_DmaIntEnable(hDev, fIsRead);
		if (!BMD_IntIsEnabled(hDev))
		{
			dwStatus = BMD_IntEnable(hDev, DiagDmaIntHandler);

			if (WD_STATUS_SUCCESS != dwStatus)
			{
				printf("\nFailed enabling DMA interrupts. Error 0x%lx - %s\n",
					dwStatus, Stat2Str(dwStatus));
				goto Error;
			}
			printf("\nDMA interrupts enabled\n");
		}
	}
	else /* Disable interrupts (polling) */
		BMD_DmaIntDisable(hDev, fIsRead);
	printf("Note: interrupts not implemented\n");

	//How to split up the buffer?
	*IngressLowerAddr = (UINT32)pDma->hDma->pDma->Page[0].pPhysicalAddr;
	*IngressUpperAddr = (UINT32)((pDma->hDma->pDma->Page[0].pPhysicalAddr >> 32) & 0xFF);
	*EgressLowerAddr = *IngressLowerAddr + (DWORD)uIngressSizeB; //offset by frame size
	*EgressUpperAddr = *IngressUpperAddr;

	printf("Buffer info: starts at 0x%08X in userspace with length 0x%08X/%d Bytes\nso from \t0x%016llX to 0x%016llX physical\n",
		pDma->hDma->pDma->pUserAddr, pDma->hDma->pDma->dwBytes, pDma->hDma->pDma->dwBytes,
		pDma->hDma->pDma->Page->pPhysicalAddr,
		(long long int) pDma->hDma->pDma->Page->pPhysicalAddr + (int)(pDma->hDma->pDma->Page->dwBytes));

	fpgaPrintStatus(READ_DMA_STATUS, TRUE);
	return dwStatus;

Error:
	DIAG_DMAClose(hDev, pDma);
	return dwStatus;
}

#endif
