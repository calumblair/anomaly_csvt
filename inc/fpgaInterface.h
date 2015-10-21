/*FPGA Characteristics - change these if FPGA HDL interface or HOG application changes significantly*/
/* CB 2/4/12 */
#define FPGA_VENDOR_ID 0x10EE
#define FPGA_DEVICE_ID 0x0000

#define FPGA_INGRESS_BUFSZ_BYTES 8192
#define FPGA_EGRESS_BUFSZ_BYTES 8192
#define FPGA_EGRESS_AE_THRESH 4096
#define FPGA_LAUNCHED_RRQS 8
#define FPGA_DMA_SIZE_INCREMENT 128
//#define FPGA_DATA_CLOCK 200000000.0
#define FPGA_DATA_CLOCK 160000000.0
#define FPGA_PIPELINE_DEPTH_GROUPS 20

#include "fpgaImageSize.h"

#ifndef FPGA_INTERFACE_H
#define FPGA_INTERFACE_H

/* DMA Struct */
typedef struct {
	BMD_DMA_HANDLE hDma;
	PVOID pBuf;
} DIAG_DMA, *PDIAG_DMA;

/* FPGA interface: Constructor/Destructor call these core C functions*/
#ifdef __cplusplus
extern "C" {
#endif
	DWORD fpgaInterfaceInit_c(WDC_DEVICE_HANDLE hDev, DIAG_DMA* dma, WDC_DEVICE_HANDLE* tmp_hDev);
	DWORD fpgaInterfaceDestructor_c(WDC_DEVICE_HANDLE* hDev, DIAG_DMA* dma);
	int fpgaProcessImage(WDC_DEVICE_HANDLE hDev, PDIAG_DMA pDma, CvMat* img, int operationType,
		UINT32 IngressLowerAddr, UINT32 IngressUpperAddr, UINT32 EgressLowerAddr, UINT32 EgressUpperAddr, int egressDataSz,
		UINT32 uIngressSizeB, UINT32 uEgressSizeB);
	int fpgaGetPayloadSizes(WDC_DEVICE_HANDLE hDev, PDIAG_DMA pDma);
	int fpgaOpenDMABuffer(WDC_DEVICE_HANDLE hDev, PDIAG_DMA pDma, UINT32 uIngressSizeB, UINT32 uEgressTxSizeB,
		UINT32* IngressLowerAddr, UINT32* IngressUpperAddr, UINT32* EgressLowerAddr, UINT32* EgressUpperAddr);
#ifdef __cplusplus
}
#endif
#endif
