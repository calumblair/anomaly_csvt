readme.md

This contains the  code for the paper "Video Anomaly Detection in Real-Time on a Power-Aware Heterogeneous Platform" By C G Blair  and N.M Robertson. See paper for full explanation.

This contains host code.

Build dependencies:
CUDA (V6 and V6.5 tested) 
Qt (V4.8.6 and v4.6.2 tested)
OpenCV (2.4.10.1 and 2.4.4 tested)
Boost (1.52 tested)
TBB (optional)
For FPGA support (optional), Jungo Windriver v10.10  or 11.90 are needed and patch files are included.
FPGA support was tested on a 32 bit WinXP PC with a ML605 board and Xilinx ISE 13.3/13.4 and has not been tested on anything else, although it should work.

CPU/GPU support tested with VS2010 and 2013 on 32 and 64 bit systems, and gcc 4.4.7 on linux x64.

Build steps:
obtain the correct OpenCV source, then apply the patch files given.
Build the modified OpenCV with QT, Boost and  CUDA support. Building with TBB will speed up runtime.
Open the  MSVC project or Eclipse makefile for this code. If using Windows, ensure all paths in the property file are set; this is the easiest approach to allow building.
Build and link the project. 
If FPGA support is desired:
* obtain and install a recent Windriver version, and apply the supplied patch files.
* flash the supplied bitstream onto the FPGA
* add USE_WINDRIVER_FPGA_INTERFACE as a compile-time definition.
* pass the --disable_fpga false argument at runtime

Testing/evaluation:
run the generated binary with the arguments:
XXX
Release mode is preferrred : we skip some of the assertions still present.
