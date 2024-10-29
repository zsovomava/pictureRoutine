
#ifndef PICTURE_ROUTINE_H
#define PICTURE_ROUTINE_H

extern "C" __declspec(dllexport) void RunNegateKernel(unsigned char* pictureIn, unsigned char* pictureOut, int width, int height, int widthStep, int channels);
extern "C" __declspec(dllexport) void RungammaKernel(unsigned char* pictureIn, unsigned char* pictureOut, int width, int height, int widthStep, int channels, float gamma);
extern "C" __declspec(dllexport) void RunLogKernel(unsigned char* pictureIn, unsigned char* pictureOut, int width, int height, int widthStep, int channels, float c);
extern "C" __declspec(dllexport) void RunGrayKernel(unsigned char* pictureIn, unsigned char* pictureOut, int width, int height, int widthStep, int channels);
extern "C" __declspec(dllexport) void RunHistogramKernel(unsigned char* pictureIn, int* histogramOut, int width, int height, int widthStep, int channels);
extern "C" __declspec(dllexport) void RunHistogramEqualizationKernel(unsigned char* pictureIn, unsigned char* pictureOut, int width, int height, int widthStep, int channels);
extern "C" __declspec(dllexport) void RunAVGKernel(unsigned char* pictureIn, unsigned char* pictureOut, int width, int height, int widthStep, int channels, int matrixDims);
extern "C" __declspec(dllexport) void RunGaussKernel(unsigned char* pictureIn, unsigned char* pictureOut, int width, int height, int widthStep, int channels, int matrixDims, double sigma);
extern "C" __declspec(dllexport) void RunSobelKernel(unsigned char* pictureIn, unsigned char* pictureOut, int width, int height, int widthStep, int channels);
extern "C" __declspec(dllexport) void RunLaplaceKernel(unsigned char* pictureIn, unsigned char* pictureOut, int width, int height, int widthStep, int channels);
extern "C" __declspec(dllexport) void RunImportantPointKernel(unsigned char* pictureIn, unsigned char* pictureOut, int width, int height, int widthStep, int channels);

#endif