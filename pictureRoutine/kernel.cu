
#include "cuda_runtime.h"
#include <cuda.h>
#include "device_launch_parameters.h"
#include "Header.cuh"
#include <cMath>
#include <stdio.h>
#include <iostream>
#include <string>
#include <fstream>

__global__ void negateKernel(unsigned char* dev_in, unsigned char* dev_out, int width, int height, int widthStep, int channels)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int pixel_index = (width * y * channels) + y*widthStep + x * channels;
        for (int c = 0; c < channels; c++) {
            dev_out[pixel_index + c] = 255 - dev_in[pixel_index + c]; // Negálás
        }
    }
}
__global__ void changeLookUpKernel(unsigned char* dev_in, unsigned char* dev_out, int width, int height, int widthStep, int channels, unsigned char* dev_lookUp) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int pixel_index = (width * y * channels) + y * widthStep + x * channels;
        for (int c = 0; c < channels; c++) {
            dev_out[pixel_index + c] = dev_lookUp[dev_in[pixel_index + c]];
        }
    }

}
__global__ void grayKernel(unsigned char* dev_in, unsigned char* dev_out, int width, int height, int widthStep, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int pixel_index = (width * y * channels) + y * widthStep + x * channels;
        int newPixel_index = width * y + y * widthStep + x;
        int tmp = (dev_in[pixel_index]*299 + dev_in[pixel_index + 1] * 587 + dev_in[pixel_index] * 114)/1000;
        dev_out[newPixel_index] = tmp;
    }

}
__global__ void histogramKernel(unsigned char* dev_grayIn, int* dev_Out, int width, int height, int widthStep, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        int pixel_index = width * y + y * widthStep + x;
        atomicAdd(&dev_Out[dev_grayIn[pixel_index]], 1);
    }
}

__global__ void MaskKernel(unsigned char* dev_in, unsigned char* dev_out, int width, int height, int widthStep, int channels, int* matrix, int matrix_x, int matrix_y) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        long divisor = 0;
        long sum = 0;

        int half_matrix_x = matrix_x / 2;
        int half_matrix_y = matrix_y / 2;

        for (int i = -half_matrix_x; i <= half_matrix_x; i++) {
            int x_in_picture = x + i;

            for (int j = -half_matrix_y; j <= half_matrix_y; j++) {
                int y_in_picture = y + j;

                // Ensure the coordinates are within the image bounds
                if (x_in_picture >= 0 && x_in_picture < width && y_in_picture >= 0 && y_in_picture < height) {
                    // Get the matrix value and calculate the linear input index
                    int matrix_value = matrix[(i + half_matrix_x) * matrix_y + (j + half_matrix_y)];
                    divisor += matrix_value;

                    int pixel_index = (width * y_in_picture) + y_in_picture * widthStep + x_in_picture;
                    sum += dev_in[pixel_index] * matrix_value;
                }
            }
        }

        // Calculate the output pixel value
        int output_pixel_index = width * y + y * widthStep + x;
        divisor = divisor != 0 ? divisor : 1;
        dev_out[output_pixel_index] = abs(sum / divisor);
    }
}
__global__ void SobelKernel(unsigned char* dev_gray_x, unsigned char* dev_gray_y, unsigned char* dev_out, int width, int height, int widthStep) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        int idx = y * width + y * widthStep + x;
        dev_out[idx] = (sqrtf(dev_gray_x[idx] * dev_gray_x[idx] + dev_gray_y[idx] * dev_gray_y[idx])) > 100 ? 0 : 255;
    }
}

extern "C" __declspec(dllexport) void RunNegateKernel(unsigned char* pictureIn, unsigned char* pictureOut, int width, int height, int widthStep, int channels)
{
    unsigned char* dev_in;
    unsigned char* dev_out;
    size_t size = (width * height * channels + widthStep * height) * sizeof(unsigned char);

    cudaMalloc((void**)&dev_in, size);
    cudaMalloc((void**)&dev_out, size);

    cudaMemcpy(dev_in, pictureIn, size, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x + 1, (height + block.y - 1) / block.y + 1);

    negateKernel << <grid, block >> > (dev_in, dev_out, width, height, widthStep, channels);
    cudaDeviceSynchronize();

    cudaMemcpy(pictureOut, dev_out, size, cudaMemcpyDeviceToHost);

    cudaFree(dev_in);
    cudaFree(dev_out);
}
extern "C" __declspec(dllexport) void RungammaKernel(unsigned char* pictureIn, unsigned char* pictureOut, int width, int height, int widthStep, int channels, float gamma) {
    unsigned char* dev_in;
    unsigned char* dev_out;
    unsigned char* dev_lookUp;
    size_t size = (width * height * channels + widthStep * height) * sizeof(unsigned char);

    //create a lookup tabble
    unsigned char lookUp[256];
    for (size_t i = 0; i < 256; i++) {
        float normalizedValue = static_cast<float>(i) / 255.0f; // Normalize i to [0, 1]
        unsigned char correctedValue = static_cast<unsigned char>(255 * pow(normalizedValue, gamma));
        lookUp[i] = correctedValue;
    }

    cudaMalloc((void**)&dev_in, size);
    cudaMalloc((void**)&dev_out, size);
    cudaMalloc((void**)&dev_lookUp, 256*sizeof(unsigned char));

    cudaMemcpy(dev_in, pictureIn, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_lookUp, lookUp, 256 * sizeof(unsigned char), cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x + 1, (height + block.y - 1) / block.y + 1);

    changeLookUpKernel << <grid, block >> > (dev_in, dev_out, width, height, widthStep, channels, dev_lookUp);
    

    cudaDeviceSynchronize();

    cudaMemcpy(pictureOut, dev_out, size, cudaMemcpyDeviceToHost);

    cudaFree(dev_in);
    cudaFree(dev_out);
    cudaFree(dev_lookUp);
}

extern "C" __declspec(dllexport) void RunLogKernel(unsigned char* pictureIn, unsigned char* pictureOut, int width, int height, int widthStep, int channels, float c) {
    unsigned char* dev_in;
    unsigned char* dev_out;
    unsigned char* dev_lookUp;
    size_t size = (width * height * channels + widthStep * height) * sizeof(unsigned char);

    //create a lookup tabble
    unsigned char lookUp[256];
    for (size_t i = 0; i < 256; i++) {
        float log = c * std::log(1 + i);
        lookUp[i] = static_cast<char>(std::min(255.0f, log));
    }
    cudaMalloc((void**)&dev_in, size);
    cudaMalloc((void**)&dev_out, size);
    cudaMalloc((void**)&dev_lookUp, 256 * sizeof(char));

    cudaMemcpy(dev_in, pictureIn, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_lookUp, lookUp, 256 * sizeof(char), cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x + 1, (height + block.y - 1) / block.y + 1);

    changeLookUpKernel << <grid, block >> > (dev_in, dev_out, width, height, widthStep, channels, dev_lookUp);


    cudaDeviceSynchronize();

    cudaMemcpy(pictureOut, dev_out, size, cudaMemcpyDeviceToHost);

    cudaFree(dev_in);
    cudaFree(dev_out);
    cudaFree(dev_lookUp);
}
extern "C" __declspec(dllexport) void RunGrayKernel(unsigned char* pictureIn, unsigned char* pictureOut, int width, int height, int widthStep, int channels) {
    unsigned char* dev_in;
    unsigned char* dev_out;
    size_t size = (width * height * channels + widthStep * height) * sizeof(unsigned char);
    size_t graySize = (width * height + widthStep * height) * sizeof(unsigned char);

    cudaMalloc((void**)&dev_in, size);
    cudaMalloc((void**)&dev_out, graySize);

    cudaMemcpy(dev_in, pictureIn, size, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x + 1, (height + block.y - 1) / block.y + 1);

    grayKernel << <grid, block >> > (dev_in, dev_out, width, height, widthStep, channels);


    cudaDeviceSynchronize();

    cudaMemcpy(pictureOut, dev_out, graySize, cudaMemcpyDeviceToHost);

    cudaFree(dev_in);
    cudaFree(dev_out);
}
extern "C" __declspec(dllexport) void RunHistogramKernel(unsigned char* pictureIn, int* histogramOut, int width, int height, int widthStep, int channels) {
    unsigned char* dev_out;
    unsigned char* dev_in;
    int* dev_histogramOut;
    
    size_t size = (width * height * channels + widthStep * height) * sizeof(unsigned char);
    size_t graySize = (width * height + widthStep * height) * sizeof(unsigned char);

    cudaMalloc((void**)&dev_out, graySize);
    cudaMalloc((void**)&dev_in, size);
    cudaMalloc((void**)&dev_histogramOut, 256 * sizeof(int));

    cudaMemcpy(dev_in, pictureIn, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_histogramOut, histogramOut, 256 * sizeof(int), cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x + 1, (height + block.y - 1) / block.y + 1);

    grayKernel << <grid, block >> > (dev_in, dev_out, width, height, widthStep, channels);
    cudaDeviceSynchronize();
    histogramKernel << <grid, block >> > (dev_out, dev_histogramOut, width, height, widthStep, channels);


    cudaDeviceSynchronize();

    cudaMemcpy(histogramOut, dev_histogramOut, 256 * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(dev_out);
    cudaFree(dev_histogramOut);
    cudaFree(dev_in);
}
extern "C" __declspec(dllexport) void RunHistogramEqualizationKernel(unsigned char* pictureIn, unsigned char* pictureOut, int width, int height, int widthStep, int channels) {
    unsigned char* dev_in;
    unsigned char* dev_out;
    int* dev_histogramOut;
    int* histogramOut = (int*)malloc((width * height + widthStep * height) * sizeof(unsigned char));
    unsigned char lookUp[256];
    unsigned char* dev_lookUp;

    size_t size = (width * height * channels + widthStep * height) * sizeof(unsigned char);
    size_t graySize = (width * height + widthStep * height) * sizeof(unsigned char);

    for (size_t i = 0; i < 256; i++)
    {
        histogramOut[i] = 0;
    }

    cudaMalloc((void**)&dev_in, size);
    cudaMalloc((void**)&dev_out, graySize);
    cudaMalloc((void**)&dev_histogramOut, 256 * sizeof(int));
    cudaMalloc((void**)&dev_lookUp, 256 * sizeof(unsigned char));

    cudaMemcpy(dev_in, pictureIn, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_histogramOut, histogramOut, 256 * sizeof(int), cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x + 1, (height + block.y - 1) / block.y + 1);


    //Szürkítés
    grayKernel << <grid, block >> > (dev_in, dev_out, width, height, widthStep, channels);
    cudaDeviceSynchronize();

    histogramKernel << <grid, block >> > (dev_out, dev_histogramOut, width, height, widthStep, channels);

    cudaDeviceSynchronize();
    //hisztogram
    cudaMemcpy(histogramOut, dev_histogramOut, 256 * sizeof(int), cudaMemcpyDeviceToHost);

    //lookup table
    int cumulateSum = 0;
    long pixelNumbers = width * height;
    for (size_t i = 0; i < 256; i++)
    {
        cumulateSum += histogramOut[i];
        lookUp[i] = 255 * ((double)cumulateSum) / pixelNumbers;
    }

    cudaMemcpy(dev_lookUp, lookUp, 256 * sizeof(unsigned char), cudaMemcpyHostToDevice);

    changeLookUpKernel << <grid, block >> > (dev_out, dev_out, width, height, widthStep, 1, dev_lookUp);
    
    cudaDeviceSynchronize();
    
    cudaMemcpy(pictureOut, dev_out, graySize, cudaMemcpyDeviceToHost);

    cudaFree(dev_in);
    cudaFree(dev_histogramOut);
    cudaFree(dev_lookUp);
    cudaFree(dev_out);
}
extern "C" __declspec(dllexport) void RunAVGKernel(unsigned char* pictureIn, unsigned char* pictureOut, int width, int height, int widthStep, int channels, int matrixDims) {
    unsigned char* dev_gray;
    unsigned char* dev_out;
    unsigned char* dev_in;
    int* dev_matrix;

    size_t size = (width * height * channels + widthStep * height) * sizeof(unsigned char);
    size_t graySize = (width * height + widthStep * height) * sizeof(unsigned char);
    
    int matrix_size = matrixDims*matrixDims;
    int* matrix = new int[matrix_size];

    for (int i = 0; i < matrix_size; i++) {
        matrix[i] = 1;
    }

    cudaMalloc((void**)&dev_in, size);
    cudaMalloc((void**)&dev_gray, graySize);
    cudaMalloc((void**)&dev_out, graySize);
    cudaMalloc((void**)&dev_matrix, matrix_size*sizeof(int));

    cudaMemcpy(dev_in, pictureIn, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_matrix, matrix, matrix_size * sizeof(int), cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x + 1, (height + block.y - 1) / block.y + 1);

    grayKernel << <grid, block >> > (dev_in, dev_gray, width, height, widthStep, channels);
    cudaDeviceSynchronize();
    
   
    MaskKernel<< <grid, block >> > (dev_gray, dev_out, width, height, widthStep, channels, dev_matrix, matrixDims, matrixDims);

    cudaDeviceSynchronize();

    cudaMemcpy(pictureOut, dev_out, graySize, cudaMemcpyDeviceToHost);

    cudaFree(dev_out);
    cudaFree(dev_in);
    cudaFree(dev_gray);
    cudaFree(dev_matrix);
}
extern "C" __declspec(dllexport) void RunGaussKernel(unsigned char* pictureIn, unsigned char* pictureOut, int width, int height, int widthStep, int channels, int matrixDims, double sigma) {
    unsigned char* dev_in;
    unsigned char* dev_gray;
    unsigned char* dev_out;
    int* dev_matrix;

    size_t size = (width * height * channels + widthStep * height) * sizeof(unsigned char);
    size_t graySize = (width * height + widthStep * height) * sizeof(unsigned char);

    int matrix_size = matrixDims * matrixDims;
    int* matrix = new int[matrix_size];
    int halfSize = matrixDims / 2;
    int idx = 0;
    long sum = 0;
    for (int i = -halfSize; i <= halfSize; i++) {
        for (int j = -halfSize; j <= halfSize; j++) {
            double value = (1.0 / (2.0 * 3.141592 * sigma * sigma)) *
                std::exp(-(i * i + j * j) / (2 * sigma * sigma));
            matrix[idx] = value * 10000;
            sum += matrix[idx++];
        }
    }
    for (size_t i = 0; i < matrix_size; i++)
    {
        matrix[i] = matrix[i] / (double)sum * 10000;
    }

    cudaMalloc((void**)&dev_in, size);
    cudaMalloc((void**)&dev_gray, graySize);
    cudaMalloc((void**)&dev_out, graySize);
    cudaMalloc((void**)&dev_matrix, matrix_size * sizeof(int));

    cudaMemcpy(dev_in, pictureIn, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_matrix, matrix, matrix_size * sizeof(int), cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x + 1, (height + block.y - 1) / block.y + 1);

    grayKernel << <grid, block >> > (dev_in, dev_gray, width, height, widthStep, channels);
    cudaDeviceSynchronize();


    MaskKernel << <grid, block >> > (dev_gray, dev_out, width, height, widthStep, channels, dev_matrix, matrixDims, matrixDims);
    cudaDeviceSynchronize();

    cudaMemcpy(pictureOut, dev_out, graySize, cudaMemcpyDeviceToHost);

    cudaFree(dev_out);
    cudaFree(dev_in);
    cudaFree(dev_gray);
    cudaFree(dev_matrix);
}
extern "C" __declspec(dllexport) void RunSobelKernel(unsigned char* pictureIn, unsigned char* pictureOut, int width, int height, int widthStep, int channels)
{
    unsigned char* dev_in;
    unsigned char* dev_gray;
    unsigned char* dev_gray_blure;
    unsigned char* dev_gray_x;
    unsigned char* dev_gray_y;
    unsigned char* dev_out;
    int* dev_matrix_x;
    int* dev_matrix_y;
    int* dev_matrix_blure;

    size_t size = (width * height * channels + widthStep * height) * sizeof(unsigned char);
    size_t graySize = (width * height + widthStep * height) * sizeof(unsigned char);
    int matrixDims = 3;
    int matrix_size = 9;
    int matrix_x[9] = {1, 0, -1, 2, 0, -2, 1, 0, -1};
    int matrix_y[9] = { 1, 2, 1, 0, 0, 0, -1, -2, -1 };

    cudaMalloc((void**)&dev_in, size);
    cudaMalloc((void**)&dev_gray, graySize);
    cudaMalloc((void**)&dev_gray_blure, graySize);
    cudaMalloc((void**)&dev_gray_x, graySize);
    cudaMalloc((void**)&dev_gray_y, graySize);
    cudaMalloc((void**)&dev_out, graySize);
    cudaMalloc((void**)&dev_matrix_x, matrix_size * sizeof(int));
    cudaMalloc((void**)&dev_matrix_y, matrix_size * sizeof(int));

    cudaMemcpy(dev_in, pictureIn, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_matrix_x, matrix_x, matrix_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_matrix_y, matrix_y, matrix_size * sizeof(int), cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x + 1, (height + block.y - 1) / block.y + 1);

    grayKernel << <grid, block >> > (dev_in, dev_gray, width, height, widthStep, channels);
    cudaDeviceSynchronize();

    double sigma = 0.5;
    int matrixDimsBlure = 7;
    int matrix_size_blure = matrixDimsBlure * matrixDimsBlure;
    int* matrix = new int[matrix_size_blure];
    int halfSize = matrixDimsBlure / 2;
    int idx = 0;
    long sum = 0;
    for (int i = -halfSize; i <= halfSize; i++) {
        for (int j = -halfSize; j <= halfSize; j++) {
            double value = (1.0 / (2.0 * 3.141592 * sigma * sigma)) *
                std::exp(-(i * i + j * j) / (2 * sigma * sigma));
            matrix[idx] = value * 10000;
            sum += matrix[idx++];
        }
    }
    for (size_t i = 0; i < matrix_size_blure; i++)
    {
        matrix[i] = matrix[i] / (double)sum * 10000;
    }
    cudaMalloc((void**)&dev_matrix_blure, matrix_size_blure * sizeof(int));
    cudaMemcpy(dev_matrix_blure, matrix, matrix_size_blure * sizeof(int), cudaMemcpyHostToDevice);

    MaskKernel << <grid, block >> > (dev_gray, dev_gray_blure, width, height, widthStep, channels, dev_matrix_blure, matrixDimsBlure, matrixDimsBlure);
    cudaDeviceSynchronize();

    MaskKernel << <grid, block >> > (dev_gray_blure, dev_gray_x, width, height, widthStep, channels, dev_matrix_x, matrixDims, matrixDims);
    MaskKernel << <grid, block >> > (dev_gray_blure, dev_gray_y, width, height, widthStep, channels, dev_matrix_y, matrixDims, matrixDims);

    cudaDeviceSynchronize();
    SobelKernel << <grid, block >> > (dev_gray_x, dev_gray_y, dev_out, width, height, widthStep);

    cudaMemcpy(pictureOut, dev_out, graySize, cudaMemcpyDeviceToHost);

    cudaFree(dev_out);
    cudaFree(dev_in);
    cudaFree(dev_gray);
    cudaFree(dev_gray_x);
    cudaFree(dev_gray_y);
    cudaFree(dev_matrix_x);
    cudaFree(dev_matrix_y);
    cudaFree(dev_matrix_blure);
}

