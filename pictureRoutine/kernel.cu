
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
__global__ void LaplaceKernel(unsigned char* dev_out, int width, int height, int widthStep) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        int idx = y * width + y * widthStep + x;
        dev_out[idx] = (dev_out[idx]*dev_out[idx]) > 50 ? 255 : 0;
    }
}
__global__ void SumAreaKernel(unsigned char* dev_in, unsigned int* dev_sum, int width, int height, int widthStep, int dims) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        int idx = y * width + y * widthStep + x;
        int idx_sum = y * width + x;
        int half_dim = dims / 2;

        for (int i = -half_dim; i <= half_dim; i++) {
            int x_in_picture = x + i;
            

            for (int j = -half_dim; j <= half_dim; j++) {
                int y_in_picture = y + j;

                if (x_in_picture >= 0 && x_in_picture < width && y_in_picture >= 0 && y_in_picture < height) {
                    
                    dev_sum[y * width + x] += dev_in[x_in_picture+y_in_picture*(width + widthStep)];
                }
            }
        }
    }
}
__global__ void KLTKernel(unsigned char* dev_gray_in, unsigned char* dev_out, unsigned int* dev_derival_xy, unsigned int* dev_Doublederival_x, unsigned int* dev_Doublederival_y, int width, int height, int widthStep, double k, int th) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return; 

    int idx = y * width + x;

    unsigned int A = dev_Doublederival_x[idx];
    unsigned int B = dev_Doublederival_y[idx];
    unsigned int C = dev_derival_xy[idx];

    double R = ((A * B - C * C) + k * (A + B) * (A + B))/100000;

    int channels = 3;
    int gidx = y * widthStep + y * width + x;
    int cidx = (width * y * channels) + y * widthStep + x * channels;;
    dev_out[cidx] = dev_gray_in[gidx];
    dev_out[cidx+1] = dev_gray_in[gidx];
    dev_out[cidx+2] = dev_gray_in[gidx];

    if (R > th) {
        for (int dy = -2; dy <= 2; dy++) {
            for (int dx = -2; dx <= 2; dx++) {
                int nx = x + dx;
                int ny = y + dy;

                if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                    int nidx = ny * width * 3 + ny * widthStep + nx * 3;
                    dev_out[nidx] = 0;       
                    dev_out[nidx + 1] = 0;     
                    dev_out[nidx + 2] = 255;     
                }
            }
        }
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
extern "C" __declspec(dllexport) void RunLaplaceKernel(unsigned char* pictureIn, unsigned char* pictureOut, int width, int height, int widthStep, int channels)
{
    unsigned char* dev_in;
    unsigned char* dev_gray;
    unsigned char* dev_gray_blure;
    unsigned char* dev_out;
    int* dev_matrix_laplace;
    int* dev_matrix_blure;

    size_t size = (width * height * channels + widthStep * height) * sizeof(unsigned char);
    size_t graySize = (width * height + widthStep * height) * sizeof(unsigned char);
    int matrixDims = 3;
    int matrix_size = 9;
    int matrix_laplace[9] = {   0, -1, 0,
                                -1, 4, -1,
                                0, -1, 0 };

    cudaMalloc((void**)&dev_in, size);
    cudaMalloc((void**)&dev_gray, graySize);
    cudaMalloc((void**)&dev_gray_blure, graySize);
    cudaMalloc((void**)&dev_out, graySize);
    cudaMalloc((void**)&dev_matrix_laplace, matrix_size * sizeof(int));

    cudaMemcpy(dev_in, pictureIn, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_matrix_laplace, matrix_laplace, matrix_size * sizeof(int), cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x + 1, (height + block.y - 1) / block.y + 1);

    grayKernel << <grid, block >> > (dev_in, dev_gray, width, height, widthStep, channels);

    {
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
        cudaDeviceSynchronize();
    }
    MaskKernel << <grid, block >> > (dev_gray_blure, dev_out, width, height, widthStep, channels, dev_matrix_laplace, matrixDims, matrixDims);
    cudaDeviceSynchronize();

    LaplaceKernel << <grid, block >> > (dev_out, width, height, widthStep);

    cudaMemcpy(pictureOut, dev_out, graySize, cudaMemcpyDeviceToHost);

    cudaFree(dev_in);
    cudaFree(dev_gray);
    cudaFree(dev_out);
    cudaFree(dev_matrix_laplace);
    cudaFree(dev_matrix_blure);
}
extern "C" __declspec(dllexport) void RunImportantPointKernel(unsigned char* pictureIn, unsigned char* pictureOut, int width, int height, int widthStep, int channels)
{
    unsigned char* dev_in;
    unsigned char* dev_gray;
    unsigned char* dev_gray_blure;
    unsigned char* dev_derival_x;
    unsigned char* dev_derival_y;
    unsigned char* dev_derival_xy;
    unsigned char* dev_DoubleDerival_x;
    unsigned char* dev_DoubleDerival_y;

    unsigned int* dev_sumderival_xy;
    unsigned int* dev_sumDoubleDerival_x;
    unsigned int* dev_sumDoubleDerival_y;

    unsigned char* dev_out;
    int* dev_matrix_derival_x;
    int* dev_matrix_derival_y;
    int* dev_matrix_blure;

    size_t size = (width * height * channels + widthStep * height) * sizeof(unsigned char);
    size_t graySize = (width * height + widthStep * height) * sizeof(unsigned char);
    size_t sumderival = (width * height) * sizeof(unsigned int);
    
    int matrixDims = 3;
    int matrix_size = 9;
    int matrix_derival_x[9] = { 1, 1, 1,
                                0, 0, 0,
                                -1, -1, -1 };

    int matrix_derival_y[9] = { 1, 0, -1,
                                1, 0, -1,
                                1, 0, -1 };
    cudaMalloc((void**)&dev_in, size);
    cudaMalloc((void**)&dev_gray, graySize);
    cudaMalloc((void**)&dev_gray_blure, graySize);
    cudaMalloc((void**)&dev_derival_x, graySize);
    cudaMalloc((void**)&dev_derival_y, graySize);
    cudaMalloc((void**)&dev_derival_xy, graySize);
    cudaMalloc((void**)&dev_DoubleDerival_x, graySize);
    cudaMalloc((void**)&dev_DoubleDerival_y, graySize);

    cudaMalloc((void**)&dev_sumderival_xy, sumderival);
    cudaMalloc((void**)&dev_sumDoubleDerival_x, sumderival);
    cudaMalloc((void**)&dev_sumDoubleDerival_y, sumderival);
    cudaMalloc((void**)&dev_out, size);

    cudaMalloc((void**)&dev_matrix_derival_x, matrix_size * sizeof(int));
    cudaMalloc((void**)&dev_matrix_derival_y, matrix_size * sizeof(int));

    cudaMemcpy(dev_in, pictureIn, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_matrix_derival_x, matrix_derival_x, matrix_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_matrix_derival_y, matrix_derival_y, matrix_size * sizeof(int), cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x + 1, (height + block.y - 1) / block.y + 1);

    grayKernel << <grid, block >> > (dev_in, dev_gray, width, height, widthStep, channels);

    {
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
    }
    
    MaskKernel << <grid, block >> > (dev_gray_blure, dev_derival_x, width, height, widthStep, channels, dev_matrix_derival_x, matrixDims, matrixDims);
    MaskKernel << <grid, block >> > (dev_gray_blure, dev_derival_y, width, height, widthStep, channels, dev_matrix_derival_y, matrixDims, matrixDims);
    cudaDeviceSynchronize();
    
    MaskKernel << <grid, block >> > (dev_derival_x, dev_derival_xy, width, height, widthStep, channels, dev_matrix_derival_y, matrixDims, matrixDims);
    MaskKernel << <grid, block >> > (dev_derival_x, dev_DoubleDerival_x, width, height, widthStep, channels, dev_matrix_derival_x, matrixDims, matrixDims);
    MaskKernel << <grid, block >> > (dev_derival_y, dev_DoubleDerival_y, width, height, widthStep, channels, dev_matrix_derival_y, matrixDims, matrixDims);
    cudaDeviceSynchronize();

    int area = 7;
    SumAreaKernel<< <grid, block >> > (dev_derival_xy, dev_sumderival_xy, width, height, widthStep, area);
    SumAreaKernel<< <grid, block >> > (dev_DoubleDerival_x, dev_sumDoubleDerival_x, width, height, widthStep, area);
    SumAreaKernel<< <grid, block >> > (dev_DoubleDerival_y, dev_sumDoubleDerival_y, width, height, widthStep, area);

    double k = 0.25;
    int th = 43000;
    KLTKernel << <grid, block >> > (dev_gray, dev_out, dev_sumderival_xy, dev_sumDoubleDerival_x, dev_sumDoubleDerival_y, width, height, widthStep, k, th);

    cudaDeviceSynchronize();



    cudaMemcpy(pictureOut, dev_out, size, cudaMemcpyDeviceToHost);

    cudaFree(dev_in);
    cudaFree(dev_gray);
    cudaFree(dev_gray_blure);
    cudaFree(dev_derival_x);
    cudaFree(dev_derival_y);
    cudaFree(dev_derival_xy);
    cudaFree(dev_DoubleDerival_x);
    cudaFree(dev_DoubleDerival_y);
    cudaFree(dev_sumderival_xy);
    cudaFree(dev_sumDoubleDerival_x);
    cudaFree(dev_sumDoubleDerival_y);
    cudaFree(dev_out);
    cudaFree(dev_matrix_blure);
    cudaFree(dev_matrix_derival_x);
    cudaFree(dev_matrix_derival_y);
}

