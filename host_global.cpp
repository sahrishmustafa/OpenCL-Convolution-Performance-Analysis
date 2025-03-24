#define CL_TARGET_OPENCL_VERSION 300
#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>
#include <string.h>
#include <time.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <filesystem>

using namespace cv;
using namespace std;
namespace fs = std::filesystem;

#define CHECK_ERROR(err, msg) if (err != CL_SUCCESS) { fprintf(stderr, "Error: %s (%d)\n", msg, err); exit(EXIT_FAILURE); }

// Function to load images
void loadPNG(const string& filename, float* floatImage, int& width, int& height) {
    Mat image = imread(filename, IMREAD_GRAYSCALE);
    if (image.empty()) {
        printf("Error: Could not load image %s\n", filename.c_str());
        exit(EXIT_FAILURE);
    }
    width = image.cols;
    height = image.rows;
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            floatImage[i * width + j] = image.at<uchar>(i, j) / 255.0f;
        }
    }
}

void padImage(const float* inputImage, float* paddedImage, int width, int height, int kernelSize) {
    int halfK = kernelSize / 2;
    int paddedWidth = width + 2 * halfK;
    int paddedHeight = height + 2 * halfK;
    memset(paddedImage, 0, sizeof(float) * paddedWidth * paddedHeight);
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            paddedImage[(y + halfK) * paddedWidth + (x + halfK)] = inputImage[y * width + x];
        }
    }
}

char* loadKernelSource(const char* filename, size_t* length) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        perror("Failed to load kernel file");
        exit(EXIT_FAILURE);
    }
    fseek(file, 0, SEEK_END);
    *length = ftell(file);
    rewind(file);
    char* source = (char*)malloc(*length + 1);
    fread(source, 1, *length, file);
    source[*length] = '\0';
    fclose(file);
    return source;
}

int main() {
    vector<string> imageFiles;
    for (const auto& entry : fs::directory_iterator("dataset/grayscale/512")) {
        imageFiles.push_back(entry.path().string());
    }
    
    if (!fs::exists("output_parallel")) {
        fs::create_directory("output_parallel");
    }
    
    const int kernelSize = 3;
    float kernel[9] = {1, 1, 1, 0, 0, 0, -1, -1, -1};

    cl_int err;
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel clKernel;
    chrono::duration<double>  total_elapsed;

    err = clGetPlatformIDs(1, &platform, NULL);
    cout << "getting platforms" << endl;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    cout << "getting device" << endl;
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    cout << "creating context" << endl;
    queue = clCreateCommandQueueWithProperties(context, device, NULL, &err);
    cout << "creating queue" << endl;

    size_t sourceSize;
    char* source = loadKernelSource("conv.cl", &sourceSize);
    program = clCreateProgramWithSource(context, 1, (const char**)&source, &sourceSize, &err);
    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    clKernel = clCreateKernel(program, "apply_conv", &err);
    cout << "creating kernel" << endl;

    for (const auto& inputFile : imageFiles) {
        int width, height;
        float* inputImage = (float*)malloc(512 * 512 * sizeof(float));
        loadPNG(inputFile, inputImage, width, height);
        int halfK = kernelSize / 2;
        int paddedWidth = width + 2 * halfK;
        int paddedHeight = height + 2 * halfK;
        float* paddedImage = (float*)malloc(paddedWidth * paddedHeight * sizeof(float));
        float* outputImage = (float*)malloc(width * height * sizeof(float));
        cout << "allocating memory" << endl;

        padImage(inputImage, paddedImage, width, height, kernelSize);
        cout << "padding image" << endl;
        cl_mem d_paddedImage = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                                             paddedWidth * paddedHeight * sizeof(float), paddedImage, &err);
                                             cout << "creating pad buffer" << endl;
        cl_mem d_outputImage = clCreateBuffer(context, CL_MEM_WRITE_ONLY, width * height * sizeof(float), NULL, &err);
        cout << "creating output buffer" << endl;
        cl_mem d_kernel = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                                         kernelSize * kernelSize * sizeof(float), kernel, &err);
                                         cout << "creating kernel filter buffer" << endl;
        err = clSetKernelArg(clKernel, 0, sizeof(cl_mem), &d_paddedImage);
        err |= clSetKernelArg(clKernel, 1, sizeof(cl_mem), &d_outputImage);
        err |= clSetKernelArg(clKernel, 2, sizeof(cl_mem), &d_kernel);
        err |= clSetKernelArg(clKernel, 3, sizeof(int), &paddedWidth);
        err |= clSetKernelArg(clKernel, 4, sizeof(int), &paddedHeight);
        err |= clSetKernelArg(clKernel, 5, sizeof(int), &width);
        err |= clSetKernelArg(clKernel, 6, sizeof(int), &height);
        err |= clSetKernelArg(clKernel, 7, sizeof(int), &kernelSize);
        cout << "setting arguments queue" << endl;

        size_t globalSize[2] = {size_t(width), size_t(height)};
        size_t localSize[2] = {16, 16};
        // Start timing
        auto start = chrono::high_resolution_clock::now();
        err = clEnqueueNDRangeKernel(queue, clKernel, 2, NULL, globalSize, localSize, 0, NULL, NULL);
        // End timing
        auto end = chrono::high_resolution_clock::now();
        chrono::duration<double> elapsed = end - start;
        total_elapsed += elapsed;
        clEnqueueReadBuffer(queue, d_outputImage, CL_TRUE, 0, width * height * sizeof(float), outputImage, 0, NULL, NULL);
        
        Mat outputMat(height, width, CV_32F, outputImage);
        outputMat *= 255.0;
        Mat outputMat8U;
        outputMat.convertTo(outputMat8U, CV_8U);
        string outputFile = "output_parallel/" + fs::path(inputFile).filename().string();
        imwrite(outputFile, outputMat8U);
        
        free(inputImage);
        free(paddedImage);
        free(outputImage);
        clReleaseMemObject(d_paddedImage);
        clReleaseMemObject(d_outputImage);
        clReleaseMemObject(d_kernel);
    }
    free(source);
    clReleaseKernel(clKernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    printf("Processing completed successfully!\n");
    cout << "Total Time to extract images: " << total_elapsed.count() << " seconds\n";
    return 0;
}
