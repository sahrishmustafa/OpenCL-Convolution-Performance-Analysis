#define CL_TARGET_OPENCL_VERSION 300
#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>
#include <string.h>
#include <time.h>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;


// Function to check OpenCL errors
#define CHECK_ERROR(err, msg) if (err != CL_SUCCESS) { fprintf(stderr, "Error: %s (%d)\n", msg, err); exit(EXIT_FAILURE); }

// Function to load a PNG image and convert it to a float array
void loadPNG(const string& filename, float* floatImage, int& width, int& height) {
    Mat image = imread(filename, IMREAD_GRAYSCALE); // Load as grayscale
    if (image.empty()) {
        printf("Error: Could not load image %s\n", filename.c_str());
        exit(EXIT_FAILURE);
    }

    width = image.cols;
    height = image.rows;

    // Normalize the image (0-255) -> (0.0 - 1.0) and store in floatImage
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            floatImage[i * width + j] = image.at<uchar>(i, j) / 255.0f;
        }
    }
}

// Function to load OpenCL kernel from file
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
    // Define image and kernel sizes
    const int kernelSize = 3;
    int width = 512, height = 512;
    int halfK = kernelSize / 2;
    int paddedWidth = width + 2 * halfK;
    int paddedHeight = height + 2 * halfK;

    // Allocate memory for images
    float* inputImage = (float*)malloc(width * height * sizeof(float));

    float* outputImage = (float*)malloc(width * height * sizeof(float));

    // Initialize input image
    string inputFile = "flickr_cat_000003.png";
    loadPNG(inputFile, inputImage, width, height);

    // Define a simple edge detection kernel
    float kernel[9] = {1, 1, 1, 0, 0, 0, -1, -1, -1};


    // OpenCL setup
    cl_int err;
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel clKernel;
    cl_mem d_paddedImage, d_outputImage, d_kernel;

    // Get OpenCL platform
    err = clGetPlatformIDs(1, &platform, NULL);
    CHECK_ERROR(err, "Failed to get platform");

    // Get GPU device
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    CHECK_ERROR(err, "Failed to get device");

    // Create OpenCL context
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    CHECK_ERROR(err, "Failed to create context");

    // Create command queue
    cl_queue_properties props[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
    queue = clCreateCommandQueueWithProperties(context, device, props, &err);
    CHECK_ERROR(err, "Failed to create command queue");


    // Load and build OpenCL kernel
    size_t sourceSize;
    char* source = loadKernelSource("conv.cl", &sourceSize);
    program = clCreateProgramWithSource(context, 1, (const char**)&source, &sourceSize, &err);
    CHECK_ERROR(err, "Failed to create program");

    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t logSize;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);
        char* log = (char*)malloc(logSize);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, log, NULL);
        fprintf(stderr, "Build error:\n%s\n", log);
        free(log);
        exit(EXIT_FAILURE);
    }

    // Create kernel
    clKernel = clCreateKernel(program, "apply_conv_shared", &err);
    CHECK_ERROR(err, "Failed to create kernel");

    // Allocate device memory
    d_paddedImage = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                                width * height * sizeof(float), inputImage, &err);
    CHECK_ERROR(err, "Failed to create padded image buffer");

    d_outputImage = clCreateBuffer(context, CL_MEM_WRITE_ONLY, width * height * sizeof(float), NULL, &err);
    CHECK_ERROR(err, "Failed to create output image buffer");

    d_kernel = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                              kernelSize * kernelSize * sizeof(float), kernel, &err);
    CHECK_ERROR(err, "Failed to create kernel buffer");

    // Set kernel arguments
    err = clSetKernelArg(clKernel, 0, sizeof(cl_mem), &d_paddedImage);
    err |= clSetKernelArg(clKernel, 1, sizeof(cl_mem), &d_outputImage);
    err |= clSetKernelArg(clKernel, 2, sizeof(cl_mem), &d_kernel);
    err |= clSetKernelArg(clKernel, 3, sizeof(int), &width);
    err |= clSetKernelArg(clKernel, 4, sizeof(int), &height);
    err |= clSetKernelArg(clKernel, 5, sizeof(int), &kernelSize);
    CHECK_ERROR(err, "Failed to set kernel arguments");

    // Define global and local work sizes
    size_t globalSize[2] = {width, height};
    size_t localSize[2] = {16, 16};

    // Execute kernel
    err = clEnqueueNDRangeKernel(queue, clKernel, 2, NULL, globalSize, localSize, 0, NULL, NULL);
    CHECK_ERROR(err, "Failed to execute kernel");

    // Read back the result
    clEnqueueReadBuffer(queue, d_outputImage, CL_TRUE, 0, width * height * sizeof(float), outputImage, 0, NULL, NULL);

    // Convert outputImage (float array) to OpenCV Mat
    Mat outputMat(height, width, CV_32F, outputImage);

    // Scale the values from (0.0 - 1.0) to (0 - 255)
    outputMat *= 255.0;

    // Convert to 8-bit format for saving
    Mat outputMat8U;
    outputMat.convertTo(outputMat8U, CV_8U);

    // Save as PNG
    string outputFile = "output.png";
    imwrite(outputFile, outputMat8U);

    printf("Output image saved as %s\n", outputFile.c_str());

    // Clean up
    free(inputImage);
    free(outputImage);
    free(source);
    clReleaseMemObject(d_paddedImage);
    clReleaseMemObject(d_outputImage);
    clReleaseMemObject(d_kernel);
    clReleaseKernel(clKernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    printf("Convolution completed successfully!\n");
    return 0;
}

