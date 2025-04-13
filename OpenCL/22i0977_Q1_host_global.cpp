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
#define BLOCKSIZE 16
#define KERNEL_SIZE 3
#define INPUT_PATH "dataset/grayscale/"
#define OUTPUT_PATH "output_parallel_global"

void loadImage(const string& filename, float* floatImage, int expected_width, int expected_height) {
    Mat image = imread(filename, IMREAD_GRAYSCALE);
    if (image.empty()) {
        printf("Error: Could not load image %s\n", filename.c_str());
        exit(EXIT_FAILURE);
    }
    if (image.cols != expected_width || image.rows != expected_height) {
        printf("Error: Image %s has dimensions %dx%d, expected %dx%d\n", filename.c_str(), image.cols, image.rows, expected_width, expected_height);
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < expected_height; i++) {
        for (int j = 0; j < expected_width; j++) {
            floatImage[i * expected_width + j] = image.at<uchar>(i, j) / 255.0f;
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

int main(int argc, char* argv[]) {
    if (argc != 2) {
        cerr << "Usage: " << argv[0] << " <512 or 1024>" << endl;
        return 1;
    }

    string resolution = argv[1];
    if (resolution != "512" && resolution != "1024") {
        cerr << "Error: Invalid input. Please enter '512' or '1024'." << endl;
        return 1;
    }

    vector<string> imageFiles;
    string inputFolder = INPUT_PATH + resolution;
    for (const auto& entry : fs::directory_iterator(inputFolder)) {
        imageFiles.push_back(entry.path().string());
    }
    
    if (!fs::exists(OUTPUT_PATH)) {
        fs::create_directory(OUTPUT_PATH);
    }
    
    const int kernelSize = KERNEL_SIZE;
    float kernel[9] = {1, 1, 1, 0, 0, 0, -1, -1, -1};

    cl_int err;
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel clKernel;
    chrono::duration<double> total_elapsed;
    int fileCount = 0; 

    err = clGetPlatformIDs(1, &platform, NULL);
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    queue = clCreateCommandQueueWithProperties(context, device, NULL, &err);

    size_t sourceSize;
    char* source = loadKernelSource("conv.cl", &sourceSize);
    program = clCreateProgramWithSource(context, 1, (const char**)&source, &sourceSize, &err);
    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    clKernel = clCreateKernel(program, "apply_conv", &err);

    int width = stoi(resolution);
    int height = width;
    int halfK = kernelSize / 2;
    int paddedWidth = width + 2 * halfK;
    int paddedHeight = height + 2 * halfK;

    // Allocate host buffers once
    float* inputImage = (float*)malloc(width * height * sizeof(float));
    float* paddedImage = (float*)malloc(paddedWidth * paddedHeight * sizeof(float));
    float* outputImage = (float*)malloc(width * height * sizeof(float));

    // Create OpenCL buffers once
    cl_mem d_paddedImage = clCreateBuffer(context, CL_MEM_READ_ONLY, paddedWidth * paddedHeight * sizeof(float), NULL, &err);
    CHECK_ERROR(err, "Creating padded image buffer");
    cl_mem d_outputImage = clCreateBuffer(context, CL_MEM_WRITE_ONLY, width * height * sizeof(float), NULL, &err);
    CHECK_ERROR(err, "Creating output buffer");
    cl_mem d_kernel = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, kernelSize * kernelSize * sizeof(float), kernel, &err);
    CHECK_ERROR(err, "Creating kernel buffer");

    // Set kernel arguments once
    err = clSetKernelArg(clKernel, 0, sizeof(cl_mem), &d_paddedImage);
    err |= clSetKernelArg(clKernel, 1, sizeof(cl_mem), &d_outputImage);
    err |= clSetKernelArg(clKernel, 2, sizeof(cl_mem), &d_kernel);
    err |= clSetKernelArg(clKernel, 3, sizeof(int), &paddedWidth);
    err |= clSetKernelArg(clKernel, 4, sizeof(int), &paddedHeight);
    err |= clSetKernelArg(clKernel, 5, sizeof(int), &width);
    err |= clSetKernelArg(clKernel, 6, sizeof(int), &height);
    err |= clSetKernelArg(clKernel, 7, sizeof(int), &kernelSize);
    CHECK_ERROR(err, "Setting kernel arguments failed");

    size_t globalSize[2] = {static_cast<size_t>(width), static_cast<size_t>(height)};
    size_t localSize[2] = {BLOCKSIZE, BLOCKSIZE};

    for (const auto& inputFile : imageFiles) {
        loadImage(inputFile, inputImage, width, height);
        padImage(inputImage, paddedImage, width, height, kernelSize);

        // Write padded image to device buffer
        err = clEnqueueWriteBuffer(queue, d_paddedImage, CL_TRUE, 0, paddedWidth * paddedHeight * sizeof(float), paddedImage, 0, NULL, NULL);
        CHECK_ERROR(err, "Writing padded image to device failed");

        // Start timing
        auto start = chrono::high_resolution_clock::now();
        err = clEnqueueNDRangeKernel(queue, clKernel, 2, NULL, globalSize, localSize, 0, NULL, NULL);
        CHECK_ERROR(err, "Enqueuing kernel failed");
        // End timing
        auto end = chrono::high_resolution_clock::now();
        chrono::duration<double> elapsed = end - start;
        total_elapsed += elapsed;

        // Read output from device
        err = clEnqueueReadBuffer(queue, d_outputImage, CL_TRUE, 0, width * height * sizeof(float), outputImage, 0, NULL, NULL);
        CHECK_ERROR(err, "Reading output buffer failed");

        // Save output image
        Mat outputMat(height, width, CV_32F, outputImage);
        outputMat *= 255.0;
        Mat outputMat8U;
        outputMat.convertTo(outputMat8U, CV_8U);
        string outputFile = string(OUTPUT_PATH) + "/" + fs::path(inputFile).filename().string();
        imwrite(outputFile, outputMat8U);

        fileCount++;
    }

    // Cleanup
    free(inputImage);
    free(paddedImage);
    free(outputImage);
    free(source);
    clReleaseMemObject(d_paddedImage);
    clReleaseMemObject(d_outputImage);
    clReleaseMemObject(d_kernel);
    clReleaseKernel(clKernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    printf("Processing using global mem completed successfully!\n");
    cout << "Total Time to extract " << fileCount << " images: " << total_elapsed.count() << " seconds\n";
    return 0;
}