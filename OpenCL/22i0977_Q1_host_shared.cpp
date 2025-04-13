#define CL_TARGET_OPENCL_VERSION 300
#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>
#include <string.h>
#include <time.h>
#include <opencv2/opencv.hpp>
#include <filesystem>

using namespace cv;
using namespace std;
namespace fs = std::filesystem;


// check OpenCL errors
#define CHECK_ERROR(err, msg) if (err != CL_SUCCESS) { fprintf(stderr, "Error: %s (%d)\n", msg, err); exit(EXIT_FAILURE); }
#define BLOCKSIZE 16
#define INPUT_PATH "dataset/grayscale/"
#define OUTPUT_PATH "output_parallel_shared"
#define KERNEL_SIZE 3

// Function to load a PNG image to array
void loadImage(const string& filename, float* floatImage, int& width, int& height) {
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

// load OpenCL kernel 
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

    const int kernelSize = KERNEL_SIZE;
    
    // get width and height
    string inputFolder = string(INPUT_PATH) + resolution;
    int width, height;
    width = stoi(resolution); height = stoi(resolution);
    
    // get image paths
    vector<string> imageFiles;
    for (const auto& entry : fs::directory_iterator(inputFolder)) {
        imageFiles.push_back(entry.path().string());
    }
    
    if (!fs::exists(OUTPUT_PATH)) {
        fs::create_directory(OUTPUT_PATH);
    }

    // declare vars
    float* inputImage = nullptr;
    float* outputImage = nullptr;
    float kernel[9] = {1, 1, 1, 0, 0, 0, -1, -1, -1};

    cl_int err;
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel clKernel;
    cl_mem d_inputImage, d_outputImage, d_kernel;
    chrono::duration<double>  total_elapsed;
    int fileCount = 0;

    // get platfrom
    err = clGetPlatformIDs(1, &platform, NULL);
    CHECK_ERROR(err, "Failed to get platform");

    // get device
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    CHECK_ERROR(err, "Failed to get device");

    // get context
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    CHECK_ERROR(err, "Failed to create context");

    // make queue
    cl_queue_properties props[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
    queue = clCreateCommandQueueWithProperties(context, device, props, &err);
    CHECK_ERROR(err, "Failed to create command queue");

    // make kernel program
    size_t sourceSize;
    char* source = loadKernelSource("conv.cl", &sourceSize);
    program = clCreateProgramWithSource(context, 1, (const char**)&source, &sourceSize, &err);
    CHECK_ERROR(err, "Failed to create program");

    // build kernel program
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

    clKernel = clCreateKernel(program, "apply_conv_shared", &err);
    CHECK_ERROR(err, "Failed to create kernel");

    // initialize image arrays
    inputImage = (float*)malloc(width * height * sizeof(float));
    outputImage = (float*)malloc(width * height * sizeof(float));

    // create input image buffer
    d_inputImage = clCreateBuffer(context, CL_MEM_READ_ONLY, width * height * sizeof(float), NULL, &err);
    CHECK_ERROR(err, "Failed to create padded image buffer");

    // create output image buffer
    d_outputImage = clCreateBuffer(context, CL_MEM_WRITE_ONLY, width * height * sizeof(float), NULL, &err);
    CHECK_ERROR(err, "Failed to create output image buffer");

    // create kernel filter buffer
    d_kernel = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                              kernelSize * kernelSize * sizeof(float), kernel, &err);
    CHECK_ERROR(err, "Failed to create kernel buffer");

    // set all arguments
    err = clSetKernelArg(clKernel, 0, sizeof(cl_mem), &d_inputImage);
    err |= clSetKernelArg(clKernel, 1, sizeof(cl_mem), &d_outputImage);
    err |= clSetKernelArg(clKernel, 2, sizeof(cl_mem), &d_kernel);
    err |= clSetKernelArg(clKernel, 5, sizeof(int), &kernelSize);
    CHECK_ERROR(err, "Failed to set kernel arguments");

    for (const auto& imageFile : imageFiles) {
        // load input into array
        loadImage(imageFile, inputImage, width, height);

        // write input into buffer
        err = clEnqueueWriteBuffer(queue, d_inputImage, CL_TRUE, 0, width * height * sizeof(float), inputImage, 0, NULL, NULL);
        CHECK_ERROR(err, "Failed to write input image to buffer");

        // incase image sizes are different
        err = clSetKernelArg(clKernel, 3, sizeof(int), &width);
        err |= clSetKernelArg(clKernel, 4, sizeof(int), &height);
        CHECK_ERROR(err, "Failed to set kernel arguments for image size");

        // launch config
        size_t globalSize[2] = {(size_t)width, (size_t)height};
        size_t localSize[2] = {BLOCKSIZE, BLOCKSIZE};

        // Start timing
        auto start = chrono::high_resolution_clock::now();
        err = clEnqueueNDRangeKernel(queue, clKernel, 2, NULL, globalSize, localSize, 0, NULL, NULL);
        CHECK_ERROR(err, "Failed to execute kernel");
        //end timing
        auto end = chrono::high_resolution_clock::now();
        chrono::duration<double> elapsed = end - start;
        total_elapsed += elapsed;

        // read output back to buffer
        clEnqueueReadBuffer(queue, d_outputImage, CL_TRUE, 0, width * height * sizeof(float), outputImage, 0, NULL, NULL);

        Mat outputMat(height, width, CV_32F, outputImage);
        outputMat *= 255.0;
        Mat outputMat8U;
        outputMat.convertTo(outputMat8U, CV_8U);

        fs::path inputPath(imageFile);
        string outputFilename = (fs::path(OUTPUT_PATH) / inputPath.filename()).string();
        imwrite(outputFilename, outputMat8U);
        fileCount++;
    }

    free(inputImage);
    free(outputImage);
    free(source);
    clReleaseMemObject(d_inputImage);
    clReleaseMemObject(d_outputImage);
    clReleaseMemObject(d_kernel);
    clReleaseKernel(clKernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    printf("Convolution completed using shared mem successfully!\n");
    cout << "Total Time to convert " << fileCount<<" images: " << total_elapsed.count() << " seconds\n";
    return 0;
}