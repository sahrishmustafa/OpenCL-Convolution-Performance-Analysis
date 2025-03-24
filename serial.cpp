#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <filesystem>  

namespace fs = std::filesystem; 
using namespace std;
using namespace cv;

#define INPUT_PATH "dataset/grayscale/512"
//#define INPUT_PATH = "dataset/grayscale/1024"

// Edge detection kernel (vertical edges)
vector<vector<int>> edgeKernel = {
    {1,  0, -1},
    {1,  0, -1},
    {1,  0, -1}
};

// Function to apply 2D convolution
Mat applyConvolution(const Mat& image, const vector<vector<int>>& kernel) {
    int kSize = kernel.size();
    int pad = kSize / 2;
    Mat output = Mat::zeros(image.size(), CV_32F);

    // Iterate over each pixel (excluding padding)
    for (int i = pad; i < image.rows - pad; i++) {
        for (int j = pad; j < image.cols - pad; j++) {
            float sum = 0.0;
            // Convolve with kernel
            for (int ki = -pad; ki <= pad; ki++) {
                for (int kj = -pad; kj <= pad; kj++) {
                    sum += kernel[ki + pad][kj + pad] * image.at<uchar>(i + ki, j + kj);
                }
            }
            // Store result
            output.at<float>(i, j) = sum;
        }
    }

    // Normalize and convert back to 8-bit
    //normalize(output, output, 0, 255, NORM_MINMAX);
    //output.convertTo(output, CV_8U);

    return output;
}

int main() {
    string inputFolder = INPUT_PATH;  // Folder containing input images
    string outputFolder = "output_serial";  // Folder to save processed images

    // Create output directory if it doesn't exist
    if (!fs::exists(outputFolder)) {
        fs::create_directory(outputFolder);
    }

    int fileCount = 0; // Count how many images were processed
    chrono::duration<double>  total_elapsed;

    // Iterate over all files in the input folder
    for (const auto& entry : fs::directory_iterator(inputFolder)) {
        string inputPath = entry.path().string();

        // Check if the file is a PNG image
        if (entry.path().extension() == ".png") {
            Mat image = imread(inputPath, IMREAD_GRAYSCALE);
            if (image.empty()) {
                cerr << "Error: Could not read " << inputPath << endl;
                continue; // Skip this file and move to the next
            }

            // Start timing
            auto start = chrono::high_resolution_clock::now();

            // Apply convolution
            Mat result = applyConvolution(image, edgeKernel);

            // End timing
            auto end = chrono::high_resolution_clock::now();
            chrono::duration<double> elapsed = end - start;
            total_elapsed += elapsed;

            // Generate output file name
            string outputPath = outputFolder + "/" + entry.path().filename().string();

            // Save result
            imwrite(outputPath, result);

            fileCount++;
        }
    }

    cout << "Total Time to extract " << fileCount <<" images: " << total_elapsed.count() << " seconds\n";

    if (fileCount == 0) {
        cout << "No PNG images found in " << inputFolder << endl;
    } else {
        cout << "Processed " << fileCount << " images successfully.\n";
    }

    return 0;
}
