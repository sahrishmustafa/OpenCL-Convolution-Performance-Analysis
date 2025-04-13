#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <filesystem>  

namespace fs = std::filesystem; 
using namespace std;
using namespace cv;

#define OUTPUT_PATH "output_serial"
#define INPUT_PATH "dataset/grayscale/"

// vertical edges
vector<vector<int>> Kernel = {
    {1,  0, -1},
    {1,  0, -1},
    {1,  0, -1}
};

// Function to apply 2D convolution with padding
Mat apply_Conv(const Mat& image, const vector<vector<int>>& kernel) {
    int kSize = kernel.size();
    int pad = kSize / 2;
    
    // Apply zero-padding
    Mat paddedImage;
    copyMakeBorder(image, paddedImage, pad, pad, pad, pad, BORDER_CONSTANT, Scalar(0));
    
    Mat output = Mat::zeros(image.size(), CV_32F);

    // Iterate over each pixel in the original image region
    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            float sum = 0.0;
            // Convolve with kernel
            for (int ki = -pad; ki <= pad; ki++) {
                for (int kj = -pad; kj <= pad; kj++) {
                    sum += kernel[ki + pad][kj + pad] * paddedImage.at<uchar>(i + ki + pad, j + kj + pad);
                }
            }
            output.at<float>(i, j) = sum;
        }
    }

    return output;
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

    string inputFolder = INPUT_PATH + resolution;
    string outputFolder = OUTPUT_PATH;

    if (!fs::exists(outputFolder)) {
        fs::create_directory(outputFolder);
    }

    int fileCount = 0; 
    chrono::duration<double> total_elapsed;

    // Iterate all input files
    for (const auto& entry : fs::directory_iterator(inputFolder)) {
        string inputPath = entry.path().string();

        if (entry.path().extension() == ".png" || entry.path().extension() == ".jpg") {
            Mat image = imread(inputPath, IMREAD_GRAYSCALE);
            if (image.empty()) {
                cerr << "Error: Could not read " << inputPath << endl;
                continue; // Skip
            }

            // Start timing
            auto start = chrono::high_resolution_clock::now();
            Mat result = apply_Conv(image, Kernel);
            // End timing
            auto end = chrono::high_resolution_clock::now();
            chrono::duration<double> elapsed = end - start;
            total_elapsed += elapsed;

            string outputPath = outputFolder + "/" + entry.path().filename().string();
            // Save result (convert to 8-bit before saving)
            Mat output_8bit;
            result.convertTo(output_8bit, CV_8U);
            imwrite(outputPath, output_8bit);

            fileCount++;
        }
    }

    cout << "Total Time to convert " << fileCount << " images: " << total_elapsed.count() << " seconds\n";

    if (fileCount == 0) {
        cout << "No PNG images found in " << inputFolder << endl;
    } else {
        cout << "Processed " << fileCount << " images successfully.\n";
    }

    return 0;
}
