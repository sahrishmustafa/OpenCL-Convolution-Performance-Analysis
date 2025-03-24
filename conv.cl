__kernel void apply_conv(
    __global const float* paddedImage,  
    __global float* outputImage,        
    __global const float* kernel_filter,       
    const int paddedWidth,              
    const int paddedHeight,             
    const int originalWidth,            
    const int originalHeight,           
    const int kernelSize)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int halfK = kernelSize / 2;

    if (x < originalWidth && y < originalHeight) {
        float sum = 0.0f;
        for (int i = -halfK; i <= halfK; i++) {
            for (int j = -halfK; j <= halfK; j++) {
                int imgIndex = (y + i + halfK) * paddedWidth + (x + j + halfK);
                int kernelIndex = (i + halfK) * kernelSize + (j + halfK);
                sum += paddedImage[imgIndex] * kernel_filter[kernelIndex];
            }
        }
        outputImage[y * originalWidth + x] = sum;
    }
}

__kernel void apply_conv_shared(
    __global const float* inputImage,  // Now, unpadded
    __global float* outputImage,
    __global const float* kernel_filter,
    const int imageWidth,
    const int imageHeight,
    const int kernelSize)
{
    // Work-group and thread indices
    int local_x = get_local_id(0);
    int local_y = get_local_id(1);
    int global_x = get_global_id(0);
    int global_y = get_global_id(1);
    int group_x = get_group_id(0);
    int group_y = get_group_id(1);

    // Define tile size: Adding padding (halo) around the shared memory
    int halfK = kernelSize / 2;
    int localSizeX = get_local_size(0);
    int localSizeY = get_local_size(1);
    int tileWidth = localSizeX + 2 * halfK;
    int tileHeight = localSizeY + 2 * halfK;

    // Allocate shared memory (local memory)
    __local float sharedTile[40][40]; // Assuming max local size is 32x32, padding included

    // Global memory location in the image
    int globalIndex = global_y * imageWidth + global_x;

    // Compute the position in the shared memory tile
    int shared_x = local_x + halfK;
    int shared_y = local_y + halfK;

    // Load the pixel into shared memory
    if (global_x < imageWidth && global_y < imageHeight) {
        sharedTile[shared_y][shared_x] = inputImage[globalIndex];
    } else {
        sharedTile[shared_y][shared_x] = 0.0f; // Zero-padding for out-of-bounds
    }

    // Load the halo (padding)
    if (local_x < halfK) {
        // Left halo
        int srcX = max(global_x - halfK, 0);
        sharedTile[shared_y][local_x] = inputImage[global_y * imageWidth + srcX];

        // Right halo
        int srcX2 = min(global_x + localSizeX, imageWidth - 1);
        sharedTile[shared_y][shared_x + localSizeX] = inputImage[global_y * imageWidth + srcX2];
    }

    if (local_y < halfK) {
        // Top halo
        int srcY = max(global_y - halfK, 0);
        sharedTile[local_y][shared_x] = inputImage[srcY * imageWidth + global_x];

        // Bottom halo
        int srcY2 = min(global_y + localSizeY, imageHeight - 1);
        sharedTile[shared_y + localSizeY][shared_x] = inputImage[srcY2 * imageWidth + global_x];
    }

    // Synchronize work-items to ensure all data is loaded
    barrier(CLK_LOCAL_MEM_FENCE);

    // Perform convolution only for valid output pixels
    if (global_x < imageWidth && global_y < imageHeight) {
        float sum = 0.0f;
        for (int ky = -halfK; ky <= halfK; ky++) {
            for (int kx = -halfK; kx <= halfK; kx++) {
                sum += sharedTile[shared_y + ky][shared_x + kx] * kernel_filter[(ky + halfK) * kernelSize + (kx + halfK)];
            }
        }
        outputImage[globalIndex] = sum;
    }
}
