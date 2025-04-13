import cv2
import numpy as np
from scipy.signal import convolve2d
from skimage.metrics import structural_similarity as ssim

def load_image_grayscale(filename):
    """Load an image as grayscale and normalize it to [0,1]."""
    image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Error: Could not load image {filename}")
    return image.astype(np.float32) / 255.0  # Normalize

def compare_convolution_results(custom_conv_path, original_image_path, kernel):
    """Compare stored custom convolution result with OpenCV and SciPy results."""
    
    # Load original image
    image = load_image_grayscale(original_image_path)

    # Load custom convolution result from stored PNG
    custom_result = load_image_grayscale(custom_conv_path)

    # OpenCV Convolution
    opencv_result = cv2.filter2D(image, -1, kernel)

    # SciPy Convolution
    scipy_result = convolve2d(image, kernel, mode='same', boundary='fill', fillvalue=0)

    # Compute Mean Squared Error (MSE)
    mse_opencv = np.mean((custom_result - opencv_result) ** 2)
    mse_scipy = np.mean((custom_result - scipy_result) ** 2)

    print(f"Comparison with OpenCV:")
    print(f" - MSE: {mse_opencv:.6f}")


    print(f"Comparison with SciPy:")
    print(f" - MSE: {mse_scipy:.6f}")


    return custom_result, opencv_result, scipy_result


if __name__ == "__main__":
    original_image_path = "flickr_cat_000003.png"  # original image path
    custom_conv_path = "flickr_cat_000003_output.png"  #  stored convolution result path

    # Example 3x3 kernel (Gaussian Blur or Edge Detection)
    kernel = np.array([[1, 0, -1],
                       [1, 0, -1],
                       [1, 0, -1]], dtype=np.float32) 
    custom_conv, opencv_conv, scipy_conv = compare_convolution_results(custom_conv_path, original_image_path, kernel)
 
