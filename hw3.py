import cv2
import matplotlib.pyplot as plt
import os 
import numpy as np
from scipy.ndimage import convolve
from scipy.ndimage import median_filter

output_path = 'output_images'

if not os.path.exists(output_path):
    os.makedirs(output_path)

cameraman = cv2.imread('cameraman.tif', cv2.IMREAD_GRAYSCALE)
noisy_cameraman = cv2.imread('noisyCameraman.tif', cv2.IMREAD_GRAYSCALE)

def save_images(image, title, filename, output_path):
    plt.imshow(image, cmap='gray', vmin=0, vmax=255)
    plt.title(title)
    plt.axis('off')

    image_path = os.path.join(output_path, f'{filename}.png')
    plt.savefig(image_path)
    plt.close()

save_images(cameraman, 'Cameraman Original', 'cameraman_original', output_path)
save_images(noisy_cameraman, 'Noisy Cameraman Image', 'noisy_cameraman_image', output_path)

h3 = np.ones((3,3))/9
h4 = np.ones((4,4))/16
h5 = np.ones((5,5))/25

cameraman_filtered_h3 = convolve(cameraman,h3)
save_images(cameraman_filtered_h3, 'Cameraman Filtered H3', 'cameraman_filtered_h3', output_path)
cameraman_filtered_h4 = convolve(cameraman,h4)
save_images(cameraman_filtered_h4, 'Cameraman Filtered H4', 'cameraman_filtered_h4', output_path)
cameraman_filtered_h5 = convolve(cameraman,h5)
save_images(cameraman_filtered_h5, 'Cameraman Filtered H5', 'cameraman_filtered_h5', output_path)

noisy_filtered_h3 = convolve(noisy_cameraman,h3)
save_images(noisy_filtered_h3, 'Noisy Filtered H3', 'noisy_filtered_h3', output_path)
noisy_filtered_h4 = convolve(noisy_cameraman,h4)
save_images(noisy_filtered_h4, 'Noisy Filtered H4', 'noisy_filtered_h4', output_path)
noisy_filtered_h5 = convolve(noisy_cameraman,h5)
save_images(noisy_filtered_h5, 'Noisy Filtered H5', 'noisy_filtered_h5', output_path)

median_cameraman = median_filter(cameraman, size=3)
save_images(median_cameraman, 'Cameraman Median Filtered', 'cameraman_median_filtered', output_path)
median_noisy = median_filter(noisy_cameraman, size=3)
save_images(median_noisy, 'Noisy Median Filtered', 'noisy_median_filtered', output_path)

l4 = np.array([[0,1,0], [1,-4,1], [0,1,0]])
l8 = np.array([[0,1,0], [1,-8,1], [0,1,0]])
sobel_v = np.array([[-1,-2,-1], [0,0,0], [1,2,1]])
sobel_h = np.array([[-1,0,1], [-2,0,2], [-1,0,-1]])

cameraman_filtered_l4 = convolve(cameraman, l4)
save_images(cameraman_filtered_l4, 'Cameraman Filtered L4', 'cameraman_filtered_l4', output_path)
cameraman_filtered_l8 = convolve(cameraman, l8)
save_images(cameraman_filtered_l8, 'Cameraman Filtered L8', 'cameraman_filtered_l8', output_path)
cameraman_filtered_sobelV = convolve(cameraman, sobel_v)
save_images(cameraman_filtered_sobelV, 'Cameraman Filtered Sobel-V', 'cameraman_filtered_sobelV', output_path)
cameraman_filtered_sobelH = convolve(cameraman, sobel_h)
save_images(cameraman_filtered_sobelH, 'Cameraman Filtered Sobel-H', 'cameraman_filtered_sobelH', output_path)

noisy_filtered_l4 = convolve(noisy_cameraman, l4)
save_images(noisy_filtered_l4, 'Noisy Filtered L4', 'noisy_filtered_l4', output_path)
noisy_filtered_l8 = convolve(noisy_cameraman, l8)
save_images(noisy_filtered_l8, 'Noisy Filtered L8', 'noisy_filtered_l8', output_path)
noisy_filtered_sobelV = convolve(noisy_cameraman, sobel_v)
save_images(noisy_filtered_sobelV, 'Noisy Filtered Sobel-V', 'noisy_filtered_sobelV', output_path)
noisy_filtered_sobelH = convolve(noisy_cameraman, sobel_h)
save_images(noisy_filtered_sobelH, 'Noisy Filtered Sobel-H', 'noisy_filtered_sobelH', output_path)

def add_original_and_filtered(original_image, filtered_image, alpha=1, beta=1):
    # Image mapping technique in showed lesson
    image = alpha*original_image + beta*filtered_image
    normalized_image = (image - image.min()) / (image.max() - image.min()) * 255
    return normalized_image.astype(np.uint8)

cameraman_sobelV_edge_enhanced = add_original_and_filtered(cameraman, cameraman_filtered_sobelV)
save_images(cameraman_sobelV_edge_enhanced, 'Cameraman Sobel-V Edge Enhanced', 'cameraman_sobelV_edge_enhanced', output_path)

# Sobel Horizontal Edge Enhanced
cameraman_sobelH_edge_enhanced = add_original_and_filtered(cameraman, cameraman_filtered_sobelH)
save_images(cameraman_sobelH_edge_enhanced, 'Cameraman Sobel-H Edge Enhanced', 'cameraman_sobelH_edge_enhanced', output_path)

# Laplacian (L4) Edge Enhanced
cameraman_l4_edge_enhanced = add_original_and_filtered(cameraman, cameraman_filtered_l4)
save_images(cameraman_l4_edge_enhanced, 'Cameraman L4 Edge Enhanced', 'cameraman_l4_edge_enhanced', output_path)

# Laplacian (L8) Edge Enhanced
cameraman_l8_edge_enhanced = add_original_and_filtered(cameraman, cameraman_filtered_l8)
save_images(cameraman_l8_edge_enhanced, 'Cameraman L8 Edge Enhanced', 'cameraman_l8_edge_enhanced', output_path)

# Similarly, do it for the noisy image
noisy_sobelV_edge_enhanced = add_original_and_filtered(noisy_cameraman, noisy_filtered_sobelV)
save_images(noisy_sobelV_edge_enhanced, 'Noisy Sobel-V Edge Enhanced', 'noisy_sobelV_edge_enhanced', output_path)

noisy_sobelH_edge_enhanced = add_original_and_filtered(noisy_cameraman, noisy_filtered_sobelH)
save_images(noisy_sobelH_edge_enhanced, 'Noisy Sobel-H Edge Enhanced', 'noisy_sobelH_edge_enhanced', output_path)

noisy_l4_edge_enhanced = add_original_and_filtered(noisy_cameraman, noisy_filtered_l4)
save_images(noisy_l4_edge_enhanced, 'Noisy L4 Edge Enhanced', 'noisy_l4_edge_enhanced', output_path)

noisy_l8_edge_enhanced = add_original_and_filtered(noisy_cameraman, noisy_filtered_l8)
save_images(noisy_l8_edge_enhanced, 'Noisy L8 Edge Enhanced', 'noisy_l8_edge_enhanced', output_path)
