import matplotlib.image as mpimg
import time
import numpy as np
from numba import cuda
import matplotlib.pyplot as plt
# Ref
# 1. https://christianjmills.com/posts/cuda-mode-notes/lecture-003/
# 2. 12201577 Stack overflow


# Read the image into a NumPy array
img = mpimg.imread('original_image.jpg')

# Display the image
plt.imshow(img)
plt.show()
print("Array shape (Height, Width, Channels):", img.shape)

# Iterate through every pixel and apply the gray scale formula.
def rgb2gray_cpu(img):
    start = time.time()
    h, w, c = img.shape
    num_of_pixel = h * w
    x = img.flatten() # Flatten the image. It will be [R00,G00,B00, R01,G01,B01, ...]
    gray_img_flat = np.zeros(num_of_pixel, dtype=img.dtype) # Initialize as a 1D array of correct size

    # Convert RGB to grayscale using weighted sum
    for pixel_idx in range(num_of_pixel):
        start_idx = pixel_idx * c
        R = x[start_idx]
        G = x[start_idx + 1]
        B = x[start_idx + 2]
        gray_img_flat[pixel_idx] = 0.2989 * R + 0.5870 * G + 0.1140 * B

    end = time.time()
    print(f"CPU convert to gray processing Time: {end-start} s", )
    return gray_img_flat.reshape((h, w)) # Reshape the 1D array back to a 2D image

gray_img = rgb2gray_cpu(img)
plt.imshow(gray_img, cmap="gray")
plt.axis("off")
plt.show()


# CUDA
# 1. Define the CUDA Kernel
@cuda.jit
def rgb_to_grayscale_kernel(src, dst, height, width):
    x, y = cuda.grid(2)

    if x < width and y < height:
        pixel = y * width + x
        idx = pixel * 3

        r = src[idx]
        g = src[idx + 1]
        b = src[idx + 2]

        dst[pixel] = 0.299 * r + 0.587 * g + 0.114 * b


def rgb2gray_gpu(img):
    start = time.time()
    height, width, channels = img.shape
    num_pixels = height * width

    # Flatten RGB image: [R00,G00,B00, R01,G01,B01, ...]
    src_host = img.reshape(-1)

    # Output: one grayscale value per pixel
    dst_host = np.zeros(num_pixels, dtype=img.dtype)

    # Transfer data to GPU
    d_src = cuda.to_device(src_host)
    d_dst = cuda.to_device(dst_host)

    # 2D CUDA configuration
    threads_per_block = (16, 16)

    blocks_per_grid = (
        (width + threads_per_block[0] - 1) // threads_per_block[0],
        (height + threads_per_block[1] - 1) // threads_per_block[1]
    )

    # Launch kernel
    rgb_to_grayscale_kernel[blocks_per_grid, threads_per_block](
        d_src,
        d_dst,
        height,
        width
    )

    # Wait for GPU
    cuda.synchronize()

    # Copy result back to CPU
    dst_host = d_dst.copy_to_host()

    end = time.time()
    print(f"GPU grayscale processing time: {end - start:.6f} s")

    return src_host, dst_host


# Run the GPU processing pipeline
original, grayscale = rgb2gray_gpu(img)

# Get original image dimensions
height, width, _ = img.shape

# Reshape grayscale output back to 2D
grayscale = grayscale.reshape(height, width)

# Display the results
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

axes[0].imshow(img)
axes[0].set_title("Original Image")
axes[0].axis('off')

axes[1].imshow(grayscale, cmap='gray')
axes[1].set_title("GPU Grayscale")
axes[1].axis('off')

plt.tight_layout()
plt.show()