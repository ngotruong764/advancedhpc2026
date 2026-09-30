import matplotlib.image as mpimg
import time
import numpy as np
from numba import cuda
import matplotlib.pyplot as plt


# Read the image into a NumPy array
img = mpimg.imread('original_image.jpg')
# Iterate through every pixel and apply the gray scale formula.
def rgb2gray_cpu(img):
    start = time.time()
    h, w, c = img.shape

    # Output grayscale image
    gray_img = np.zeros((h, w), dtype=img.dtype)

    for y in range(h):
        for x in range(w):
            R = img[y, x, 0] / 3.0
            G = img[y, x, 1] / 3.0
            B = img[y, x, 2] / 3.0
            gray_img[y, x] = R + G + B

    end = time.time()
    print(f"CPU convert to gray processing Time: {end - start} s")

    return gray_img

gray_img = rgb2gray_cpu(img)
plt.imshow(gray_img, cmap="gray")
plt.axis("off")
plt.show()


# CUDA
# 1. Define the CUDA Kernel
@cuda.jit
def rgb2gray_kernel(src, dst):
    y, x = cuda.grid(2)

    h = src.shape[0]
    w = src.shape[1]

    if y < h and x < w:
        R = src[y, x, 0] / 3.0
        G = src[y, x, 1] / 3.0
        B = src[y, x, 2] / 3.0

        dst[y, x] = R + G + B


def rgb2gray_gpu(img):
    start = time.time()
    h, w, c = img.shape

    dst_host = np.zeros((h, w), dtype=img.dtype)

    # Transfer data to GPU
    d_src = cuda.to_device(img)
    d_dst = cuda.to_device(dst_host)

    # 2D CUDA configuration
    threads_per_block = (32, 32)
    blocks_per_grid = (
        (h + threads_per_block[0] - 1) // threads_per_block[0],
        (w + threads_per_block[1] - 1) // threads_per_block[1]
    )
    print(blocks_per_grid)
    # Launch kernel
    rgb2gray_kernel[blocks_per_grid, threads_per_block](d_src,d_dst)

    # Wait for GPU
    cuda.synchronize()

    # Copy result back to CPU
    dst_host = d_dst.copy_to_host()

    end = time.time()
    print(f"GPU grayscale processing time: {end - start} s")

    return dst_host


# Run the GPU processing pipeline
grayscale = rgb2gray_gpu(img)
plt.imshow(grayscale, cmap="gray")
plt.axis("off")
plt.show()


# Run the GPU processing pipeline
grayscale = rgb2gray_gpu(img)
plt.imshow(grayscale, cmap="gray")
plt.axis("off")
plt.show()