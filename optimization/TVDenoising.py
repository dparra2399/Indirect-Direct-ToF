import numpy as np
from matplotlib.image import imread
import cvxpy as cp
import matplotlib.pyplot as plt
from scipy.linalg import convolution_matrix
from scipy import ndimage

def create_convolution_matrix(image_shape, kernel):
    image_height, image_width = image_shape
    kernel_height, kernel_width = kernel.shape
    output_height = image_height
    output_width = image_width
    convolution_matrix = np.zeros((output_height * output_width,
                                   image_height * image_width))

    for i in range(output_height):
        for j in range(output_width):
            output_index = i * output_width + j
            for m in range(kernel_height):
                for n in range(kernel_width):
                    input_i = i + m - kernel_height // 2
                    input_j = j + n - kernel_width // 2
                    if 0 <= input_i < image_height and 0 <= input_j < image_width:
                        input_index = input_i * image_width + input_j
                        convolution_matrix[output_index, input_index] = kernel[m, n]

    return convolution_matrix



def deblur(img):


    # dftmtx_n = np.fft.fft2(np.eye(N=img.shape[0]))
    # dftmtx_m = np.fft.fft2(np.eye(N=img.shape[1]))
    #
    # dftimtx_n = np.fft.ifft2(np.eye(N=img.shape[0]))
    # dftimtx_m = np.fft.ifft2(np.eye(N=img.shape[1]))


    # img_fft = dftmtx_n @ img_flat
    kernel = np.ones((3, 3)) / 9.0
    conv_matrix = create_convolution_matrix(img.shape, kernel)
    img_flat = conv_matrix @ img.flatten()

    # kernel = gaussian_kernel(3, sigma=0.1)
    # kernel_fft = np.fft.fft2(kernel, s=img.shape)
    #
    # img_fft = dftmtx_n @ img @ dftmtx_m
    # #img_blur = np.fft.ifft2(np.fft.fft2(img, s=img.shape)*kernel_fft, s=img.shape)
    # img_blur = dftimtx_n @ np.multiply(img_fft, kernel_fft) @ dftimtx_m
    # img_blur = img_blur.real
    U = cp.Variable(shape=img.size)
    lam = 0.1


    objective = cp.Minimize(0.5 * cp.sum_squares(conv_matrix @ U
                                                 - img_flat) + lam * cp.tv(U))
    # constraints = [
    #     cp.multiply()
    # ]
    prob = cp.Problem(objective)
    prob.solve(verbose=True, solver=cp.SCS)

    img_denoised = U.value
    return img_denoised

def main():
    img = imread(r'C:\\Users\\Patron\\Documents\\MATLAB\\Black_and_white_zebra.jpeg')
    img_deblurred = np.zeros_like(img)
    x_patch = 393
    y_patch = 64
    for i in range(0, img.shape[0], x_patch):
        for j in range(0, img.shape[1], y_patch):
            patch = img[i:i + x_patch, j:j + y_patch]
            if patch.shape[0] == x_patch and patch.shape[1] == y_patch:
                img_deblurred[i:i + x_patch, j:j + y_patch] = np.reshape(deblur(patch),
                                                                         (x_patch, y_patch))

    img_median = ndimage.median_filter(img_deblurred, size=3)
    fig, axs = plt.subplots()
    axs.imshow(img_median, cmap='gray')
    fig.savefig('hw2-722_zebra.png', dpi=1000)
    plt.show()
    return 0

main()
