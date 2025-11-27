import h5py
import numpy as np
from pygrappa import grappa
import matplotlib.pyplot as plt

file_name = './data/fastMRI/multicoil_test/file1000082.h5'
hf = h5py.File(file_name)

volume_kspace = hf['kspace'][()]

middle_index = volume_kspace.shape[0] //2
slice_kspace = volume_kspace[middle_index] # Choosing the middle slice of this volume

def show_coils(data, slice_nums, cmap=None):
    fig = plt.figure()
    for i, num in enumerate(slice_nums):
        plt.subplot(1, len(slice_nums), i + 1)
        plt.imshow(data[num], cmap=cmap)
        plt.show()

show_coils(np.log(np.abs(slice_kspace) + 1e-9), [0, 5, 10])  # This shows coils 0, 5 and 10
# Note that a small constant is added for numerical stability.

# def inverse_fft2_shift(kspace):
#     return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(kspace, axes=(-2,-1)), norm='ortho'),axes=(-2,-1))

# show_coils(np.abs(inverse_fft2_shift(slice_kspace)), [0, 5, 10], cmap='gray')  # This shows coils 0, 5 and 10

# image_data = inverse_fft2_shift(slice_kspace)

# # Extract the magnitude of the image data.
# magnitude_data = np.abs(image_data)

# # Extract the phase of the image data.
# phase_data = np.angle(image_data)

# # Separate out the real component of the image data.
# real_data = np.real(image_data)

# # Separate out the imaginary component of the image data.
# imaginary_data = np.imag(image_data)

# # Combine multi-coil data using Root Sum of Squares (RSS) to create magnitude image
# # magnitude_slice = np.sqrt(np.sum(np.abs(image_space_slice)**2, axis=0))