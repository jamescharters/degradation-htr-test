import h5py
import matplotlib.pyplot as plt
import numpy as np

# Open the h5 file
with h5py.File('./data/fastMRI/multicoil_test/file1000136.h5', 'r') as f:
    # Get the k-space data (shape: slices, coils, height, width)
    kspace = f['kspace'][:]
    
    # Take the middle slice
    mid_slice = kspace.shape[0] // 2
    kspace_slice = kspace[mid_slice]  # shape: (coils, height, width)
    
    # Reconstruct each coil image using inverse FFT
    coil_images = np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(kspace_slice, axes=(-2, -1))), axes=(-2, -1))
    
    # Combine coils using root sum of squares
    combined_image = np.sqrt(np.sum(np.abs(coil_images) ** 2, axis=0))
    
    # Display results
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(abs(kspace_slice[0]), cmap='gray')
    plt.title('K-space (coil 0)')
    plt.colorbar()
    
    plt.subplot(1, 3, 2)
    plt.imshow(abs(coil_images[0]), cmap='gray')
    plt.title('Single Coil Image')
    plt.colorbar()
    
    plt.subplot(1, 3, 3)
    plt.imshow(combined_image, cmap='gray')
    plt.title('Combined Image (RSS)')
    plt.colorbar()
    
    plt.tight_layout()
    plt.show()