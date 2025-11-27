# import h5py
# import numpy as np
# from pygrappa import grappa

# # Load FastMRI data
# with h5py.File('./data/fastMRI/multicoil_test/file1000082.h5', 'r') as f:
#     # Get k-space data (coils, height, width)
#     kspace = f['kspace'][()]
    
# # Convert to format pyGRAPPA expects: (height, width, coils)
# kspace = np.transpose(kspace, (1, 2, 0))

# # Create undersampled k-space (example: keep every 2nd line)
# # In practice, FastMRI data is already undersampled
# calib = kspace[::2, :, :]  # Calibration region (ACS lines)
# undersamp = kspace.copy()
# undersamp[1::2, :, :] = 0  # Zero out every other line

# # Apply GRAPPA reconstruction
# recon = grappa(undersamp, calib, kernel_size=(5, 5), coil_axis=-1)

# # Combine coils (simple root sum of squares)
# img = np.sqrt(np.sum(np.abs(np.fft.ifft2(recon, axes=(0, 1)))**2, axis=-1))

# print(f"Original k-space shape: {kspace.shape}")
# print(f"Reconstructed image shape: {img.shape}")


import h5py
import numpy as np
from pygrappa import grappa, grappaop
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr, mean_squared_error as mse

# Load FastMRI data
with h5py.File('./data/fastMRI/multicoil_test/file1000136.h5', 'r') as f:
    print(f"Available keys in file: {list(f.keys())}")
    kspace_all = f['kspace'][()]
    mask = f['mask'][()] if 'mask' in f else None
    print(f"Loaded k-space shape: {kspace_all.shape}")
    if mask is not None:
        print(f"Loaded mask shape: {mask.shape}")

# Choose slice to reconstruct with GRAPPA
slice_idx = 18

# Get single slice: (coils, height, width)
kspace_orig = kspace_all[slice_idx]  # Shape: (15, 640, 368)

# Convert to GRAPPA format: (height, width, coils)
kspace = np.transpose(kspace_orig, (1, 2, 0))

if mask is not None:
    # Use the FastMRI mask to identify which lines are sampled
    if len(mask.shape) == 1:
        current_mask = mask
    else:
        current_mask = mask[slice_idx] if mask.shape[0] == kspace_all.shape[0] else mask[0]
    
    print(f"Mask shape: {current_mask.shape}")
    print(f"Number of sampled lines: {np.sum(current_mask > 0)}/{len(current_mask)}")
    
    # Find the densely sampled center region for calibration (ACS)
    sampled_indices = np.where(current_mask > 0)[0]
    
    # Use the center portion as calibration
    center = len(current_mask) // 2
    acs_width = 32
    calib = kspace[center-acs_width//2:center+acs_width//2, :, :]
    
    # The undersampled data is already in kspace (FastMRI is undersampled)
    undersamp = kspace.copy()
    
    print(f"Calibration region size: {calib.shape}")
    print(f"Applying GRAPPA reconstruction to FastMRI undersampled data...")
    
    # Apply GRAPPA
    recon = grappa(undersamp, calib, kernel_size=(5, 5), coil_axis=-1, lamda=0.01)
    
else:
    print("No mask found - cannot identify undersampling pattern")
    print("Showing original image only")
    recon = kspace.copy()

# Convert to image space
img_orig_coils = np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(kspace, axes=(0, 1)), axes=(0, 1)), axes=(0, 1))
img_recon_coils = np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(recon, axes=(0, 1)), axes=(0, 1)), axes=(0, 1))

# Combine coils (RSS)
img_orig = np.sqrt(np.sum(np.abs(img_orig_coils)**2, axis=-1))
img_recon = np.sqrt(np.sum(np.abs(img_recon_coils)**2, axis=-1))

# Create zero-filled image for comparison
zerofilled_coils = np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(undersamp, axes=(0, 1)), axes=(0, 1)), axes=(0, 1))
img_zerofilled = np.sqrt(np.sum(np.abs(zerofilled_coils)**2, axis=-1))

print(f"GRAPPA reconstruction complete!")

# Normalize images to [0, 1] for metrics
img_orig_norm = (img_orig - img_orig.min()) / (img_orig.max() - img_orig.min())
img_recon_norm = (img_recon - img_recon.min()) / (img_recon.max() - img_recon.min())
img_zerofilled_norm = (img_zerofilled - img_zerofilled.min()) / (img_zerofilled.max() - img_zerofilled.min())

# Calculate metrics (using original as reference)
print("\n=== Image Quality Metrics ===")
print("Zero-Filled vs Original:")
print(f"  SSIM: {ssim(img_orig_norm, img_zerofilled_norm, data_range=1.0):.4f}")
print(f"  PSNR: {psnr(img_orig_norm, img_zerofilled_norm, data_range=1.0):.2f} dB")
print(f"  NMSE: {mse(img_orig_norm, img_zerofilled_norm):.6f}")

print("\nGRAPPA vs Original:")
print(f"  SSIM: {ssim(img_orig_norm, img_recon_norm, data_range=1.0):.4f}")
print(f"  PSNR: {psnr(img_orig_norm, img_recon_norm, data_range=1.0):.2f} dB")
print(f"  NMSE: {mse(img_orig_norm, img_recon_norm):.6f}")

# Visualize comparison
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

axes[0].imshow(img_zerofilled, cmap='gray')
axes[0].set_title(f'Zero-Filled (Aliasing) - Slice {slice_idx}', fontsize=14)
axes[0].axis('off')

axes[1].imshow(img_orig, cmap='gray')
axes[1].set_title(f'Original (Zero-Filled) - Slice {slice_idx}', fontsize=14)
axes[1].axis('off')

axes[2].imshow(img_recon, cmap='gray')
axes[2].set_title(f'GRAPPA Reconstructed - Slice {slice_idx}', fontsize=14)
axes[2].axis('off')

plt.tight_layout()
plt.show()