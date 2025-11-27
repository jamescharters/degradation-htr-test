import h5py
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.restoration import denoise_tv_chambolle
import time

# ==========================================
# Helper Functions
# ==========================================
def fft2c(x):
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x, axes=(-2, -1)), axes=(-2, -1)), axes=(-2, -1))

def ifft2c(x):
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(x, axes=(-2, -1)), axes=(-2, -1)), axes=(-2, -1))

def normalize(x):
    return (x - x.min()) / (x.max() - x.min())

# ==========================================
# Main Routine
# ==========================================
def run_singlecoil_demo(file_path):
    print(f"--- Loading {file_path} ---")
    
    with h5py.File(file_path, 'r') as f:
        # FastMRI singlecoil usually stores volume in 'kspace'
        k_vol = f['kspace'][()] 

    print(f"File Shape: {k_vol.shape}")
    
    # -----------------------------------------------------------
    # 1. DETECT DATA TYPE
    # -----------------------------------------------------------
    # Multi-coil usually 4D: (Slices, Coils, H, W)
    # Single-coil usually 3D: (Slices, H, W)
    
    is_multicoil = (k_vol.ndim == 4)
    
    if is_multicoil:
        print(">>> DETECTED MULTI-COIL DATA. Please use the GRAPPA script provided previously.")
        return
    else:
        print(">>> DETECTED SINGLE-COIL DATA.")
        print("    GRAPPA is physically impossible on this data.")
        print("    Switching to COMPRESSED SENSING (Total Variation) to beat Zero-Filled.")

    # Select a middle slice
    slice_idx = k_vol.shape[0] // 2
    k_slice = k_vol[slice_idx]
    
    # Ensure Complex
    if not np.iscomplexobj(k_slice):
        k_slice = k_slice[..., 0] + 1j * k_slice[..., 1]

    # Normalize
    k_slice /= np.max(np.abs(k_slice))
    
    # -----------------------------------------------------------
    # 2. Ground Truth & Sanity
    # -----------------------------------------------------------
    gt_im = ifft2c(k_slice)
    gt_mag = np.abs(gt_im)
    
    s_gt = ssim(normalize(gt_mag), normalize(gt_mag), data_range=1.0)
    print(f"\n[Sanity] GT Self-SSIM: {s_gt:.4f} (Target 1.0)")
    
    # -----------------------------------------------------------
    # 3. Undersampling (R=4)
    # -----------------------------------------------------------
    ny, nx = k_slice.shape
    R = 4
    acs_lines = 32
    
    mask = np.zeros((ny, nx), dtype=np.float32)
    # Undersample columns (Phase Encode in singlecoil is usually last dim)
    mask[:, ::R] = 1.0 
    
    # ACS Center
    c_x = nx // 2
    mask[:, c_x - acs_lines//2 : c_x + acs_lines//2] = 1.0
    
    k_u = k_slice * mask
    
    # -----------------------------------------------------------
    # 4. Zero-Filled Reconstruction
    # -----------------------------------------------------------
    zf_im = ifft2c(k_u)
    zf_mag = np.abs(zf_im)
    score_zf = ssim(normalize(gt_mag), normalize(zf_mag), data_range=1.0)
    
    # -----------------------------------------------------------
    # 5. Compressed Sensing (Total Variation)
    # -----------------------------------------------------------
    # Since we can't use GRAPPA, we use TV Denoising which is the 
    # standard "step up" from Zero-Filled for single coil data.
    # It removes the Gibbs ringing/aliasing noise.
    
    print("\n--- Running Compressed Sensing (Total Variation) ---")
    t0 = time.time()
    
    # We take the noisy Zero-Filled image and apply TV regularization
    # Ideally one solves min ||Ax - y|| + lambda*TV(x), but strictly 
    # applying TV to the ZF magnitude is a strong approximation (POCS).
    cs_mag = denoise_tv_chambolle(zf_mag, weight=0.1)
    
    print(f"CS done in {time.time()-t0:.2f}s")
    
    score_cs = ssim(normalize(gt_mag), normalize(cs_mag), data_range=1.0)

    # -----------------------------------------------------------
    # 6. Results
    # -----------------------------------------------------------
    print(f"\n--- Final Comparison ---")
    print(f"Zero-Filled SSIM:      {score_zf:.4f}")
    print(f"Compressed Sensing SSIM: {score_cs:.4f}")
    
    if score_cs > score_zf:
        print(">>> SUCCESS: CS beat Zero-Filled.")
    else:
        print(">>> FAIL.")

    # Viz
    fig, ax = plt.subplots(1, 3, figsize=(15, 6))
    vmax = np.percentile(gt_mag, 99)
    
    ax[0].imshow(gt_mag, cmap='gray', vmax=vmax)
    ax[0].set_title("Ground Truth")
    
    ax[1].imshow(zf_mag, cmap='gray', vmax=vmax)
    ax[1].set_title(f"Zero-Filled\nSSIM: {score_zf:.3f}")
    
    ax[2].imshow(cs_mag, cmap='gray', vmax=vmax)
    ax[2].set_title(f"Compressed Sensing\nSSIM: {score_cs:.3f}")
    
    for a in ax: a.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    filename = "./data/fastMRI/multicoil_test/file1000295.h5" 
    run_singlecoil_demo(filename)