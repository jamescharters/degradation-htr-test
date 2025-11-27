"""
Quick diagnostic for FastMRI knee dataset
Run this to understand your data before training
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt

FILE_PATH = "./data/fastMRI/multicoil_test/file1000136.h5"

def fastmri_to_complex(tensor):
    if tensor.shape[-1] == 2:
        return tensor[..., 0] + 1j * tensor[..., 1]
    return tensor

def diagnose_knee_data():
    print("="*80)
    print("FASTMRI KNEE DATASET DIAGNOSTIC")
    print("="*80)
    
    with h5py.File(FILE_PATH, "r") as hf:
        print(f"\n1. FILE CONTENTS:")
        print(f"   Keys: {list(hf.keys())}")
        for key in hf.keys():
            if hasattr(hf[key], 'shape'):
                print(f"   {key}: shape {hf[key].shape}, dtype {hf[key].dtype}")
        
        # Check attributes
        print(f"\n2. METADATA:")
        for key in hf.attrs.keys():
            print(f"   {key}: {hf.attrs[key]}")
        
        # Analyze multiple slices
        print(f"\n3. SLICE ANALYSIS:")
        print(f"   {'Slice':<6} | {'K-max':<12} | {'IMG-max':<12} | {'IMG-mean':<12} | {'Quality':<10}")
        print("   " + "-"*65)
        
        kspace_dataset = hf['kspace']
        num_slices = min(kspace_dataset.shape[0], 35)  # Check up to 35 slices
        
        best_slices = []
        
        for slice_idx in range(num_slices):
            kspace_raw = kspace_dataset[slice_idx]
            kspace = fastmri_to_complex(kspace_raw)
            
            # Simple RSS reconstruction
            img_coils = np.fft.ifft2(kspace, norm='ortho')
            img_rss = np.sqrt(np.sum(np.abs(img_coils)**2, axis=0))
            
            k_max = np.abs(kspace).max()
            img_max = img_rss.max()
            img_mean = img_rss.mean()
            
            # Quality score
            quality = "Good" if img_max > 0.02 and img_mean > 0.001 else "Poor"
            
            print(f"   {slice_idx:<6} | {k_max:<12.2e} | {img_max:<12.6f} | {img_mean:<12.6f} | {quality:<10}")
            
            if img_max > 0.02:
                best_slices.append((slice_idx, img_max, img_mean))
        
        # Recommendations
        print(f"\n4. RECOMMENDATIONS:")
        
        if best_slices:
            best_slices.sort(key=lambda x: x[1], reverse=True)
            print(f"   ✓ Found {len(best_slices)} good slices")
            print(f"\n   Top 5 slices to try:")
            for i, (idx, img_max, img_mean) in enumerate(best_slices[:5], 1):
                print(f"      {i}. Slice {idx}: max={img_max:.6f}, mean={img_mean:.6f}")
            
            # Visualize best slice
            best_idx = best_slices[0][0]
            print(f"\n   Creating visualization for best slice: {best_idx}")
            visualize_slice(FILE_PATH, best_idx)
            
            # Also show slice 18 for comparison
            print(f"\n   Creating visualization for current slice: 18")
            visualize_slice(FILE_PATH, 18)
            
            print(f"\n5. RECOMMENDED SETTINGS:")
            print(f"   SLICE_IDX = {best_idx}  # Best slice found")
            print(f"   Use norm='ortho' for FFT")
            print(f"   No additional k-space scaling needed")
            
        else:
            print(f"   ⚠️  No good slices found!")
            print(f"   This file may have issues. Try:")
            print(f"   - A different file from the dataset")
            print(f"   - multicoil_train instead of multicoil_test")
            print(f"   - Check if file is corrupted")

def visualize_slice(file_path, slice_idx):
    """Visualize a specific slice."""
    with h5py.File(file_path, "r") as hf:
        kspace_raw = hf['kspace'][slice_idx]
        kspace = fastmri_to_complex(kspace_raw)
    
    # Reconstruct
    img_coils = np.fft.ifft2(kspace, norm='ortho')
    img_rss = np.sqrt(np.sum(np.abs(img_coils)**2, axis=0))
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # K-space (log magnitude, first coil)
    axes[0, 0].imshow(np.log(np.abs(kspace[0]) + 1e-10), cmap='gray')
    axes[0, 0].set_title(f'K-space Coil 0\n(log magnitude)')
    axes[0, 0].axis('off')
    
    # RSS image
    im1 = axes[0, 1].imshow(img_rss, cmap='gray')
    axes[0, 1].set_title(f'RSS Image\nMax: {img_rss.max():.6f}')
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1])
    
    # Image histogram
    axes[0, 2].hist(img_rss.flatten(), bins=50)
    axes[0, 2].set_xlabel('Intensity')
    axes[0, 2].set_title('Image Histogram')
    axes[0, 2].set_yscale('log')
    
    # Individual coil images
    for i in range(3):
        axes[1, i].imshow(np.abs(img_coils[i]), cmap='gray')
        axes[1, i].set_title(f'Coil {i} Image')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'diagnostic_slice_{slice_idx}.png', dpi=150, bbox_inches='tight')
    print(f"   Saved: diagnostic_slice_{slice_idx}.png")
    plt.close()

if __name__ == "__main__":
    diagnose_knee_data()
    print("\n" + "="*80)
    print("DIAGNOSTIC COMPLETE")
    print("="*80)