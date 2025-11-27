"""
Explore fastMRI dataset to find good slices and check data format
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def fastmri_to_complex(tensor):
    """Convert fastMRI (..., 2) format to complex."""
    if tensor.shape[-1] == 2:
        return tensor[..., 0] + 1j * tensor[..., 1]
    return tensor


def explore_h5_structure(file_path):
    """Explore the structure of the HDF5 file."""
    print("="*80)
    print(f"EXPLORING: {file_path}")
    print("="*80)
    
    with h5py.File(file_path, "r") as hf:
        print("\nTop-level keys:")
        for key in hf.keys():
            print(f"  - {key}")
        
        # Check kspace
        if 'kspace' in hf:
            kspace_dataset = hf['kspace']
            print(f"\nK-space dataset:")
            print(f"  Shape: {kspace_dataset.shape}")
            print(f"  Dtype: {kspace_dataset.dtype}")
            print(f"  # Slices: {kspace_dataset.shape[0]}")
            
            # Sample first slice
            first_slice = kspace_dataset[0]
            print(f"  Single slice shape: {first_slice.shape}")
            
        # Check for other common keys
        for key in ['reconstruction_rss', 'reconstruction_esc', 'ismrmrd_header']:
            if key in hf:
                print(f"\n{key}: {hf[key].shape if hasattr(hf[key], 'shape') else 'exists'}")
        
        # Check attributes
        print("\nDataset attributes:")
        for key in hf.attrs.keys():
            print(f"  {key}: {hf.attrs[key]}")


def analyze_all_slices(file_path, max_slices=None):
    """Analyze all slices to find good ones."""
    print("\n" + "="*80)
    print("ANALYZING ALL SLICES")
    print("="*80)
    
    with h5py.File(file_path, "r") as hf:
        kspace_dataset = hf['kspace']
        num_slices = kspace_dataset.shape[0]
        
        if max_slices:
            num_slices = min(num_slices, max_slices)
        
        results = []
        
        for slice_idx in range(num_slices):
            kspace_raw = kspace_dataset[slice_idx]
            kspace = fastmri_to_complex(kspace_raw)
            
            # Compute metrics
            k_max = np.abs(kspace).max()
            k_mean = np.abs(kspace).mean()
            k_nonzero = np.count_nonzero(kspace) / kspace.size
            
            # Get image
            kspace_scaled = kspace / (k_max + 1e-10)
            img_coils = np.fft.ifft2(kspace_scaled, norm='ortho')
            img_rss = np.sqrt(np.sum(np.abs(img_coils)**2, axis=0))
            
            img_max = img_rss.max()
            img_mean = img_rss.mean()
            dynamic_range = img_max / (img_mean + 1e-10)
            
            # Score: prefer slices with good k-space values and dynamic range
            score = 0
            if k_max > 1e-4:  # Reasonable k-space magnitude
                score += 5
            if k_nonzero > 0.1:  # Not too sparse
                score += 3
            if dynamic_range > 3:  # Good contrast
                score += 5
            if img_max > 0.01:  # Reasonable image values
                score += 3
            
            results.append({
                'slice': slice_idx,
                'k_max': k_max,
                'k_mean': k_mean,
                'k_nonzero': k_nonzero,
                'img_max': img_max,
                'img_mean': img_mean,
                'dynamic_range': dynamic_range,
                'score': score
            })
            
            if slice_idx % 5 == 0:
                print(f"Slice {slice_idx:3d}: k_max={k_max:.2e}, img_max={img_max:.4f}, "
                      f"DR={dynamic_range:.2f}, score={score}")
        
        # Sort by score
        results.sort(key=lambda x: x['score'], reverse=True)
        
        print("\n" + "="*80)
        print("TOP 10 SLICES (by score)")
        print("="*80)
        print(f"{'Slice':<6} | {'K-max':<10} | {'Img-max':<10} | {'Dyn.Range':<10} | {'Score':<6}")
        print("-" * 60)
        
        for r in results[:10]:
            print(f"{r['slice']:<6} | {r['k_max']:<10.2e} | {r['img_max']:<10.4f} | "
                  f"{r['dynamic_range']:<10.2f} | {r['score']:<6}")
        
        return results


def visualize_slice(file_path, slice_idx):
    """Create detailed visualization of a specific slice."""
    print(f"\n{'='*80}")
    print(f"VISUALIZING SLICE {slice_idx}")
    print(f"{'='*80}")
    
    with h5py.File(file_path, "r") as hf:
        kspace_raw = hf['kspace'][slice_idx]
        kspace = fastmri_to_complex(kspace_raw)
    
    print(f"K-space shape: {kspace.shape}")
    print(f"K-space magnitude: [{np.abs(kspace).min():.2e}, {np.abs(kspace).max():.2e}]")
    
    # Try scaled reconstruction
    k_max = np.abs(kspace).max()
    kspace_scaled = kspace / (k_max + 1e-10)
    img_coils = np.fft.ifft2(kspace_scaled, norm='ortho')
    img_rss = np.sqrt(np.sum(np.abs(img_coils)**2, axis=0))
    
    print(f"RSS image: [{img_rss.min():.4f}, {img_rss.max():.4f}]")
    
    # Create visualization
    fig = plt.figure(figsize=(18, 10))
    
    # Row 1: K-space (3 coils)
    for i in range(min(3, kspace.shape[0])):
        plt.subplot(3, 4, i+1)
        plt.imshow(np.log(np.abs(kspace[i]) + 1e-10), cmap='gray')
        plt.title(f'K-space Coil {i}\n(log magnitude)')
        plt.colorbar()
        plt.axis('off')
    
    # Row 1, col 4: RSS image
    plt.subplot(3, 4, 4)
    plt.imshow(img_rss, cmap='gray')
    plt.title(f'RSS Image\nMax: {img_rss.max():.4f}')
    plt.colorbar()
    plt.axis('off')
    
    # Row 2: Image domain (3 coils)
    for i in range(min(3, kspace.shape[0])):
        plt.subplot(3, 4, i+5)
        plt.imshow(np.abs(img_coils[i]), cmap='gray')
        plt.title(f'Image Coil {i}')
        plt.colorbar()
        plt.axis('off')
    
    # Row 2, col 4: K-space center zoom
    plt.subplot(3, 4, 8)
    H, W = kspace.shape[1:]
    center_h, center_w = H//2, W//2
    k_center = kspace[0, center_h-20:center_h+20, center_w-20:center_w+20]
    plt.imshow(np.abs(k_center), cmap='viridis')
    plt.title('K-space Center\n(Coil 0, 40x40 region)')
    plt.colorbar()
    
    # Row 3: Histograms
    plt.subplot(3, 4, 9)
    k_mags = np.abs(kspace.flatten())
    k_mags_nonzero = k_mags[k_mags > 0]
    if len(k_mags_nonzero) > 0:
        plt.hist(np.log10(k_mags_nonzero + 1e-10), bins=50)
        plt.xlabel('Log10(K-space magnitude)')
        plt.ylabel('Count')
        plt.title('K-space Distribution')
    
    plt.subplot(3, 4, 10)
    img_vals = img_rss.flatten()
    plt.hist(img_vals, bins=50)
    plt.xlabel('Image intensity')
    plt.ylabel('Count')
    plt.title('Image Distribution')
    
    # Row 3: Line profiles
    plt.subplot(3, 4, 11)
    mid_row = H // 2
    plt.plot(np.abs(kspace[0, mid_row, :]))
    plt.xlabel('Column')
    plt.ylabel('K-space magnitude')
    plt.title('K-space Profile (middle row)')
    plt.grid(True)
    
    plt.subplot(3, 4, 12)
    plt.plot(img_rss[mid_row, :])
    plt.xlabel('Column')
    plt.ylabel('Image intensity')
    plt.title('Image Profile (middle row)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'slice_{slice_idx}_analysis.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved visualization to 'slice_{slice_idx}_analysis.png'")
    plt.show()


def compare_files(data_dir):
    """Compare multiple files in the directory."""
    print("\n" + "="*80)
    print("COMPARING MULTIPLE FILES")
    print("="*80)
    
    data_path = Path(data_dir)
    h5_files = list(data_path.glob("*.h5"))[:5]  # Check first 5 files
    
    if not h5_files:
        print(f"No .h5 files found in {data_dir}")
        return
    
    print(f"Found {len(h5_files)} files to check\n")
    
    for file_path in h5_files:
        print(f"\n{file_path.name}:")
        try:
            with h5py.File(file_path, "r") as hf:
                kspace = fastmri_to_complex(hf['kspace'][10])  # Check slice 10
                k_max = np.abs(kspace).max()
                
                kspace_scaled = kspace / (k_max + 1e-10)
                img_coils = np.fft.ifft2(kspace_scaled, norm='ortho')
                img_rss = np.sqrt(np.sum(np.abs(img_coils)**2, axis=0))
                
                print(f"  K-space max: {k_max:.2e}")
                print(f"  Image range: [{img_rss.min():.4f}, {img_rss.max():.4f}]")
                print(f"  # Slices: {hf['kspace'].shape[0]}")
                
        except Exception as e:
            print(f"  Error: {e}")


def main():
    """Main exploration script."""
    FILE_PATH = "./data/fastMRI/multicoil_test/file1000110.h5"
    DATA_DIR = "./data/fastMRI/multicoil_test/"
    
    print("FASTMRI DATASET EXPLORER")
    print("="*80)
    
    # 1. Explore file structure
    explore_h5_structure(FILE_PATH)
    
    # 2. Analyze all slices
    results = analyze_all_slices(FILE_PATH, max_slices=30)
    
    # 3. Visualize best slice
    if results:
        best_slice = results[0]['slice']
        print(f"\nVisualizing best slice: {best_slice}")
        visualize_slice(FILE_PATH, best_slice)
        
        # Also visualize current slice for comparison
        print(f"\nVisualizing current slice: 18")
        visualize_slice(FILE_PATH, 18)
    
    # 4. Compare multiple files
    compare_files(DATA_DIR)
    
    # 5. Recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    
    if results:
        best = results[0]
        print(f"\n✓ Best slice found: {best['slice']}")
        print(f"  K-space max: {best['k_max']:.2e}")
        print(f"  Image max: {best['img_max']:.4f}")
        print(f"  Dynamic range: {best['dynamic_range']:.2f}")
        print(f"\nUpdate your script with: SLICE_IDX = {best['slice']}")
    
    print("\nIf all slices have small k-space values, consider:")
    print("  1. Using a different file from the dataset")
    print("  2. Checking if this is training vs validation data")
    print("  3. Looking at fastMRI documentation for data format")
    print("  4. Using the knee dataset instead of brain (different contrast)")


if __name__ == "__main__":
    main()