"""
Extract raw wrapped phase from QSM dataset
The 'phi' files are already unwrapped - we need to go back to raw data
"""

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

DATASET_PATH = Path("./data/OSF_QSM_Dataset")
SUBJECT_ID = "Subject1"
ORIENTATION = 1

def load_and_create_wrapped_phase(dataset_path, subject_id, orientation):
    """
    Since phi is already unwrapped, we need to:
    1. Load the magnitude and phase (if raw data available)
    2. Or artificially add wrapping to phi for testing
    """
    
    train_path = dataset_path / "train_data" / subject_id
    test_path = dataset_path / "test_data" / subject_id
    
    subject_path = train_path if train_path.exists() else test_path
    
    # Load unwrapped phi and COSMOS
    phi_file = subject_path / f"phi{orientation}.nii.gz"
    cosmos_file = subject_path / f"cosmos{orientation}.nii.gz"
    mask_file = subject_path / f"mask{orientation}.nii.gz"
    
    phi_3d = nib.load(str(phi_file)).get_fdata().astype(np.float32)
    cosmos_3d = nib.load(str(cosmos_file)).get_fdata().astype(np.float32)
    mask_3d = nib.load(str(mask_file)).get_fdata().astype(np.float32)
    
    print("Loaded unwrapped data:")
    print(f"  Phi range: [{phi_3d.min():.3f}, {phi_3d.max():.3f}]")
    print(f"  COSMOS range: [{cosmos_3d.min():.3f}, {cosmos_3d.max():.3f}]")
    
    # APPROACH: Add controlled wrapping to phi for testing
    # Multiply by a factor to push values outside [-π, π]
    
    slice_idx = phi_3d.shape[2] // 2
    phi_2d = phi_3d[:, :, slice_idx]
    cosmos_2d = cosmos_3d[:, :, slice_idx]
    mask_2d = mask_3d[:, :, slice_idx]
    
    # Scale phi to create wrapping
    scale_factor = 15.0  # Adjust to control amount of wrapping
    phi_scaled = phi_2d * scale_factor
    
    # Now wrap it
    phi_wrapped = np.angle(np.exp(1j * phi_scaled))
    
    print(f"\nAfter scaling by {scale_factor}x:")
    print(f"  Scaled phi range: [{phi_scaled.min():.3f}, {phi_scaled.max():.3f}]")
    print(f"  Wrapped phi range: [{phi_wrapped.min():.3f}, {phi_wrapped.max():.3f}]")
    
    # For ground truth, use the scaled (unwrapped) version
    phase_true = phi_scaled
    phase_wrapped = phi_wrapped
    magnitude = mask_2d
    
    # Visualize the difference
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    axes[0,0].imshow(phi_2d, cmap='twilight')
    axes[0,0].set_title('Original Phi\n(Already Unwrapped)')
    
    axes[0,1].imshow(phi_scaled, cmap='twilight')
    axes[0,1].set_title(f'Scaled Phi (×{scale_factor})\n(Ground Truth)')
    
    axes[0,2].imshow(phi_wrapped, cmap='twilight')
    axes[0,2].set_title('Wrapped Phase\n(Input to Model)')
    
    axes[1,0].imshow(mask_2d, cmap='gray')
    axes[1,0].set_title('Mask')
    
    # Show the wraps
    wrap_count = (phi_scaled - phi_wrapped) / (2 * np.pi)
    axes[1,1].imshow(wrap_count, cmap='RdBu', vmin=-5, vmax=5)
    axes[1,1].set_title('Wrap Count (k)')
    plt.colorbar(axes[1,1].images[0], ax=axes[1,1])
    
    # Histogram of wrap counts
    wrap_counts_masked = wrap_count[mask_2d > 0.5]
    axes[1,2].hist(wrap_counts_masked.flatten(), bins=50)
    axes[1,2].set_title('Distribution of Wrap Counts')
    axes[1,2].set_xlabel('k value')
    
    plt.tight_layout()
    plt.savefig('wrapped_phase_creation.png', dpi=150)
    plt.show()
    
    print(f"\nWrap statistics (inside mask):")
    print(f"  Min wrap: {wrap_counts_masked.min():.1f}")
    print(f"  Max wrap: {wrap_counts_masked.max():.1f}")
    print(f"  Mean |wrap|: {np.abs(wrap_counts_masked).mean():.2f}")
    print(f"  % with wraps: {(np.abs(wrap_counts_masked) > 0.5).sum() / len(wrap_counts_masked) * 100:.1f}%")
    
    return magnitude, phase_true, phase_wrapped

if __name__ == "__main__":
    mag, p_true, p_wrapped = load_and_create_wrapped_phase(
        DATASET_PATH, SUBJECT_ID, ORIENTATION
    )
    
    print("\n✓ Created wrapped phase for testing!")
    print("\nNow modify your KAN experiment to use:")
    print("  mag, p_true, p_wrapped = load_and_create_wrapped_phase(...)")