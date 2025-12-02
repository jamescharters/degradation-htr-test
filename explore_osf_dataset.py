"""
Quick script to explore the OSF QSM dataset structure
Run this first to see what subjects and orientations you have
"""

from pathlib import Path
import nibabel as nib

# ==========================================
# CONFIGURATION
# ==========================================
DATASET_PATH = Path("./data/OSF_QSM_Dataset")

# ==========================================
# EXPLORATION FUNCTIONS
# ==========================================

def explore_dataset_structure(dataset_path):
    """
    Scan the dataset and report its structure
    """
    print("="*60)
    print("OSF QSM DATASET EXPLORER")
    print("="*60)
    
    # Check if path exists
    if not dataset_path.exists():
        print(f"\n❌ Dataset path not found: {dataset_path}")
        print("\nPlease update DATASET_PATH in the script.")
        return
    
    print(f"\nDataset location: {dataset_path}")
    
    # Check for train_data and test_data folders
    train_path = dataset_path / "train_data"
    test_path = dataset_path / "test_data"
    
    splits = []
    if train_path.exists():
        splits.append(("train", train_path))
    if test_path.exists():
        splits.append(("test", test_path))
    
    if not splits:
        print("\n⚠ No train_data or test_data folders found!")
        print("   Checking for subjects in root directory...")
        splits = [("root", dataset_path)]
    
    # Explore each split
    all_subjects = []
    
    for split_name, split_path in splits:
        print(f"\n{'='*60}")
        print(f"{split_name.upper()} SPLIT")
        print(f"{'='*60}")
        
        # Find subject folders
        subject_dirs = [d for d in split_path.iterdir() if d.is_dir()]
        
        if not subject_dirs:
            print(f"  No subject folders found in {split_path}")
            continue
        
        print(f"\nFound {len(subject_dirs)} subjects:")
        
        for subject_dir in sorted(subject_dirs):
            subject_id = subject_dir.name
            print(f"\n  📁 {subject_id}")
            
            # Count orientations
            phi_files = list(subject_dir.glob("phi*.nii*"))
            cosmos_files = list(subject_dir.glob("cosmos*.nii*"))
            mask_files = list(subject_dir.glob("mask*.nii*"))
            
            # Extract orientation numbers
            orientations = set()
            for f in phi_files:
                # Extract number from filename like "phi1.nii.gz" or "phi_1.nii.gz"
                name = f.stem.replace('.nii', '')
                num = ''.join(c for c in name if c.isdigit())
                if num:
                    orientations.add(int(num))
            
            orientations = sorted(orientations)
            
            print(f"     Orientations: {len(orientations)} found")
            if orientations:
                print(f"     Available: {orientations}")
            
            print(f"     Files:")
            print(f"       - phi files: {len(phi_files)}")
            print(f"       - cosmos files: {len(cosmos_files)}")
            print(f"       - mask files: {len(mask_files)}")
            
            # Load one file to check dimensions
            if phi_files:
                try:
                    img = nib.load(str(phi_files[0]))
                    shape = img.shape
                    voxel_size = img.header.get_zooms()[:3]
                    print(f"     Shape: {shape}")
                    print(f"     Voxel size: {voxel_size} mm")
                except Exception as e:
                    print(f"     ⚠ Could not load file: {e}")
            
            all_subjects.append({
                'id': subject_id,
                'split': split_name,
                'path': subject_dir,
                'orientations': orientations,
                'n_orientations': len(orientations)
            })
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"\nTotal subjects found: {len(all_subjects)}")
    
    if all_subjects:
        train_subjects = [s for s in all_subjects if s['split'] == 'train']
        test_subjects = [s for s in all_subjects if s['split'] == 'test']
        
        print(f"  Train subjects: {len(train_subjects)}")
        print(f"  Test subjects: {len(test_subjects)}")
        
        # Average orientations
        avg_orientations = sum(s['n_orientations'] for s in all_subjects) / len(all_subjects)
        print(f"  Average orientations per subject: {avg_orientations:.1f}")
        
        print("\n" + "="*60)
        print("RECOMMENDATIONS FOR YOUR EXPERIMENTS")
        print("="*60)
        
        if train_subjects:
            print(f"\n1. For initial testing, use:")
            print(f"   SUBJECT_ID = '{train_subjects[0]['id']}'")
            print(f"   ORIENTATION = 1")
            
        print(f"\n2. For cross-validation:")
        print(f"   - Use train subjects for development")
        print(f"   - Hold out test subjects for final evaluation")
        
        print(f"\n3. For multi-orientation experiments:")
        if all_subjects[0]['n_orientations'] > 1:
            print(f"   - Each subject has {all_subjects[0]['n_orientations']} orientations")
            print(f"   - Train on orientation 1, test on others")
            print(f"   - Or use all orientations for data augmentation")
        
        print("\n" + "="*60)
        
        # Generate ready-to-use code
        print("\n📋 COPY-PASTE CONFIG:")
        print("-"*60)
        if train_subjects:
            print(f'DATASET_PATH = Path("./data/OSF_QSM_Dataset")')
            print(f'SUBJECT_ID = "{train_subjects[0]["id"]}"')
            print(f'ORIENTATION = 1')
            print(f'# Available subjects: {[s["id"] for s in train_subjects[:5]]}...')
        print("-"*60)


# ==========================================
# RUN EXPLORATION
# ==========================================

if __name__ == "__main__":
    explore_dataset_structure(DATASET_PATH)
    
    print("\n✓ Exploration complete!")
    print("\nNext steps:")
    print("  1. Update your config with the subject ID shown above")
    print("  2. Run load_osf_qsm_dataset.py to visualize the data")
    print("  3. Run fastmri_kan_REAL_QSM_data.py to train your models")