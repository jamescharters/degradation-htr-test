import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.fft import fft2, ifft2
import nibabel as nib
from skimage.transform import resize

# ============================================================================
# 1. SETUP
# ============================================================================
def get_data():
    try:
        nii = nib.load("IXI002-Guys-0828-T2.nii")
        data = nii.get_fdata()[:, :, nii.shape[2]//2]
        data = resize(np.rot90(data), (256, 256))
        # Robust Normalize
        p99 = np.percentile(data, 99)
        data = np.clip(data, 0, p99)
        data = (data - data.min()) / (data.max() - data.min())
        img = torch.tensor(data, dtype=torch.float32)
    except:
        x, y = np.meshgrid(np.linspace(-1,1,256), np.linspace(-1,1,256))
        img = torch.tensor((x**2 + y**2 < 0.6).astype(float), dtype=torch.float32)
    return img

def create_problem(img):
    TRUE_T = 100
    TRUE_S = 12.5 # The case that failed
    
    H, W = img.shape
    FOV = 256.0
    
    curve = torch.zeros(H)
    curve[TRUE_T:] = TRUE_S
    
    ky = torch.arange(H)
    phase = 2 * np.pi * (curve.view(-1, 1) / FOV) * ((ky - H//2)/H).view(-1, 1) * H
    
    kspace = fft2(img)
    corrupted_k = kspace * torch.exp(1j * phase)
    corrupted_img = ifft2(corrupted_k).real
    
    return corrupted_k, corrupted_img, TRUE_T, TRUE_S

def get_corrected_image(kspace, t, s):
    H, W = kspace.shape
    FOV = 256.0
    
    curve = torch.zeros(H, device=kspace.device)
    t = int(np.clip(t, 0, H-1))
    curve[t:] = float(s)
    
    ky = torch.arange(H, device=kspace.device)
    phase = 2 * np.pi * (curve.view(-1, 1) / FOV) * ((ky - H//2)/H).view(-1, 1) * H
    
    fixed = ifft2(kspace * torch.exp(-1j * phase)).real
    return fixed

# ============================================================================
# 2. THE METRIC: IMAGE ENTROPY
# ============================================================================
def measure_image_entropy(image):
    """
    Calculates the Entropy of the pixel intensity histogram.
    Sharp images have 'peaky' histograms (Background + Tissue).
    Blurred/Ghosted/Cancelled images have flat histograms.
    Lower is Better.
    """
    # 1. Binning (Differentiable-ish approximation not needed for grid search)
    # We normalize image to 0-1
    img = torch.abs(image)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    
    # 2. Histogram (100 bins)
    hist = torch.histc(img, bins=100, min=0, max=1)
    
    # 3. Probability distribution
    p = hist / (torch.sum(hist) + 1e-8)
    
    # 4. Entropy
    # remove zero probs to avoid log(0)
    p = p[p > 0]
    entropy = -torch.sum(p * torch.log(p))
    
    return entropy.item()

# ============================================================================
# 3. THE SOLVER
# ============================================================================
def solve_entropy(kspace, corrupted_img):
    
    # STEP 1: SCAN SHIFT (Assumption T=128)
    # We scan a wide range to catch the 12.5mm shift
    print("Step 1: Scanning Shift (Metric: Image Entropy)...")
    shifts = np.linspace(-20, 20, 81) # 0.5mm steps
    scores = []
    
    # We fix T at center (usually good enough to find approx shift)
    test_t = 128
    
    for s in shifts:
        img = get_corrected_image(kspace, test_t, s)
        score = measure_image_entropy(img)
        scores.append(score)
        
    # Pick Best Shift
    best_idx = np.argmin(scores)
    est_s = shifts[best_idx]
    
    # Plot the curve so we can SEE if 2.5mm was a trap
    plt.figure(figsize=(10, 4))
    plt.plot(shifts, scores, 'b-')
    plt.plot(est_s, scores[best_idx], 'ro', label=f'Min: {est_s:.1f}mm')
    plt.title("Entropy Landscape (Shift)")
    plt.xlabel("Shift (mm)")
    plt.ylabel("Entropy (Lower is Better)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("entropy_curve.png")
    
    print(f"  > Best Shift Candidate: {est_s:.2f}mm")
    
    # STEP 2: SCAN TIME (Refine T)
    print("Step 2: Scanning Time...")
    times = np.arange(0, 256, 4)
    scores_t = []
    
    for t in times:
        img = get_corrected_image(kspace, t, est_s)
        score = measure_image_entropy(img)
        scores_t.append(score)
        
    best_t_idx = np.argmin(scores_t)
    est_t = times[best_t_idx]
    
    print(f"  > Best Time: Line {est_t}")
    
    return est_t, est_s

# ============================================================================
# 4. RUN
# ============================================================================
def main():
    img = get_data()
    kspace, corrupted, true_t, true_s = create_problem(img)
    
    # Solve
    est_t, est_s = solve_entropy(kspace, corrupted)
    
    fixed = get_corrected_image(kspace, est_t, est_s)
    
    plt.figure(figsize=(15, 6))
    
    plt.subplot(1, 3, 1)
    plt.imshow(corrupted, cmap='gray')
    plt.title(f"Corrupted (T={true_t}, S={true_s}mm)")
    
    plt.subplot(1, 3, 2)
    plt.imshow(fixed, cmap='gray')
    plt.title(f"Entropy Result (T={est_t}, S={est_s:.2f}mm)")
    
    plt.subplot(1, 3, 3)
    diff = torch.abs(img - fixed)
    plt.imshow(diff, cmap='inferno', vmin=0, vmax=0.2)
    plt.title("Residual Error")
    
    plt.tight_layout()
    plt.savefig("entropy_autofocus.png")
    plt.show()

if __name__ == "__main__":
    main()