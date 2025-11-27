import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import nibabel as nib
from skimage.metrics import structural_similarity as ssim

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Running on: {device}")

# ==========================================
# 1. KAN / MLP MODELS (Optimized)
# ==========================================
class ComplexVectorRBFLayer(nn.Module):
    def __init__(self, in_features, out_features, num_rbfs=32, init_data=None):
        super().__init__()
        self.in_features = in_features
        self.num_rbfs = num_rbfs
        
        # Data-Driven Initialization (Crucial for KAN)
        if init_data is not None:
            indices = torch.randperm(init_data.size(0))[:num_rbfs]
            self.centers = nn.Parameter(init_data[indices].clone())
        else:
            self.centers = nn.Parameter(torch.randn(num_rbfs, in_features, dtype=torch.cfloat))

        # Adjusted Gamma for Real Brain Data
        self.log_gamma = nn.Parameter(torch.ones(num_rbfs) * -0.5) 
        self.rbf_weights = nn.Parameter(torch.randn(num_rbfs, out_features, dtype=torch.cfloat) * 0.1)
        self.linear = nn.Linear(in_features, out_features, bias=False, dtype=torch.cfloat)

    def forward(self, z):
        z_expanded = z.unsqueeze(1) 
        mu_expanded = self.centers.unsqueeze(0)
        dist_sq = ((z_expanded - mu_expanded) * (z_expanded - mu_expanded).conj()).real.sum(dim=2)
        rbf_response = torch.exp(-torch.exp(self.log_gamma) * dist_sq)
        return self.linear(z) + torch.matmul(rbf_response.type(torch.complex64), self.rbf_weights)

class KANGrappaNet(nn.Module):
    def __init__(self, kernel_size=3, num_rbfs=64, init_data=None):
        super().__init__()
        self.in_features = kernel_size * kernel_size
        self.layer1 = ComplexVectorRBFLayer(self.in_features, 16, num_rbfs=num_rbfs, init_data=init_data)
        with torch.no_grad():
            if init_data is not None:
                l2_init = torch.tanh(self.layer1(init_data[:500]).real) + 1j * torch.tanh(self.layer1(init_data[:500]).imag)
            else: l2_init = None
        self.layer2 = ComplexVectorRBFLayer(16, 1, num_rbfs=num_rbfs//2, init_data=l2_init)
    def forward(self, x):
        x = self.layer1(x)
        x = torch.tanh(x.real) + 1j * torch.tanh(x.imag)
        return self.layer2(x)

class BaselineMLP(nn.Module):
    def __init__(self, kernel_size=3, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(kernel_size*kernel_size, hidden_dim, dtype=torch.cfloat),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1, dtype=torch.cfloat)
        )
    def forward(self, x):
        h = self.net[0](x)
        h = torch.nn.functional.relu(h.real) + 1j * torch.nn.functional.relu(h.imag)
        return self.net[2](h)

def count_parameters(model):
    total = 0
    for p in model.parameters(): mult = 2 if p.is_complex() else 1; total += p.numel() * mult
    return total

# ==========================================
# 2. HYBRID DATA LOADER (Anatomy + Physics)
# ==========================================
def get_hybrid_brain_data(filename="IXI016-Guys-0697-T2.nii.gz", accel=2):
    if not os.path.exists(filename):
        print(f"Error: {filename} not found.")
        return None

    print(f"Loading Real Anatomy: {filename}...")
    img_nii = nib.load(filename).get_fdata()
    
    # 1. Extract a nice slice
    # Middle slice usually has good ventricles/structure
    slice_idx = img_nii.shape[2] // 2
    img_real = img_nii[:, :, slice_idx]
    
    # 2. Crop to Square (160x160) for speed
    h, w = img_real.shape
    c_h, c_w = h//2, w//2
    crop = 160
    img_crop = img_real[c_h-crop//2:c_h+crop//2, c_w-crop//2:c_w+crop//2]
    
    # Normalize magnitude to 0-1
    img_crop = img_crop / np.max(img_crop)
    
    # --- 3. INJECT PHYSICS (The Benchmark Maker) ---
    # We add a Quadratic Phase Ramp. This mimics B0 field inhomogeneity.
    # This is what breaks the MLP but KAN should handle.
    x = np.linspace(-1, 1, crop)
    y = np.linspace(-1, 1, crop)
    X, Y = np.meshgrid(x, y)
    
    # Complex Phase Map: Spiraling phase
    phase_map = np.exp(1j * (3.0 * (X**2 + Y**2) + 2.0 * X))
    
    # Combine Real Anatomy with Synthetic Phase
    img_complex = img_crop * phase_map
    
    # 4. Convert to k-space
    kspace_gt = np.fft.fftshift(np.fft.fft2(img_complex))
    
    # Max Normalize K-space (Critical for RBFs)
    norm_factor = np.abs(kspace_gt).max() + 1e-8
    kspace_gt = torch.tensor(kspace_gt / norm_factor, dtype=torch.cfloat).to(device)
    
    # 5. Create CS Mask
    H, W = kspace_gt.shape
    torch.manual_seed(42)
    mask = torch.zeros_like(kspace_gt)
    center_y = H // 2
    
    # ACS (Center lines)
    mask[center_y-12:center_y+12, :] = 1.0 
    
    # Random sampling
    idx_list = list(range(0, center_y-12)) + list(range(center_y+12, H))
    perm = torch.randperm(len(idx_list))
    keep = (H // accel) - 24
    for i in perm[:keep]: mask[idx_list[i], :] = 1.0
    
    kspace_input = kspace_gt * mask
    kspace_acs = kspace_gt[center_y-12:center_y+12, :]
    
    return kspace_gt, kspace_input, kspace_acs, mask

# ==========================================
# 3. BENCHMARK
# ==========================================
def extract_patches_masked(kspace, kernel_size=3):
    k_real = kspace.real.unsqueeze(0).unsqueeze(0)
    k_imag = kspace.imag.unsqueeze(0).unsqueeze(0)
    unfold = nn.Unfold(kernel_size=(kernel_size, kernel_size), padding=kernel_size//2)
    patches = torch.complex(unfold(k_real), unfold(k_imag))
    patches = patches.permute(0, 2, 1).squeeze(0)
    center_idx = (kernel_size * kernel_size) // 2
    patches_masked = patches.clone()
    patches_masked[:, center_idx] = 0 + 0j
    return patches_masked

def train_model(model, name, X_train, Y_train, kspace_input, mask, epochs=1000):
    print(f"\n--- Training {name} ---")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = model(X_train)
        diff = pred.squeeze() - Y_train
        # L1 + Phase Loss
        loss = torch.abs(diff).mean() + 0.1 * (torch.angle(pred.squeeze()) - torch.angle(Y_train)).abs().mean()
        loss.backward()
        optimizer.step()
        if epoch % 500 == 0: print(f"Epoch {epoch} | Loss: {loss.item():.5f}")

    model.eval()
    with torch.no_grad():
        patches = extract_patches_masked(kspace_input)
        recon_flat = model(patches).squeeze()
        recon_grid = recon_flat.view(kspace_input.shape)
        final_kspace = kspace_input + (recon_grid * (1 - mask))
    return final_kspace

def run_benchmark():
    # Use the IXI file you already have
    data = get_hybrid_brain_data("IXI016-Guys-0697-T2.nii.gz")
    if data is None: return
    
    kspace_gt, kspace_input, kspace_acs, mask = data
    patches_acs = extract_patches_masked(kspace_acs)
    targets_acs = kspace_acs.flatten()
    
    # Models
    kan_model = KANGrappaNet(init_data=patches_acs).to(device)
    mlp_model = BaselineMLP(hidden_dim=200).to(device)
    
    print(f"Params: KAN={count_parameters(kan_model)} | MLP={count_parameters(mlp_model)}")
    
    kan_kspace = train_model(kan_model, "RBF-KAN", patches_acs, targets_acs, kspace_input, mask)
    mlp_kspace = train_model(mlp_model, "ReLU-MLP", patches_acs, targets_acs, kspace_input, mask)
    
    # Metrics
    gt_c = np.fft.ifft2(np.fft.ifftshift(kspace_gt.cpu().numpy()))
    kan_c = np.fft.ifft2(np.fft.ifftshift(kan_kspace.cpu().numpy()))
    mlp_c = np.fft.ifft2(np.fft.ifftshift(mlp_kspace.cpu().numpy()))
    
    # d_max = np.max(np.abs(gt_c))
    
    # def get_metrics(gt, recon):
    #     s = ssim(np.abs(gt), np.abs(recon), data_range=d_max)
    #     p = np.sum(np.abs(np.angle(gt)-np.angle(recon)) * np.abs(gt)) / np.sum(np.abs(gt))
    #     return s, p

    # --- SCALING FIX ---
    # Determine the peak brightness of the Ground Truth image
    plot_max = np.max(np.abs(gt_c))
    print(f"Plotting Max Intensity: {plot_max:.5f}")
    
    def get_metrics(gt, recon):
        s = ssim(np.abs(gt), np.abs(recon), data_range=plot_max)
        p = np.sum(np.abs(np.angle(gt)-np.angle(recon)) * np.abs(gt)) / np.sum(np.abs(gt))
        return s, p


    k_ssim, k_phase = get_metrics(gt_c, kan_c)
    m_ssim, m_phase = get_metrics(gt_c, mlp_c)
    
    print(f"\nRESULTS:\nKAN: SSIM={k_ssim:.3f}, Phase={k_phase:.4f}\nMLP: SSIM={m_ssim:.3f}, Phase={m_phase:.4f}")

    # # Plot
    # fig = plt.figure(figsize=(15, 10))
    # gs = fig.add_gridspec(2, 3)
    
    # # Normalize images for display
    # def norm_disp(x): return np.abs(x) / np.max(np.abs(x))

    # ax1 = fig.add_subplot(gs[0, 0]); ax1.imshow(norm_disp(gt_c), cmap='gray'); ax1.set_title("Real Brain (Simulated Phase)")
    # ax2 = fig.add_subplot(gs[0, 1]); ax2.imshow(norm_disp(kan_c), cmap='gray'); ax2.set_title(f"KAN (P.Err: {k_phase:.3f})")
    # ax3 = fig.add_subplot(gs[0, 2]); ax3.imshow(norm_disp(mlp_c), cmap='gray'); ax3.set_title(f"MLP (P.Err: {m_phase:.3f})")
    
    # ax4 = fig.add_subplot(gs[1, 0]); ax4.imshow(np.angle(gt_c), cmap='hsv'); ax4.set_title("Ground Truth Phase")
    # ax5 = fig.add_subplot(gs[1, 1]); ax5.imshow(np.abs(np.angle(gt_c)-np.angle(kan_c)), cmap='twilight', vmin=0, vmax=3.14); ax5.set_title("KAN Phase Error")
    # ax6 = fig.add_subplot(gs[1, 2]); ax6.imshow(np.abs(np.angle(gt_c)-np.angle(mlp_c)), cmap='twilight', vmin=0, vmax=3.14); ax6.set_title("MLP Phase Error")
    
    # plt.tight_layout()
    # plt.show()

    # --- PLOT WITH CORRECT SCALING ---
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 3)
    
    # Row 1: Magnitude Images (Scaled to plot_max)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(np.abs(gt_c), cmap='gray', vmin=0, vmax=plot_max)
    ax1.set_title("Real Brain (Simulated Phase)")
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(np.abs(kan_c), cmap='gray', vmin=0, vmax=plot_max)
    ax2.set_title(f"KAN Recon (SSIM: {k_ssim:.3f})")
    
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(np.abs(mlp_c), cmap='gray', vmin=0, vmax=plot_max)
    ax3.set_title(f"MLP Recon (SSIM: {m_ssim:.3f})")
    
    # Row 2: Phase Analysis
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.imshow(np.angle(gt_c), cmap='hsv')
    ax4.set_title("Ground Truth Phase")
    
    ax5 = fig.add_subplot(gs[1, 1])
    # Phase error is 0 to Pi (3.14)
    im5 = ax5.imshow(np.abs(np.angle(gt_c)-np.angle(kan_c)), cmap='twilight', vmin=0, vmax=3.14)
    ax5.set_title(f"KAN Phase Error: {k_phase:.3f}")
    
    ax6 = fig.add_subplot(gs[1, 2])
    im6 = ax6.imshow(np.abs(np.angle(gt_c)-np.angle(mlp_c)), cmap='twilight', vmin=0, vmax=3.14)
    ax6.set_title(f"MLP Phase Error: {m_phase:.3f}")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_benchmark()