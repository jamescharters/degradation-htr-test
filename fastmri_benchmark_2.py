import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import h5py
from skimage.metrics import structural_similarity as ssim

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Running on: {device}")

# ==========================================
# STABLE KAN IMPLEMENTATION
# ==========================================

class StableComplexRBFKAN(nn.Module):
    """
    Stable Radial Basis Function Kolmogorov-Arnold Network for complex k-space.
    """
    def __init__(self, in_dim=9, hidden_dim=16, out_dim=1, num_rbfs=32, init_data=None):
        super().__init__()
        
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_rbfs = num_rbfs
        
        # Layer 1: Input -> Hidden
        self.linear1 = nn.Linear(in_dim, hidden_dim, bias=True, dtype=torch.cfloat)
        nn.init.xavier_uniform_(self.linear1.weight, gain=0.1)
        nn.init.zeros_(self.linear1.bias)
        
        # RBF Centers - data-driven if available
        if init_data is not None and len(init_data) >= num_rbfs:
            indices = torch.randperm(len(init_data))[:num_rbfs]
            centers_init = init_data[indices].clone()
        else:
            centers_init = torch.randn(num_rbfs, in_dim, dtype=torch.cfloat) * 0.01
        
        self.rbf_centers1 = nn.Parameter(centers_init)
        self.log_gamma1 = nn.Parameter(torch.zeros(num_rbfs))
        self.rbf_weights1 = nn.Parameter(
            torch.randn(num_rbfs, hidden_dim, dtype=torch.cfloat) * 0.01
        )
        
        # Layer 2: Hidden -> Output
        self.linear2 = nn.Linear(hidden_dim, out_dim, bias=True, dtype=torch.cfloat)
        nn.init.xavier_uniform_(self.linear2.weight, gain=0.1)
        nn.init.zeros_(self.linear2.bias)
        
        self.rbf_centers2 = nn.Parameter(
            torch.randn(num_rbfs // 2, hidden_dim, dtype=torch.cfloat) * 0.01
        )
        self.log_gamma2 = nn.Parameter(torch.zeros(num_rbfs // 2))
        self.rbf_weights2 = nn.Parameter(
            torch.randn(num_rbfs // 2, out_dim, dtype=torch.cfloat) * 0.01
        )
        
        # Learnable scaling factors
        self.rbf_scale1 = nn.Parameter(torch.tensor(0.1))
        self.rbf_scale2 = nn.Parameter(torch.tensor(0.1))
        
    def complex_rbf(self, x, centers, log_gamma, weights, scale):
        x_exp = x.unsqueeze(1)
        c_exp = centers.unsqueeze(0)
        
        diff = x_exp - c_exp
        dist_sq = (diff * diff.conj()).real.sum(dim=2)
        
        gamma = torch.exp(log_gamma).clamp(min=0.01, max=10.0)
        rbf_response = torch.exp(-gamma * dist_sq)
        rbf_response = rbf_response.clamp(max=10.0)
        
        output = torch.matmul(rbf_response.type(torch.complex64), weights)
        return scale * output
    
    def complex_activation(self, x):
        return torch.tanh(x.real) + 1j * torch.tanh(x.imag)
    
    def forward(self, x):
        # Layer 1
        h = self.linear1(x)
        h_rbf = self.complex_rbf(x, self.rbf_centers1, self.log_gamma1, 
                                  self.rbf_weights1, self.rbf_scale1)
        h = h + h_rbf
        h = self.complex_activation(h)
        
        # Layer 2
        out = self.linear2(h)
        out_rbf = self.complex_rbf(h, self.rbf_centers2, self.log_gamma2,
                                     self.rbf_weights2, self.rbf_scale2)
        out = out + out_rbf
        
        return out


# ==========================================
# BASELINE MLP
# ==========================================

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


# ==========================================
# HELPER FUNCTIONS
# ==========================================

def extract_patches_masked(kspace, kernel_size=3):
    """Extract overlapping patches with center pixel zeroed."""
    k_real = kspace.real.unsqueeze(0).unsqueeze(0)
    k_imag = kspace.imag.unsqueeze(0).unsqueeze(0)
    
    pad = kernel_size // 2
    unfold = nn.Unfold(kernel_size=(kernel_size, kernel_size), padding=pad)
    
    patches_real = unfold(k_real)
    patches_imag = unfold(k_imag)
    
    patches = torch.complex(patches_real, patches_imag)
    patches = patches.permute(0, 2, 1).squeeze(0)
    
    center_idx = (kernel_size * kernel_size) // 2
    patches[:, center_idx] = 0 + 0j
    
    return patches


def count_parameters(model):
    total = 0
    for p in model.parameters():
        mult = 2 if p.is_complex() else 1
        total += p.numel() * mult
    return total


# ==========================================
# TRAINING FUNCTIONS
# ==========================================

def train_stable_kan(model, X_train, Y_train, epochs=1000, lr=0.001):
    """Train stable KAN with monitoring."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=100
    )
    
    model.train()
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        pred = model(X_train).squeeze()
        diff = pred - Y_train
        loss_mag = torch.abs(diff).mean()
        loss_phase = torch.abs(torch.angle(pred) - torch.angle(Y_train)).mean()
        loss = loss_mag + 0.1 * loss_phase
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step(loss)
        
        if epoch % 500 == 0:
            pred_mag = torch.abs(pred).mean().item()
            target_mag = torch.abs(Y_train).mean().item()
            mag_ratio = pred_mag / target_mag if target_mag > 0 else 0
            print(f"Epoch {epoch} | Loss: {loss.item():.5f} | Mag Ratio: {mag_ratio:.2f}x")
    
    return model


def train_mlp(model, X_train, Y_train, epochs=1000, lr=0.005):
    """Train baseline MLP."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = model(X_train).squeeze()
        diff = pred - Y_train
        loss = torch.abs(diff).mean() + 0.1 * (torch.angle(pred) - torch.angle(Y_train)).abs().mean()
        loss.backward()
        optimizer.step()
        
        if epoch % 500 == 0:
            print(f"Epoch {epoch} | Loss: {loss.item():.5f}")
    
    return model


# ==========================================
# RECONSTRUCTION FUNCTIONS
# ==========================================

def reconstruct_with_kan(model, kspace_input, mask):
    """Reconstruct with stable KAN."""
    model.eval()
    H, W = kspace_input.shape
    
    with torch.no_grad():
        patches = extract_patches_masked(kspace_input, kernel_size=3)
        recon_flat = model(patches).squeeze()
        
        # Output normalization
        input_nonzero = kspace_input[kspace_input.abs() > 1e-10]
        if len(input_nonzero) > 0:
            target_magnitude = input_nonzero.abs().mean()
            recon_magnitude = recon_flat.abs().mean()
            
            if recon_magnitude > 3 * target_magnitude:
                scale_factor = target_magnitude / recon_magnitude
                print(f"  Scaling KAN output by {scale_factor:.3f}")
                recon_flat = recon_flat * scale_factor
        
        recon_grid = recon_flat.view(H, W)
        final_kspace = kspace_input + (recon_grid * (1 - mask))
        
        # Energy check
        output_energy = torch.abs(final_kspace).sum()
        input_energy = torch.abs(kspace_input).sum()
        energy_ratio = output_energy / input_energy
        print(f"  KAN energy ratio: {energy_ratio:.2f}x")
    
    return final_kspace


def reconstruct_with_mlp(model, kspace_input, mask):
    """Reconstruct with MLP."""
    model.eval()
    H, W = kspace_input.shape
    
    with torch.no_grad():
        patches = extract_patches_masked(kspace_input)
        recon_flat = model(patches).squeeze()
        recon_grid = recon_flat.view(H, W)
        final_kspace = kspace_input + (recon_grid * (1 - mask))
    
    return final_kspace


def grappa_baseline(kspace_acs, kspace_input, mask, kernel_size=3):
    """Classical GRAPPA reconstruction."""
    patches_acs = extract_patches_masked(kspace_acs, kernel_size)
    targets_acs = kspace_acs.flatten()
    
    try:
        W = torch.linalg.lstsq(patches_acs, targets_acs).solution
    except:
        W = torch.linalg.pinv(patches_acs) @ targets_acs
    
    patches_full = extract_patches_masked(kspace_input, kernel_size)
    recon_flat = patches_full @ W
    recon_grid = recon_flat.view(kspace_input.shape)
    
    final_kspace = kspace_input + (recon_grid * (1 - mask))
    return final_kspace


# ==========================================
# FASTMRI DATA LOADER
# ==========================================

def load_fastmri_slice(filepath, slice_idx=None):
    """Load a slice from fastMRI h5 file."""
    with h5py.File(filepath, 'r') as f:
        kspace_data = f['kspace'][:]
        
        # Select middle slice if not specified
        if slice_idx is None:
            slice_idx = kspace_data.shape[0] // 2
        
        kspace = kspace_data[slice_idx]
        
        # Handle multi-coil (take first coil for simplicity)
        if kspace.ndim == 3:
            kspace = kspace[0]
    
    return torch.tensor(kspace, dtype=torch.cfloat).to(device)


def create_cartesian_mask(shape, acceleration=2, acs_lines=24):
    """Create Cartesian undersampling mask."""
    H, W = shape
    mask = torch.zeros((H, W), dtype=torch.cfloat, device=device)
    
    center = H // 2
    
    # ACS lines (fully sampled center)
    mask[center - acs_lines//2:center + acs_lines//2, :] = 1.0
    
    # Random peripheral lines
    idx_list = list(range(0, center - acs_lines//2)) + list(range(center + acs_lines//2, H))
    torch.manual_seed(42)
    perm = torch.randperm(len(idx_list))
    keep = (H // acceleration) - acs_lines
    
    for i in perm[:keep]:
        mask[idx_list[i], :] = 1.0
    
    return mask


# ==========================================
# EVALUATION METRICS
# ==========================================

def get_metrics(gt, recon, data_range):
    """Compute SSIM and phase error."""
    s = ssim(np.abs(gt), np.abs(recon), data_range=data_range)
    p = np.sum(np.abs(np.angle(gt) - np.angle(recon)) * np.abs(gt)) / np.sum(np.abs(gt))
    return s, p


# ==========================================
# MAIN BENCHMARK
# ==========================================

def run_benchmark(filepath, slice_idx=None):
    """Run complete benchmark on one fastMRI file."""
    print(f"\nLoading FastMRI File: {filepath}...")
    
    # 1. Load data
    kspace_gt = load_fastmri_slice(filepath, slice_idx)
    print(f"Slice Shape: {kspace_gt.shape}")
    
    # 2. Create mask and undersampled k-space
    mask = create_cartesian_mask(kspace_gt.shape, acceleration=2, acs_lines=24)
    kspace_input = kspace_gt * mask
    
    # 3. Extract ACS calibration data
    center = kspace_gt.shape[0] // 2
    kspace_acs = kspace_gt[center-12:center+12, :]
    
    # 4. Create training data
    patches_acs = extract_patches_masked(kspace_acs)
    targets_acs = kspace_acs.flatten()
    
    # 5. Initialize models
    print("\n" + "="*60)
    kan_model = StableComplexRBFKAN(
        in_dim=9, hidden_dim=16, out_dim=1, num_rbfs=32, 
        init_data=patches_acs
    ).to(device)
    
    mlp_model = BaselineMLP(kernel_size=3, hidden_dim=64).to(device)
    
    print(f"Params: KAN={count_parameters(kan_model)} | MLP={count_parameters(mlp_model)}")
    
    # 6. Train KAN
    print("\n--- Training Stable KAN ---")
    kan_model = train_stable_kan(kan_model, patches_acs, targets_acs, epochs=1000, lr=0.001)
    kan_kspace = reconstruct_with_kan(kan_model, kspace_input, mask)
    
    # 7. Train MLP
    print("\n--- Training ReLU-MLP ---")
    mlp_model = train_mlp(mlp_model, patches_acs, targets_acs, epochs=1000, lr=0.005)
    mlp_kspace = reconstruct_with_mlp(mlp_model, kspace_input, mask)
    
    # 8. GRAPPA
    print("\n--- Running Classical GRAPPA ---")
    grappa_kspace = grappa_baseline(kspace_acs, kspace_input, mask)
    
    # 9. Convert to images
    gt_c = np.fft.ifft2(np.fft.ifftshift(kspace_gt.cpu().numpy()))
    kan_c = np.fft.ifft2(np.fft.ifftshift(kan_kspace.cpu().numpy()))
    mlp_c = np.fft.ifft2(np.fft.ifftshift(mlp_kspace.cpu().numpy()))
    grappa_c = np.fft.ifft2(np.fft.ifftshift(grappa_kspace.cpu().numpy()))
    zf_c = np.fft.ifft2(np.fft.ifftshift(kspace_input.cpu().numpy()))
    
    # 10. Compute metrics
    plot_max = np.max(np.abs(gt_c))
    print(f"Plotting Max Intensity: {plot_max:.5f}")
    
    g_ssim, g_phase = get_metrics(gt_c, grappa_c, plot_max)
    k_ssim, k_phase = get_metrics(gt_c, kan_c, plot_max)
    m_ssim, m_phase = get_metrics(gt_c, mlp_c, plot_max)
    zf_ssim, zf_phase = get_metrics(gt_c, zf_c, plot_max)
    
    # 11. Sanity checks
    gt_ssim_check, gt_phase_check = get_metrics(gt_c, gt_c, plot_max)
    
    print(f"\n{'='*60}")
    print("RESULTS:")
    print(f"{'='*60}")
    print(f"Zero-filled: SSIM={zf_ssim:.3f}, Phase={zf_phase:.4f}")
    print(f"GRAPPA:      SSIM={g_ssim:.3f}, Phase={g_phase:.4f}")
    print(f"KAN:         SSIM={k_ssim:.3f}, Phase={k_phase:.4f}")
    print(f"MLP:         SSIM={m_ssim:.3f}, Phase={m_phase:.4f}")
    print(f"\nSanity Check:")
    print(f"GT vs GT:    SSIM={gt_ssim_check:.3f}, Phase={gt_phase_check:.4f}")
    
    # Assertions
    assert gt_ssim_check > 0.999, "GT vs GT should be 1.0!"
    assert gt_phase_check < 0.001, "GT vs GT phase should be 0.0!"
    
    if g_ssim < zf_ssim:
        print(f"\nWARNING: GRAPPA ({g_ssim:.3f}) worse than zero-filled ({zf_ssim:.3f})")
        print("This can happen at low acceleration (R=2) - GRAPPA adds high-freq noise")
    
    # 12. Visualize
    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(2, 5)
    
    # Row 1: Magnitude
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(np.abs(gt_c), cmap='gray', vmin=0, vmax=plot_max)
    ax1.set_title("Ground Truth")
    ax1.axis('off')
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(np.abs(zf_c), cmap='gray', vmin=0, vmax=plot_max)
    ax2.set_title(f"Zero-Filled\nSSIM: {zf_ssim:.3f}")
    ax2.axis('off')
    
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(np.abs(grappa_c), cmap='gray', vmin=0, vmax=plot_max)
    ax3.set_title(f"GRAPPA\nSSIM: {g_ssim:.3f}")
    ax3.axis('off')
    
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.imshow(np.abs(kan_c), cmap='gray', vmin=0, vmax=plot_max)
    ax4.set_title(f"KAN\nSSIM: {k_ssim:.3f}")
    ax4.axis('off')
    
    ax5 = fig.add_subplot(gs[0, 4])
    ax5.imshow(np.abs(mlp_c), cmap='gray', vmin=0, vmax=plot_max)
    ax5.set_title(f"MLP\nSSIM: {m_ssim:.3f}")
    ax5.axis('off')
    
    # Row 2: Phase error
    ax6 = fig.add_subplot(gs[1, 0])
    ax6.imshow(np.angle(gt_c), cmap='hsv')
    ax6.set_title("GT Phase")
    ax6.axis('off')
    
    ax7 = fig.add_subplot(gs[1, 1])
    ax7.imshow(np.abs(np.angle(gt_c) - np.angle(zf_c)), cmap='twilight', vmin=0, vmax=3.14)
    ax7.set_title(f"ZF Phase Err: {zf_phase:.3f}")
    ax7.axis('off')
    
    ax8 = fig.add_subplot(gs[1, 2])
    ax8.imshow(np.abs(np.angle(gt_c) - np.angle(grappa_c)), cmap='twilight', vmin=0, vmax=3.14)
    ax8.set_title(f"GRAPPA Phase Err: {g_phase:.3f}")
    ax8.axis('off')
    
    ax9 = fig.add_subplot(gs[1, 3])
    ax9.imshow(np.abs(np.angle(gt_c) - np.angle(kan_c)), cmap='twilight', vmin=0, vmax=3.14)
    ax9.set_title(f"KAN Phase Err: {k_phase:.3f}")
    ax9.axis('off')
    
    ax10 = fig.add_subplot(gs[1, 4])
    ax10.imshow(np.abs(np.angle(gt_c) - np.angle(mlp_c)), cmap='twilight', vmin=0, vmax=3.14)
    ax10.set_title(f"MLP Phase Err: {m_phase:.3f}")
    ax10.axis('off')
    
    plt.tight_layout()
    plt.savefig('reconstruction_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return {
        'grappa_ssim': g_ssim, 'kan_ssim': k_ssim, 'mlp_ssim': m_ssim,
        'grappa_phase': g_phase, 'kan_phase': k_phase, 'mlp_phase': m_phase
    }


# ==========================================
# ENTRY POINT
# ==========================================

if __name__ == "__main__":
    # Replace with your fastMRI file path
    filepath = "./data/fastMRI/singlecoil_val/file1000291.h5"  # Change this to your file
    
    results = run_benchmark(filepath, slice_idx=None)
    
    print("\n" + "="*60)
    print("BENCHMARK COMPLETE")
    print("="*60)