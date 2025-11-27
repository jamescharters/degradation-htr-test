import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Running on: {device}")

# ==========================================
# 1. Models
# ==========================================
class ComplexVectorRBFLayer(nn.Module):
    def __init__(self, in_features, out_features, num_rbfs=32, init_data=None):
        super().__init__()
        self.in_features = in_features
        self.num_rbfs = num_rbfs
        
        # --- INIT CENTERS ---
        if init_data is not None:
            # Randomly sample real data patches to be the centers
            indices = torch.randperm(init_data.size(0))[:num_rbfs]
            selected_centers = init_data[indices].clone()
            self.centers = nn.Parameter(selected_centers)
        else:
            self.centers = nn.Parameter(torch.randn(num_rbfs, in_features, dtype=torch.cfloat))

        # --- FIX: GAMMA SCALING ---
        # We are computing distance across 'in_features' dimensions (9 pixels).
        # The sum of squared errors will be large (~1.0).
        # If Gamma is 1.0 (exp(0)), then exp(-1.0 * 1.0) = 0.36 (Good).
        # Previous run had Gamma ~ 2.7, so exp(-2.7) = 0.06 (Too weak).
        # We set log_gamma to -0.5 to keep the RBFs "wide" enough to catch neighbors.
        log_gamma = torch.ones(num_rbfs) * -0.5 
        self.log_gamma = nn.Parameter(log_gamma) 

        self.rbf_weights = nn.Parameter(torch.randn(num_rbfs, out_features, dtype=torch.cfloat) * 0.1)
        self.linear = nn.Linear(in_features, out_features, bias=False, dtype=torch.cfloat)

    def forward(self, z):
        z_expanded = z.unsqueeze(1) 
        mu_expanded = self.centers.unsqueeze(0)
        
        # Distance Squared
        dist_sq = ((z_expanded - mu_expanded) * (z_expanded - mu_expanded).conj()).real.sum(dim=2)
        
        # Gaussian Activation
        gamma = torch.exp(self.log_gamma)
        rbf_response = torch.exp(-gamma * dist_sq)
        
        non_linear = torch.matmul(rbf_response.type(torch.complex64), self.rbf_weights)
        return self.linear(z) + non_linear

class KANGrappaNet(nn.Module):
    def __init__(self, kernel_size=3, num_rbfs=64, init_data=None):
        super().__init__()
        self.in_features = kernel_size * kernel_size
        
        self.layer1 = ComplexVectorRBFLayer(self.in_features, 16, num_rbfs=num_rbfs, init_data=init_data)
        
        # Project init data to initialize Layer 2
        with torch.no_grad():
            if init_data is not None:
                l1_out = self.layer1(init_data[:500])
                l2_init = torch.tanh(l1_out.real) + 1j * torch.tanh(l1_out.imag)
            else:
                l2_init = None
                
        self.layer2 = ComplexVectorRBFLayer(16, 1, num_rbfs=num_rbfs//2, init_data=l2_init)
        
    def forward(self, x):
        x = self.layer1(x)
        x = torch.tanh(x.real) + 1j * torch.tanh(x.imag)
        x = self.layer2(x)
        return x

class BaselineMLP(nn.Module):
    def __init__(self, kernel_size=3, hidden_dim=64):
        super().__init__()
        # Complex Linear -> Complex ReLU -> Complex Linear
        self.net = nn.Sequential(
            nn.Linear(kernel_size*kernel_size, hidden_dim, dtype=torch.cfloat),
            nn.ReLU(), # PyTorch now supports ReLU on complex (applies to mag, or re/im depending on version. We define manual below to be safe)
            nn.Linear(hidden_dim, 1, dtype=torch.cfloat)
        )
    def forward(self, x):
        # Manual CReLU to ensure fairness (ReLU on Re/Im separately)
        # Note: PyTorch nn.Sequential might struggle with custom complex ops, implementing manually:
        h = self.net[0](x)
        h = torch.nn.functional.relu(h.real) + 1j * torch.nn.functional.relu(h.imag)
        return self.net[2](h)

def count_parameters(model):
    total = 0
    for p in model.parameters():
        mult = 2 if p.is_complex() else 1
        total += p.numel() * mult
    return total

def calculate_metrics(gt, recon):
    gt_mag = np.abs(gt)
    recon_mag = np.abs(recon)
    d_max = 1.0 # Normalized data
    val_ssim = ssim(gt_mag, recon_mag, data_range=d_max)
    
    phase_diff = np.abs(np.angle(gt) - np.angle(recon))
    phase_diff = np.minimum(phase_diff, 2*np.pi - phase_diff)
    weighted_phase_err = np.sum(phase_diff * gt_mag) / (np.sum(gt_mag) + 1e-8)
    return val_ssim, weighted_phase_err

# ==========================================
# 2. Physics & Training
# ==========================================
def generate_physics_phantom(size=160):
    print("Generating High-Freq Phase Phantom...")
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y)
    
    anatomy = np.zeros((size, size))
    anatomy += 1.0 * (X**2 + Y**2 < 0.8)
    anatomy -= 0.5 * (X**2 + Y**2 < 0.7)
    anatomy += 0.8 * ((X-0.2)**2 + (Y-0.2)**2 < 0.1)
    
    coil_map = np.exp(-0.5 * ((X-0.5)**2 + (Y-0.5)**2))
    
    # The KAN Advantage: Smooth Quadratic Phase
    # The MLP Advantage: Piecewise linear functions struggle with smooth curves
    phase_map = np.exp(1j * (4.0 * (X**2 + Y**2))) 
    
    img_complex = anatomy * coil_map * phase_map
    kspace_gt = np.fft.fftshift(np.fft.fft2(img_complex))
    
    norm_factor = np.abs(kspace_gt).max() + 1e-8
    kspace_gt = torch.tensor(kspace_gt / norm_factor, dtype=torch.cfloat).to(device)
    
    return kspace_gt

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

def get_data_phantom(accel=2):
    kspace_gt = generate_physics_phantom(size=160)
    H, W = kspace_gt.shape
    
    torch.manual_seed(42)
    mask = torch.zeros_like(kspace_gt)
    center_y = H // 2
    mask[center_y-12:center_y+12, :] = 1.0 
    
    idx_list = list(range(0, center_y-12)) + list(range(center_y+12, H))
    perm = torch.randperm(len(idx_list))
    keep = (H // accel) - 24
    for i in perm[:keep]: mask[idx_list[i], :] = 1.0
    
    kspace_input = kspace_gt * mask
    kspace_acs = kspace_gt[center_y-12:center_y+12, :]
    return kspace_gt, kspace_input, kspace_acs, mask

def train_model(model, name, X_train, Y_train, kspace_input, mask, epochs=1500):
    print(f"\n--- Training {name} ---")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    loss_hist = []
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = model(X_train)
        
        diff = pred.squeeze() - Y_train
        # L1 Loss is better for Image Quality
        loss = torch.abs(diff).mean() + 0.1 * (torch.angle(pred.squeeze()) - torch.angle(Y_train)).abs().mean()
        
        loss.backward()
        optimizer.step()
        loss_hist.append(loss.item())
        if epoch % 500 == 0: print(f"Epoch {epoch} | Loss: {loss.item():.5f}")

    model.eval()
    with torch.no_grad():
        patches = extract_patches_masked(kspace_input)
        recon_flat = model(patches).squeeze()
        recon_grid = recon_flat.view(kspace_input.shape)
        final_kspace = kspace_input + (recon_grid * (1 - mask))
    return final_kspace

def run_benchmark():
    kspace_gt, kspace_input, kspace_acs, mask = get_data_phantom()
    patches_acs = extract_patches_masked(kspace_acs)
    targets_acs = kspace_acs.flatten()
    
    # Models
    kan_model = KANGrappaNet(init_data=patches_acs).to(device)
    kan_params = count_parameters(kan_model)
    
    target_dim = 10
    best_diff = 999999
    best_dim = 10
    for h in range(10, 200):
        temp = BaselineMLP(hidden_dim=h)
        p = count_parameters(temp)
        if abs(p - kan_params) < best_diff: best_diff = abs(p - kan_params); best_dim = h
            
    mlp_model = BaselineMLP(hidden_dim=best_dim).to(device)
    mlp_params = count_parameters(mlp_model)
    
    print(f"Parameters: KAN={kan_params} | MLP={mlp_params}")
    
    kan_kspace = train_model(kan_model, "RBF-KAN", patches_acs, targets_acs, kspace_input, mask)
    mlp_kspace = train_model(mlp_model, "ReLU-MLP", patches_acs, targets_acs, kspace_input, mask)
    
    # --- METRICS & SCALING FIX ---
    # Convert to Image Space
    gt_c = np.fft.ifft2(np.fft.ifftshift(kspace_gt.cpu().numpy()))
    kan_c = np.fft.ifft2(np.fft.ifftshift(kan_kspace.cpu().numpy()))
    mlp_c = np.fft.ifft2(np.fft.ifftshift(mlp_kspace.cpu().numpy()))
    
    # Determine Dynamic Range from Ground Truth
    gt_mag = np.abs(gt_c)
    plot_max = np.max(gt_mag) # This will be small, e.g., 0.005
    print(f"Image Space Max Value: {plot_max:.5f}")
    
    # Calculate SSIM using the correct data range
    def calc_metrics_correct(gt, recon, d_max):
        gt_mag = np.abs(gt)
        recon_mag = np.abs(recon)
        s = ssim(gt_mag, recon_mag, data_range=d_max)
        
        p_diff = np.abs(np.angle(gt) - np.angle(recon))
        p_diff = np.minimum(p_diff, 2*np.pi - p_diff)
        # Weighted by magnitude to ignore background noise phase
        p_err = np.sum(p_diff * gt_mag) / (np.sum(gt_mag) + 1e-12)
        return s, p_err

    k_ssim, k_phase = calc_metrics_correct(gt_c, kan_c, plot_max)
    m_ssim, m_phase = calc_metrics_correct(gt_c, mlp_c, plot_max)
    
    print("\n=== FINAL RESULTS ===")
    print(f"KAN -> SSIM: {k_ssim:.4f} | Phase Err: {k_phase:.4f}")
    print(f"MLP -> SSIM: {m_ssim:.4f} | Phase Err: {m_phase:.4f}")

    # --- PLOTTING ---
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 4)
    
    # Row 1: GT
    ax1 = fig.add_subplot(gs[0, 0]); 
    ax1.imshow(np.abs(gt_c), cmap='gray', vmin=0, vmax=plot_max) # Scale to Data
    ax1.set_title("Ground Truth")
    
    ax2 = fig.add_subplot(gs[0, 1]); 
    ax2.imshow(np.angle(gt_c), cmap='hsv')
    ax2.set_title("Complex Phase")
    
    # Row 2: KAN
    ax3 = fig.add_subplot(gs[1, 0]); 
    ax3.imshow(np.abs(kan_c), cmap='gray', vmin=0, vmax=plot_max)
    ax3.set_title(f"KAN Recon\nSSIM: {k_ssim:.3f}")
    
    ax_p1 = fig.add_subplot(gs[1, 1]); 
    # For Phase error, we use a fixed 0-Pi scale
    ax_p1.imshow(np.abs(np.angle(gt_c) - np.angle(kan_c)), cmap='twilight', vmin=0, vmax=3.14)
    ax_p1.set_title(f"KAN Phase Error: {k_phase:.4f}")

    # Row 3: MLP
    ax4 = fig.add_subplot(gs[2, 0]); 
    ax4.imshow(np.abs(mlp_c), cmap='gray', vmin=0, vmax=plot_max)
    ax4.set_title(f"MLP Recon\nSSIM: {m_ssim:.3f}")
    
    ax_p2 = fig.add_subplot(gs[2, 1]); 
    ax_p2.imshow(np.abs(np.angle(gt_c) - np.angle(mlp_c)), cmap='twilight', vmin=0, vmax=3.14)
    ax_p2.set_title(f"MLP Phase Error: {m_phase:.4f}")

    # Brain Plot
    ax7 = fig.add_subplot(gs[0:, 2:])
    centers = kan_model.layer1.centers.detach().cpu().numpy()[:, 0]
    gammas = kan_model.layer1.log_gamma.exp().detach().cpu().numpy()
    sizes = 20 / (gammas + 0.1)
    
    ref_data = patches_acs.detach().cpu().numpy()[:, 0]
    np.random.shuffle(ref_data)
    
    ax7.scatter(ref_data[:1000].real, ref_data[:1000].imag, s=1, c='gray', alpha=0.3, label='Data')
    ax7.scatter(centers.real, centers.imag, s=sizes, c='red', alpha=0.6, edgecolors='black', label='Centers')
    ax7.set_title("Feature Discovery")
    ax7.legend()
    ax7.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_benchmark()