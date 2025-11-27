import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import h5py
import glob
import random
from skimage.metrics import structural_similarity as ssim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Running on: {device}")








import torch
import torch.nn as nn
import numpy as np

# ==========================================
# CLEAN KAN IMPLEMENTATION FROM SCRATCH
# ==========================================

class StableComplexRBFKAN(nn.Module):
    """
    Stable Radial Basis Function Kolmogorov-Arnold Network for complex k-space.
    
    Key stability features:
    - Conservative initialization
    - Gradient clipping built-in
    - Parameter constraints
    - Output normalization
    """
    def __init__(self, in_dim=9, hidden_dim=16, out_dim=1, num_rbfs=32, init_data=None):
        super().__init__()
        
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_rbfs = num_rbfs
        
        # ========================================
        # Layer 1: Input -> Hidden (with RBFs)
        # ========================================
        
        # Linear component (small init)
        self.linear1 = nn.Linear(in_dim, hidden_dim, bias=True, dtype=torch.cfloat)
        nn.init.xavier_uniform_(self.linear1.weight, gain=0.1)
        nn.init.zeros_(self.linear1.bias)
        
        # RBF Centers - data-driven if available
        if init_data is not None and len(init_data) >= num_rbfs:
            # Sample from actual data distribution
            indices = torch.randperm(len(init_data))[:num_rbfs]
            centers_init = init_data[indices].clone()
        else:
            # Small random initialization
            centers_init = torch.randn(num_rbfs, in_dim, dtype=torch.cfloat) * 0.01
        
        self.rbf_centers1 = nn.Parameter(centers_init)
        
        # RBF Bandwidths (log-space for positivity, conservative init)
        # log_gamma = 0 means gamma = 1, which is reasonable
        self.log_gamma1 = nn.Parameter(torch.zeros(num_rbfs))
        
        # RBF Weights (very small init - RBFs should be weak initially)
        self.rbf_weights1 = nn.Parameter(
            torch.randn(num_rbfs, hidden_dim, dtype=torch.cfloat) * 0.01
        )
        
        # ========================================
        # Layer 2: Hidden -> Output (with RBFs)
        # ========================================
        
        # Linear component
        self.linear2 = nn.Linear(hidden_dim, out_dim, bias=True, dtype=torch.cfloat)
        nn.init.xavier_uniform_(self.linear2.weight, gain=0.1)
        nn.init.zeros_(self.linear2.bias)
        
        # Smaller RBF layer for output (half the RBFs)
        self.rbf_centers2 = nn.Parameter(
            torch.randn(num_rbfs // 2, hidden_dim, dtype=torch.cfloat) * 0.01
        )
        self.log_gamma2 = nn.Parameter(torch.zeros(num_rbfs // 2))
        self.rbf_weights2 = nn.Parameter(
            torch.randn(num_rbfs // 2, out_dim, dtype=torch.cfloat) * 0.01
        )
        
        # Learnable scaling factors for RBF contributions
        self.rbf_scale1 = nn.Parameter(torch.tensor(0.1))
        self.rbf_scale2 = nn.Parameter(torch.tensor(0.1))
        
    def complex_rbf(self, x, centers, log_gamma, weights, scale):
        """
        Compute RBF activations for complex inputs.
        
        Args:
            x: [batch, in_dim] complex tensor
            centers: [num_rbfs, in_dim] complex tensor
            log_gamma: [num_rbfs] real tensor (bandwidths)
            weights: [num_rbfs, out_dim] complex tensor
            scale: scalar - overall scaling factor
            
        Returns:
            [batch, out_dim] complex tensor
        """
        # Expand dimensions for broadcasting
        x_exp = x.unsqueeze(1)  # [batch, 1, in_dim]
        c_exp = centers.unsqueeze(0)  # [1, num_rbfs, in_dim]
        
        # Complex squared distance (real-valued result)
        diff = x_exp - c_exp
        dist_sq = (diff * diff.conj()).real.sum(dim=2)  # [batch, num_rbfs]
        
        # Constrained gamma (prevent collapse or explosion)
        gamma = torch.exp(log_gamma).clamp(min=0.01, max=10.0)
        
        # RBF activations (real-valued Gaussians)
        rbf_response = torch.exp(-gamma * dist_sq)  # [batch, num_rbfs]
        
        # Safety clamp (prevent numerical issues)
        rbf_response = rbf_response.clamp(max=10.0)
        
        # Weighted sum (complex output)
        # Convert to complex64 for matmul compatibility
        output = torch.matmul(rbf_response.type(torch.complex64), weights)
        
        # Apply learned scaling
        return scale * output
    
    def complex_activation(self, x):
        """
        Complex activation: separate tanh on real and imaginary parts.
        Keeps magnitude bounded.
        """
        return torch.tanh(x.real) + 1j * torch.tanh(x.imag)
    
    def forward(self, x):
        """
        Forward pass with stability checks.
        
        Args:
            x: [batch, in_dim] complex tensor
            
        Returns:
            [batch, out_dim] complex tensor
        """
        batch_size = x.shape[0]
        
        # ========================================
        # Layer 1: Input -> Hidden
        # ========================================
        
        # Linear part
        h = self.linear1(x)
        
        # RBF part (additive)
        h_rbf = self.complex_rbf(x, self.rbf_centers1, self.log_gamma1, 
                                  self.rbf_weights1, self.rbf_scale1)
        
        h = h + h_rbf
        
        # Complex activation (bounded)
        h = self.complex_activation(h)
        
        # ========================================
        # Layer 2: Hidden -> Output
        # ========================================
        
        # Linear part
        out = self.linear2(h)
        
        # RBF part
        out_rbf = self.complex_rbf(h, self.rbf_centers2, self.log_gamma2,
                                     self.rbf_weights2, self.rbf_scale2)
        
        out = out + out_rbf
        
        # NO final activation - we want full complex range for k-space
        
        return out
    
    def get_parameters_status(self):
        """Diagnostic: check if any parameters are exploding."""
        status = {}
        
        for name, param in self.named_parameters():
            if param.requires_grad:
                param_magnitude = torch.abs(param).max().item()
                status[name] = param_magnitude
                
        return status


# ==========================================
# TRAINING FUNCTION WITH MONITORING
# ==========================================

def train_stable_kan(model, X_train, Y_train, epochs=1000, lr=0.001, verbose=True):
    """
    Train KAN with extensive stability monitoring.
    
    Args:
        model: StableComplexRBFKAN instance
        X_train: [N, in_dim] complex training inputs (k-space patches)
        Y_train: [N] complex training targets (center k-space values)
        epochs: number of training iterations
        lr: learning rate (conservative default)
        verbose: print training progress
        
    Returns:
        trained model, loss_history
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Learning rate scheduler (reduce on plateau)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=100, verbose=False
    )
    
    loss_history = []
    
    model.train()
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Forward pass
        pred = model(X_train).squeeze()
        
        # Loss: L1 magnitude + phase difference
        diff = pred - Y_train
        loss_mag = torch.abs(diff).mean()
        loss_phase = torch.abs(torch.angle(pred) - torch.angle(Y_train)).mean()
        
        loss = loss_mag + 0.1 * loss_phase
        
        # Backward pass
        loss.backward()
        
        # CRITICAL: Gradient clipping to prevent explosion
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step(loss)
        
        loss_history.append(loss.item())
        
        # ========================================
        # Monitoring (every 100 epochs)
        # ========================================
        if epoch % 100 == 0 and verbose:
            with torch.no_grad():
                # Check output magnitude
                pred_mag = torch.abs(pred).mean().item()
                target_mag = torch.abs(Y_train).mean().item()
                mag_ratio = pred_mag / target_mag if target_mag > 0 else float('inf')
                
                print(f"Epoch {epoch:4d} | Loss: {loss.item():.5f} | "
                      f"Mag Ratio: {mag_ratio:.2f}x | LR: {optimizer.param_groups[0]['lr']:.1e}")
                
                # Check for explosion
                if mag_ratio > 10.0:
                    print(f"  WARNING: Output magnitude is {mag_ratio:.1f}x target!")
                
                # Check parameter health
                param_status = model.get_parameters_status()
                max_param = max(param_status.values())
                if max_param > 100:
                    print(f"  WARNING: Max parameter magnitude: {max_param:.2e}")
    
    if verbose:
        print(f"Training complete. Final loss: {loss_history[-1]:.5f}")
    
    return model, loss_history


# ==========================================
# INFERENCE WITH OUTPUT NORMALIZATION
# ==========================================

def reconstruct_with_kan(model, kspace_input, mask, kernel_size=3):
    """
    Reconstruct missing k-space using trained KAN.
    
    Args:
        model: trained StableComplexRBFKAN
        kspace_input: [H, W] complex undersampled k-space
        mask: [H, W] binary mask (1=sampled, 0=missing)
        kernel_size: patch size (default 3x3=9)
        
    Returns:
        reconstructed k-space [H, W] complex tensor
    """
    model.eval()
    
    H, W = kspace_input.shape
    device = kspace_input.device
    
    with torch.no_grad():
        # Extract patches with center pixel masked
        patches = extract_patches_masked(kspace_input, kernel_size)
        
        # Forward pass
        recon_flat = model(patches).squeeze()
        
        # ========================================
        # CRITICAL: Output normalization
        # ========================================
        
        # Compute expected magnitude from input
        input_nonzero = kspace_input[kspace_input.abs() > 1e-10]
        if len(input_nonzero) > 0:
            target_magnitude = input_nonzero.abs().mean()
            recon_magnitude = recon_flat.abs().mean()
            
            # If output magnitude is unreasonable, scale it
            if recon_magnitude > 3 * target_magnitude:
                scale_factor = target_magnitude / recon_magnitude
                print(f"  Scaling KAN output by {scale_factor:.3f} "
                      f"(was {recon_magnitude:.2e}, target {target_magnitude:.2e})")
                recon_flat = recon_flat * scale_factor
        
        # Reshape to k-space grid
        recon_grid = recon_flat.view(H, W)
        
        # Fill in missing k-space lines
        final_kspace = kspace_input + (recon_grid * (1 - mask))
        
        # ========================================
        # Sanity check
        # ========================================
        output_energy = torch.abs(final_kspace).sum()
        input_energy = torch.abs(kspace_input).sum()
        energy_ratio = output_energy / input_energy
        
        print(f"  KAN k-space energy ratio: {energy_ratio:.2f}x")
        
        if energy_ratio > 5.0:
            print(f"  ERROR: KAN output has {energy_ratio:.1f}x more energy than input!")
            print(f"  This indicates training instability. Results are unreliable.")
    
    return final_kspace


# ==========================================
# HELPER: EXTRACT PATCHES (same as before)
# ==========================================

def extract_patches_masked(kspace, kernel_size=3):
    """
    Extract overlapping patches from k-space with center pixel zeroed.
    
    Args:
        kspace: [H, W] complex k-space
        kernel_size: patch size (must be odd)
        
    Returns:
        [N, kernel_size^2] complex patches with center=0
    """
    device = kspace.device
    
    # Use unfold to extract patches
    k_real = kspace.real.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    k_imag = kspace.imag.unsqueeze(0).unsqueeze(0)
    
    pad = kernel_size // 2
    unfold = nn.Unfold(kernel_size=(kernel_size, kernel_size), padding=pad)
    
    patches_real = unfold(k_real)  # [1, kernel_size^2, N]
    patches_imag = unfold(k_imag)
    
    # Combine into complex patches
    patches = torch.complex(patches_real, patches_imag)
    patches = patches.permute(0, 2, 1).squeeze(0)  # [N, kernel_size^2]
    
    # Zero out center pixel (this is what we're trying to predict)
    center_idx = (kernel_size * kernel_size) // 2
    patches[:, center_idx] = 0 + 0j
    
    return patches


# ==========================================
# EXAMPLE USAGE
# ==========================================

def example_training_pipeline(kspace_gt, kspace_input, mask):
    """
    Complete example of training and using the stable KAN.
    
    Args:
        kspace_gt: [H, W] ground truth k-space
        kspace_input: [H, W] undersampled k-space
        mask: [H, W] sampling mask
        
    Returns:
        reconstructed k-space
    """
    print("=" * 60)
    print("STABLE KAN TRAINING PIPELINE")
    print("=" * 60)
    
    # 1. Extract calibration data (ACS region)
    H, W = kspace_gt.shape
    center = H // 2
    kspace_acs = kspace_gt[center-12:center+12, :]
    
    print(f"\n1. Extracting calibration data:")
    print(f"   Full k-space: {kspace_gt.shape}")
    print(f"   ACS region: {kspace_acs.shape} (24 lines)")
    
    # 2. Create training data from ACS
    patches_acs = extract_patches_masked(kspace_acs, kernel_size=3)
    targets_acs = kspace_acs.flatten()
    
    print(f"\n2. Creating training data:")
    print(f"   Patches: {patches_acs.shape}")
    print(f"   Targets: {targets_acs.shape}")
    
    # 3. Initialize model with data-driven RBF centers
    print(f"\n3. Initializing model...")
    model = StableComplexRBFKAN(
        in_dim=9,           # 3x3 patches
        hidden_dim=16,
        out_dim=1,
        num_rbfs=32,
        init_data=patches_acs  # Data-driven initialization
    )
    
    param_count = sum(p.numel() * (2 if p.is_complex() else 1) 
                      for p in model.parameters())
    print(f"   Parameters: {param_count}")
    
    # 4. Train model
    print(f"\n4. Training model...")
    model, loss_history = train_stable_kan(
        model, patches_acs, targets_acs,
        epochs=1000,
        lr=0.001,
        verbose=True
    )
    
    # 5. Reconstruct full k-space
    print(f"\n5. Reconstructing k-space...")
    recon_kspace = reconstruct_with_kan(model, kspace_input, mask, kernel_size=3)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    
    return recon_kspace, model, loss_history


# ==========================================
# COMPARISON FUNCTION
# ==========================================

def compare_methods(kspace_gt, kspace_input, mask):
    """
    Compare KAN vs GRAPPA vs MLP on same data.
    
    Returns dictionary with reconstructed k-spaces for each method.
    """
    results = {}
    
    # 1. Zero-filled (baseline)
    results['zero_filled'] = kspace_input.clone()
    
    # 2. GRAPPA (classical)
    print("\n" + "=" * 60)
    print("CLASSICAL GRAPPA")
    print("=" * 60)
    # (use your existing grappa_baseline function)
    
    # 3. Stable KAN (new implementation)
    print("\n" + "=" * 60)
    print("STABLE KAN")
    print("=" * 60)
    kan_kspace, kan_model, kan_loss = example_training_pipeline(
        kspace_gt, kspace_input, mask
    )
    results['kan'] = kan_kspace
    
    # 4. MLP (for comparison)
    # (use your existing BaselineMLP)
    
    return results

















# ==========================================
# 1. KAN / MLP MODELS (Optimized)
# ==========================================
class ComplexVectorRBFLayer(nn.Module):
    def __init__(self, in_features, out_features, num_rbfs=32, init_data=None):
        super().__init__()
        self.in_features = in_features
        self.num_rbfs = num_rbfs
        if init_data is not None:
            indices = torch.randperm(init_data.size(0))[:num_rbfs]
            self.centers = nn.Parameter(init_data[indices].clone())
        else:
            self.centers = nn.Parameter(torch.randn(num_rbfs, in_features, dtype=torch.cfloat))

        # self.log_gamma = nn.Parameter(torch.ones(num_rbfs) * -0.5) 
        # self.rbf_weights = nn.Parameter(torch.randn(num_rbfs, out_features, dtype=torch.cfloat) * 0.1)

        # CRITICAL: Start with reasonable bandwidth
        self.log_gamma = nn.Parameter(torch.ones(num_rbfs) * 0.0)  # Was -0.5, now 0.0
        
        # CRITICAL: Smaller weight initialization
        self.rbf_weights = nn.Parameter(torch.randn(num_rbfs, out_features, dtype=torch.cfloat) * 0.01)  # Was 0.1, now 0.01
        

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
# 2. FastMRI DATA LOADER
# ==========================================
def load_fastmri_file(folder_path):
    # Find all h5 files
    files = glob.glob(os.path.join(folder_path, "*.h5"))
    if not files:
        raise ValueError(f"No .h5 files found in {folder_path}")
    
    # Pick a random one
    filename = random.choice(files)
    print(f"Loading FastMRI File: {os.path.basename(filename)}...")
    
    with h5py.File(filename, 'r') as f:
        # FastMRI SingleCoil has key 'kspace' -> Shape [Slices, Height, Width]
        if 'kspace' not in f:
            raise ValueError("File does not contain 'kspace' key.")
        
        kspace_vol = f['kspace'][()]
        
        # Pick the middle slice (usually the best anatomy)
        num_slices = kspace_vol.shape[0]
        mid_slice = num_slices // 2
        kspace_slice = kspace_vol[mid_slice]
        
    print(f"Slice Shape: {kspace_slice.shape}")
    
    # FastMRI data is not centered. FFTShift it.
    # Note: FastMRI kspace is usually already FFT-shifted in storage, 
    # but sometimes needs IFFTshift depending on the viewer.
    # We will assume standard centering.
    
    # Normalize
    norm_factor = np.abs(kspace_slice).max() + 1e-8
    kspace_slice = kspace_slice / norm_factor
    
    return torch.tensor(kspace_slice, dtype=torch.cfloat).to(device)

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

# ==========================================
# 3. BENCHMARK
# ==========================================
def train_model(model, name, X_train, Y_train, kspace_input, mask, epochs=1000):
    print(f"\n--- Training {name} ---")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    # ADD GRADIENT CLIPPING
    max_grad_norm = 0.1

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = model(X_train)
        diff = pred.squeeze() - Y_train
        loss = torch.abs(diff).mean() + 0.1 * (torch.angle(pred.squeeze()) - torch.angle(Y_train)).abs().mean()
        loss.backward()

        # CRITICAL: Clip gradients to prevent explosion
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        # MONITOR FOR EXPLOSION EVERY 100 EPOCHS
        if epoch % 100 == 0:
            with torch.no_grad():
                test_out = model(X_train[:10])
                out_magnitude = torch.abs(test_out).mean()
                in_magnitude = torch.abs(Y_train[:10]).mean()
                ratio = out_magnitude / in_magnitude
                print(f"Epoch {epoch} | Loss: {loss.item():.5f} | Out/In: {ratio:.2f}x")
                
                # STOP TRAINING IF EXPLODING
                if ratio > 5.0:
                    print(f"WARNING: Output exploding at epoch {epoch}! Stopping early.")
                    break

        optimizer.step()
        if epoch % 500 == 0: print(f"Epoch {epoch} | Loss: {loss.item():.5f}")

    model.eval()
    with torch.no_grad():
        patches = extract_patches_masked(kspace_input)
        recon_flat = model(patches).squeeze()
        recon_grid = recon_flat.view(kspace_input.shape)

        # ADD SANITY CHECK HERE
        recon_energy = torch.abs(recon_grid).sum()
        input_energy = torch.abs(kspace_input).sum()
        print(f"Recon/Input energy ratio: {recon_energy/input_energy:.2f}x")
        
        # If ratio > 5, something is very wrong
        if recon_energy / input_energy > 5.0:
            print("WARNING: Reconstruction energy suspiciously high!")

        final_kspace = kspace_input + (recon_grid * (1 - mask))
    return final_kspace

def grappa_baseline(kspace_acs, kspace_input, mask, kernel_size=3):
    """
    Classical GRAPPA reconstruction (Griswold et al., MRM 2002)
    
    Args:
        kspace_acs: Calibration data (e.g., [24, W] for 24 ACS lines)
        kspace_input: Undersampled k-space [H, W]
        mask: Binary sampling mask [H, W]
        kernel_size: GRAPPA kernel size (typically 3 or 5)
    
    Returns:
        Reconstructed k-space [H, W]
    """
    # Extract patches from ACS region (same as your neural networks)
    patches_acs = extract_patches_masked(kspace_acs, kernel_size)
    targets_acs = kspace_acs.flatten()
    
    # GRAPPA is just linear regression: find W that minimizes ||patches @ W - targets||²
    # This is equivalent to a neural network with NO activation function
    try:
        # Least squares solution
        W = torch.linalg.lstsq(patches_acs, targets_acs).solution
    except:
        # If singular, use pseudo-inverse
        W = torch.linalg.pinv(patches_acs) @ targets_acs
    
    # Apply learned weights to full k-space
    patches_full = extract_patches_masked(kspace_input, kernel_size)
    recon_flat = patches_full @ W
    recon_grid = recon_flat.view(kspace_input.shape)
    
    # Fill in missing k-space lines
    final_kspace = kspace_input + (recon_grid * (1 - mask))
    
    return final_kspace


def run_benchmark():
    # UPDATE THIS PATH TO WHERE YOU EXTRACTED THE DATA
    DATA_FOLDER = "./data/fastMRI/singlecoil_val" 
    
    if not os.path.exists(DATA_FOLDER):
        print(f"Folder '{DATA_FOLDER}' not found. Please extract the tar.xz file.")
        return

    kspace_gt = load_fastmri_file(DATA_FOLDER)
    H, W = kspace_gt.shape
    
    # Create Mask (Center 24 lines + Random 2x)
    mask = torch.zeros_like(kspace_gt)
    center_y = H // 2
    mask[center_y-12:center_y+12, :] = 1.0 
    idx_list = list(range(0, center_y-12)) + list(range(center_y+12, H))
    perm = torch.randperm(len(idx_list))
    keep = (H // 2) - 24
    for i in perm[:keep]: mask[idx_list[i], :] = 1.0
    
    kspace_input = kspace_gt * mask
    kspace_acs = kspace_gt[center_y-12:center_y+12, :]
    
    patches_acs = extract_patches_masked(kspace_acs)
    targets_acs = kspace_acs.flatten()
    
    # Models
    # kan_model = KANGrappaNet(init_data=patches_acs).to(device)
    kan_model = StableComplexRBFKAN(
        in_dim=9,
        hidden_dim=16,
        out_dim=1,
        num_rbfs=32,
        init_data=patches_acs  # Data-driven centers
    ).to(device)

    mlp_model = BaselineMLP(hidden_dim=200).to(device)
    
    print(f"Params: KAN={count_parameters(kan_model)} | MLP={count_parameters(mlp_model)}")
    
    # kan_kspace = train_model(kan_model, "RBF-KAN", patches_acs, targets_acs, kspace_input, mask)
    kan_model, loss_history = train_stable_kan(
        kan_model, patches_acs, targets_acs,
        epochs=1000, lr=0.001, verbose=True
    )

    # Reconstruct (replaces your current inference)
    kan_kspace = reconstruct_with_kan(kan_model, kspace_input, mask)

    
    mlp_kspace = train_model(mlp_model, "ReLU-MLP", patches_acs, targets_acs, kspace_input, mask)
    
    print("\n--- Running Classical GRAPPA ---")
    grappa_kspace = grappa_baseline(kspace_acs, kspace_input, mask, kernel_size=3)
    

    # Metrics
    gt_c = np.fft.ifft2(np.fft.ifftshift(kspace_gt.cpu().numpy()))
    kan_c = np.fft.ifft2(np.fft.ifftshift(kan_kspace.cpu().numpy()))
    mlp_c = np.fft.ifft2(np.fft.ifftshift(mlp_kspace.cpu().numpy()))
    grappa_c = np.fft.ifft2(np.fft.ifftshift(grappa_kspace.cpu().numpy()))  # *** ADD THIS ***

    d_max = np.max(np.abs(gt_c))
    print(f"Plotting Max Intensity: {d_max:.5f}")

    def get_metrics(gt, recon):
        s = ssim(np.abs(gt), np.abs(recon), data_range=d_max)
        p = np.sum(np.abs(np.angle(gt)-np.angle(recon)) * np.abs(gt)) / np.sum(np.abs(gt))
        return s, p

    k_ssim, k_phase = get_metrics(gt_c, kan_c)
    m_ssim, m_phase = get_metrics(gt_c, mlp_c)
    g_ssim, g_phase = get_metrics(gt_c, grappa_c)  # *** ADD THIS ***
    
    #print(f"\nRESULTS:\nKAN: SSIM={k_ssim:.3f}, Phase={k_phase:.4f}\nMLP: SSIM={m_ssim:.3f}, Phase={m_phase:.4f}")
    print(f"\nRESULTS:")
    print(f"GRAPPA: SSIM={g_ssim:.3f}, Phase={g_phase:.4f}")  # *** ADD THIS ***
    print(f"KAN:    SSIM={k_ssim:.3f}, Phase={k_phase:.4f}")
    print(f"MLP:    SSIM={m_ssim:.3f}, Phase={m_phase:.4f}")



    def check_kan_sanity(kan_kspace, kspace_gt, mask):
        missing_mask = (torch.abs(1 - mask) > 0.5).bool()
        
        # Check energy in missing regions
        kan_missing_energy = torch.abs(kan_kspace[missing_mask]).sum()
        gt_missing_energy = torch.abs(kspace_gt[missing_mask]).sum()
        
        print(f"\n=== KAN SANITY CHECK ===")
        print(f"KAN energy in missing region: {kan_missing_energy:.2e}")
        print(f"GT energy in missing region:  {gt_missing_energy:.2e}")
        print(f"Ratio: {kan_missing_energy/gt_missing_energy:.2f}x")
        
        # Check for outliers
        kan_max = torch.abs(kan_kspace).max()
        gt_max = torch.abs(kspace_gt).max()
        print(f"KAN max k-space value: {kan_max:.2e}")
        print(f"GT max k-space value:  {gt_max:.2e}")
        
        # This is the smoking gun - KAN should NOT exceed GT significantly
        assert kan_missing_energy < 3 * gt_missing_energy, \
            "KAN is generating way too much energy!"

    check_kan_sanity(kan_kspace, kspace_gt, mask)




    def debug_reconstruction(kspace_gt, kspace_input, kspace_acs, mask, grappa_kspace, kan_kspace):
        """Comprehensive debugging of reconstruction pipeline"""
        
        print("\n=== DEBUGGING RECONSTRUCTION ===")
        
        center = kspace_gt.shape[0] // 2
        
        # 1. Check mask properties
        sampling_ratio = float(mask.sum().real / mask.numel())  # Convert to float
        acs_ratio = float(mask[center-12:center+12, :].sum().real / (24 * mask.shape[1]))
        print(f"\n1. MASK PROPERTIES:")
        print(f"   Mask dtype: {mask.dtype}")  # Check if complex
        print(f"   Sampling ratio: {sampling_ratio:.2%} (should be ~25-50%)")
        print(f"   ACS fully sampled: {acs_ratio:.2%} (should be 100%)")
        print(f"   Mask shape: {mask.shape}")
        
        # 2. Check ACS extraction
        print(f"\n2. ACS EXTRACTION:")
        print(f"   ACS shape: {kspace_acs.shape}")
        
        # 3. Check k-space energy
        input_energy = float(torch.abs(kspace_input).sum())
        grappa_energy = float(torch.abs(grappa_kspace).sum())
        kan_energy = float(torch.abs(kan_kspace).sum())
        gt_energy = float(torch.abs(kspace_gt).sum())
        
        print(f"\n3. K-SPACE ENERGY:")
        print(f"   Input:   {input_energy:.2e} ({input_energy/gt_energy:.1%} of GT)")
        print(f"   GRAPPA:  {grappa_energy:.2e} ({grappa_energy/gt_energy:.1%} of GT)")
        print(f"   KAN:     {kan_energy:.2e} ({kan_energy/gt_energy:.1%} of GT)")
        print(f"   GT:      {gt_energy:.2e}")
        
        # 4. Check actual reconstruction in missing regions
        missing_mask = (torch.abs(1 - mask) > 0.5).bool()  # Use abs for complex mask
        input_missing = float(torch.abs(kspace_input[missing_mask]).mean())
        grappa_missing = float(torch.abs(grappa_kspace[missing_mask]).mean())
        gt_missing = float(torch.abs(kspace_gt[missing_mask]).mean())
        
        print(f"\n4. MISSING REGION RECONSTRUCTION:")
        print(f"   Input (should be ~0):  {input_missing:.2e}")
        print(f"   GRAPPA:                {grappa_missing:.2e}")
        print(f"   GT:                    {gt_missing:.2e}")
        print(f"   GRAPPA/GT ratio:       {grappa_missing/gt_missing:.2%}")
        
        # 5. Verify GRAPPA actually changes something
        grappa_diff = float(torch.abs(grappa_kspace - kspace_input).sum())
        print(f"\n5. RECONSTRUCTION DIFFERENCE:")
        print(f"   ||GRAPPA - Input||₁: {grappa_diff:.2e}")
        print(f"   Changes input: {grappa_diff > 1e-6}")
        
        # 6. Check patch extraction
        patches_test = extract_patches_masked(kspace_acs, kernel_size=3)
        center_idx = (3 * 3) // 2
        center_max = float(torch.abs(patches_test[:, center_idx]).max())
        print(f"\n6. PATCH EXTRACTION:")
        print(f"   Patches shape: {patches_test.shape}")
        print(f"   Center pixel zeroed: {center_max < 1e-10}")
        print(f"   Center pixel max value: {center_max:.2e}")

    # Call it like this in run_benchmark():
    debug_reconstruction(kspace_gt, kspace_input, kspace_acs, mask, grappa_kspace, kan_kspace)











    # Add to your benchmark code:
    def verify_implementations():
        """Sanity checks for GRAPPA/metrics"""
        
        # Test 1: Zero-filled baseline
        zf_img = np.fft.ifft2(np.fft.ifftshift(kspace_input.cpu().numpy()))
        zf_ssim, _ = get_metrics(gt_c, zf_img)
        print(f"Zero-filled SSIM: {zf_ssim:.3f}")  # Should be ~0.40-0.60
        
        # Test 2: Perfect reconstruction
        gt_ssim, gt_phase = get_metrics(gt_c, gt_c)
        print(f"GT vs GT: SSIM={gt_ssim:.3f}, Phase={gt_phase:.4f}")
        # Should be: SSIM=1.000, Phase=0.0000
        
        # Test 3: Ordering check
        assert zf_ssim < g_ssim < 1.0, "GRAPPA not better than zero-fill!"
        assert zf_ssim < k_ssim < 1.0, "KAN not better than zero-fill!"
        
        # Test 4: MLP should at least beat zero-fill (if not, truly broken)
        print(f"MLP vs Zero-fill: {m_ssim:.3f} vs {zf_ssim:.3f}")
        
        # Test 5: Verify GRAPPA actually reconstructs
        reconstructed_points = torch.sum(mask == 0)
        filled_points = torch.sum(torch.abs(grappa_kspace - kspace_input) > 1e-8)
        print(f"GRAPPA filled {filled_points}/{reconstructed_points} missing points")
        assert filled_points > 0, "GRAPPA filled nothing!"
        
        return zf_ssim
        
    # sanity check
    # Run before reporting results
    zf_baseline = verify_implementations()
    print(f"\nBaseline (zero-filled): {zf_baseline:.3f}")
    # end sanity check






    # Sanity checks to ensure we are doing fair comparisons etc
    # Verify GRAPPA isn't just copying input
    assert not torch.allclose(grappa_kspace, kspace_input), "GRAPPA is identity!"
    # Verify it's actually reconstructing
    assert torch.sum(torch.abs(grappa_kspace)) > torch.sum(torch.abs(kspace_input)), "GRAPPA isn't filling data!"

    # Test 1: Zero-filled (no reconstruction)
    zero_filled = kspace_input  # Just the undersampled data
    zf_img = np.fft.ifft2(np.fft.ifftshift(zero_filled.cpu().numpy()))
    zf_ssim, _ = get_metrics(gt_c, zf_img)
    print(f"Zero-filled SSIM: {zf_ssim:.3f}")  # Should be ~0.50-0.60

    # Test 2: Ground truth (perfect reconstruction)
    gt_ssim, _ = get_metrics(gt_c, gt_c)
    print(f"GT vs GT SSIM: {gt_ssim:.3f}")  # Should be 1.000

    # Test 3: GRAPPA should beat zero-filled
    assert g_ssim > zf_ssim, "GRAPPA worse than doing nothing!"

    # Test 4: No method should beat GT
    assert g_ssim < 1.0, "GRAPPA claims perfection?!"
    assert k_ssim < 1.0, "KAN claims perfection?!"








    # Plot
    # Update plotting to include GRAPPA (optional)
    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(2, 4)  # Change from 3 to 4 columns
    
    # Row 1: Magnitude Images
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(np.abs(gt_c), cmap='gray', vmin=0, vmax=plot_max)
    ax1.set_title("Ground Truth")
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(np.abs(grappa_c), cmap='gray', vmin=0, vmax=plot_max)
    ax2.set_title(f"GRAPPA (SSIM: {g_ssim:.3f})")
    
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(np.abs(kan_c), cmap='gray', vmin=0, vmax=plot_max)
    ax3.set_title(f"KAN (SSIM: {k_ssim:.3f})")
    
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.imshow(np.abs(mlp_c), cmap='gray', vmin=0, vmax=plot_max)
    ax4.set_title(f"MLP (SSIM: {m_ssim:.3f})")
    
    # Row 2: Phase Errors
    ax5 = fig.add_subplot(gs[1, 0])
    ax5.imshow(np.angle(gt_c), cmap='hsv')
    ax5.set_title("GT Phase")
    
    ax6 = fig.add_subplot(gs[1, 1])
    ax6.imshow(np.abs(np.angle(gt_c)-np.angle(grappa_c)), cmap='twilight', vmin=0, vmax=3.14)
    ax6.set_title(f"GRAPPA P.Err: {g_phase:.3f}")
    
    ax7 = fig.add_subplot(gs[1, 2])
    ax7.imshow(np.abs(np.angle(gt_c)-np.angle(kan_c)), cmap='twilight', vmin=0, vmax=3.14)
    ax7.set_title(f"KAN P.Err: {k_phase:.3f}")
    
    ax8 = fig.add_subplot(gs[1, 3])
    ax8.imshow(np.abs(np.angle(gt_c)-np.angle(mlp_c)), cmap='twilight', vmin=0, vmax=3.14)
    ax8.set_title(f"MLP P.Err: {m_phase:.3f}")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_benchmark()