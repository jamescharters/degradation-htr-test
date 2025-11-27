"""
Automatic diagnostic and comparison script that tries multiple approaches
to find the correct data scaling and normalization.
"""

import h5py
import numpy as np
import sigpy as sp
import sigpy.mri as mr
import matplotlib.pyplot as plt

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# ============================================================================
# Configuration
# ============================================================================

FILE_PATH = "./data/fastMRI/multicoil_test/file1000082.h5"
SLICE_IDX = 18
ESPIRIT_CALIB_WIDTH = 24
ACCEL_FACTOR = 4
CENTER_FRACTION = 0.08

# ============================================================================
# Helper functions
# ============================================================================

def fastmri_to_complex(tensor):
    """Convert fastMRI (..., 2) format to complex numpy array."""
    if tensor.shape[-1] == 2:
        return tensor[..., 0] + 1j * tensor[..., 1]
    return tensor


def get_coordinate_grid(height, width):
    """Generate normalized coordinate grid in range [-1, 1]."""
    x = np.linspace(-1, 1, width)
    y = np.linspace(-1, 1, height)
    X, Y = np.meshgrid(x, y)
    grid = np.stack([X, Y], axis=-1)
    return grid


def compute_recon_metrics(img_ref, img_recon):
    """Compute PSNR and SSIM between reference and reconstructed images."""
    ref_mag = np.abs(img_ref)
    rec_mag = np.abs(img_recon)
    
    d_range = ref_mag.max() - ref_mag.min()
    if d_range < 1e-6:
        return -999, 0.0
    
    val_psnr = psnr(ref_mag, rec_mag, data_range=d_range)
    val_ssim = ssim(ref_mag, rec_mag, data_range=d_range)
    
    return val_psnr, val_ssim


def normalize_sensitivity_maps(sens_maps):
    """Normalize sensitivity maps using sum-of-squares method."""
    sos = np.sqrt(np.sum(np.abs(sens_maps)**2, axis=0) + 1e-8)
    sens_normalized = sens_maps / (sos[np.newaxis, :, :] + 1e-8)
    return sens_normalized


def make_undersampling_mask(H, W, accel_factor=4, center_fraction=0.08):
    """Create 1D undersampling mask."""
    mask = np.zeros((H, W), dtype=np.float32)
    num_low_freq = int(round(W * center_fraction))
    pad = (W - num_low_freq + 1) // 2
    mask[:, pad:pad + num_low_freq] = 1.0
    outer_indices = np.concatenate([np.arange(0, pad),
                                    np.arange(pad + num_low_freq, W)])
    mask[:, outer_indices[::accel_factor]] = 1.0
    return mask


# ============================================================================
# Model Architectures
# ============================================================================

class VanillaMLP(nn.Module):
    """Standard MLP with ReLU activations."""
    def __init__(self, in_features=2, hidden_features=128, out_features=30, num_layers=3):
        super().__init__()
        layers = []
        layers.append(nn.Linear(in_features, hidden_features))
        layers.append(nn.ReLU())
        
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_features, hidden_features))
            layers.append(nn.ReLU())
        
        layers.append(nn.Linear(hidden_features, out_features))
        self.net = nn.Sequential(*layers)
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        return self.net(x)


class SineLayer(nn.Module):
    """SIREN layer with sine activation."""
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                k = np.sqrt(6 / self.in_features) / self.omega_0
                self.linear.weight.uniform_(-k, k)
            
    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))


class SIREN(nn.Module):
    """SIREN network."""
    def __init__(self, in_features=2, hidden_features=128, out_features=30, 
                 first_omega_0=30.0, hidden_omega_0=30.0):
        super().__init__()
        
        self.net = []
        self.net.append(SineLayer(in_features, hidden_features, 
                                  is_first=True, omega_0=first_omega_0))
        self.net.append(SineLayer(hidden_features, hidden_features, 
                                  is_first=False, omega_0=hidden_omega_0))
        self.net.append(SineLayer(hidden_features, hidden_features, 
                                  is_first=False, omega_0=hidden_omega_0))
        
        self.final_linear = nn.Linear(hidden_features, out_features)
        
        with torch.no_grad():
            self.final_linear.weight.uniform_(-np.sqrt(6/hidden_features)/hidden_omega_0, 
                                               np.sqrt(6/hidden_features)/hidden_omega_0)
            
        self.net = nn.Sequential(*self.net)
    
    def forward(self, x):
        x = self.net(x)
        return self.final_linear(x)


class KANLayer(nn.Module):
    """B-spline KAN layer."""
    def __init__(self, in_features, out_features, grid_size=5, spline_order=3):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        
        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = nn.Parameter(torch.Tensor(out_features, in_features, grid_size))
        self.register_buffer("grid", torch.linspace(0.0, 1.0, grid_size))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_weight, a=np.sqrt(5))
        nn.init.zeros_(self.spline_weight)
    
    def base_activation(self, x):
        return x
    
    def b_splines(self, x):
        x_normalized = (x + 1.0) / 2.0
        x_clamped = torch.clamp(x_normalized, 0.0, 1.0)
        diff = x_clamped.unsqueeze(-1) - self.grid
        basis = torch.exp(- (diff**2) * 10.0)
        basis = basis / (basis.sum(dim=-1, keepdim=True) + 1e-8)
        return basis
    
    def forward(self, x):
        original_shape = x.shape
        x = x.view(-1, self.in_features)
        
        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_basis = self.b_splines(x)
        spline_out = torch.einsum("nig,oig->no", spline_basis, self.spline_weight)
        
        out = base_output + spline_out
        out = out.view(*original_shape[:-1], self.out_features)
        return out


class KAN(nn.Module):
    """Multi-layer KAN."""
    def __init__(self, layers_hidden, grid_size=5, spline_order=3):
        super().__init__()
        self.layers = nn.ModuleList()
        for in_dim, out_dim in zip(layers_hidden[:-1], layers_hidden[1:]):
            self.layers.append(
                KANLayer(in_dim, out_dim, grid_size=grid_size, spline_order=spline_order)
            )
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# ============================================================================
# Data Loading & Scaling Detection
# ============================================================================

def try_multiple_scaling_approaches(kspace):
    """
    Try multiple approaches to get properly scaled images.
    Returns the best approach found.
    """
    print("\n" + "="*80)
    print("TESTING MULTIPLE SCALING APPROACHES")
    print("="*80)
    
    approaches = []
    
    # Approach 1: ortho normalization
    img1 = np.fft.ifft2(kspace, norm='ortho')
    rss1 = np.sqrt(np.sum(np.abs(img1)**2, axis=0))
    approaches.append(('ortho', rss1, 1.0))
    print(f"\n1. norm='ortho': range [{rss1.min():.6e}, {rss1.max():.6e}]")
    
    # Approach 2: no normalization
    img2 = np.fft.ifft2(kspace)
    rss2 = np.sqrt(np.sum(np.abs(img2)**2, axis=0))
    approaches.append(('none', rss2, 1.0))
    print(f"2. norm=None: range [{rss2.min():.6e}, {rss2.max():.6e}]")
    
    # Approach 3: Scale k-space first, then ortho
    kspace_scale = np.abs(kspace).max()
    if kspace_scale > 0:
        kspace_scaled = kspace / kspace_scale
        img3 = np.fft.ifft2(kspace_scaled, norm='ortho')
        rss3 = np.sqrt(np.sum(np.abs(img3)**2, axis=0))
        approaches.append(('scaled_ortho', rss3, kspace_scale))
        print(f"3. Scaled k-space + norm='ortho': range [{rss3.min():.6e}, {rss3.max():.6e}]")
    
    # Approach 4: ifftshift + ortho
    kspace_shifted = np.fft.ifftshift(kspace, axes=(-2, -1))
    img4 = np.fft.ifft2(kspace_shifted, norm='ortho')
    rss4 = np.sqrt(np.sum(np.abs(img4)**2, axis=0))
    approaches.append(('shifted_ortho', rss4, 1.0))
    print(f"4. ifftshift + norm='ortho': range [{rss4.min():.6e}, {rss4.max():.6e}]")
    
    # Approach 5: Manual scaling to reasonable range
    for scale_factor in [1e3, 1e6, 1e9]:
        rss_scaled = rss1 * scale_factor
        if 0.1 < rss_scaled.max() < 10:
            approaches.append((f'manual_scale_{scale_factor}', rss_scaled, scale_factor))
            print(f"5. Manual scale {scale_factor}: range [{rss_scaled.min():.6e}, {rss_scaled.max():.6e}]")
    
    # Find best approach (one with reasonable dynamic range)
    best_approach = None
    best_score = -1
    
    for name, rss, scale in approaches:
        # Good image should have:
        # 1. Max value between 0.01 and 100
        # 2. Good dynamic range (max/mean > 2)
        # 3. Not all zeros
        
        if rss.max() < 1e-6:
            continue
            
        dynamic_range = rss.max() / (rss.mean() + 1e-10)
        
        # Score based on how reasonable the values are
        score = 0
        if 0.01 < rss.max() < 100:
            score += 10
        if dynamic_range > 2:
            score += 5
        if 0.0001 < rss.mean() < 10:
            score += 3
        
        print(f"   {name}: score={score:.1f}, dynamic_range={dynamic_range:.2f}")
        
        if score > best_score:
            best_score = score
            best_approach = (name, rss, scale)
    
    if best_approach is None:
        print("\n⚠️  WARNING: No approach produced reasonable images!")
        print("Using 'ortho' as fallback, but results may be poor.")
        best_approach = approaches[0]
    else:
        print(f"\n✓ Best approach: {best_approach[0]} (score={best_score:.1f})")
    
    return best_approach


def load_and_prepare_data(file_path, slice_idx, calib_width):
    """
    Load data and automatically find the best scaling approach.
    """
    print(f"\n{'='*80}")
    print(f"LOADING: {file_path} | slice {slice_idx}")
    print(f"{'='*80}")
    
    with h5py.File(file_path, "r") as hf:
        kspace_raw = hf["kspace"][slice_idx]
        kspace = fastmri_to_complex(kspace_raw)
    
    print(f"K-space shape: {kspace.shape}")
    print(f"K-space magnitude: [{np.abs(kspace).min():.6e}, {np.abs(kspace).max():.6e}]")
    
    # Try different scaling approaches
    best_name, img_ref, scale_factor = try_multiple_scaling_approaches(kspace)
    
    print(f"\nUsing approach: {best_name}")
    print(f"Final image range: [{img_ref.min():.6e}, {img_ref.max():.6e}]")
    
    # Compute ESPIRiT maps
    print("\nComputing ESPIRiT maps...")
    device_sp = sp.Device(-1)
    maps_gt = mr.app.EspiritCalib(
        kspace,
        calib_width=calib_width,
        device=device_sp,
        crop=0.95,
        kernel_width=6,
        thresh=0.02,
        show_pbar=False,
    ).run()
    
    # Normalize maps
    maps_gt = normalize_sensitivity_maps(maps_gt)
    print(f"ESPIRiT maps shape: {maps_gt.shape}")
    
    return kspace, maps_gt, img_ref, best_name


# ============================================================================
# Training & Evaluation
# ============================================================================

def build_training_data(maps_gt):
    """Build training data from sensitivity maps."""
    num_coils, H, W = maps_gt.shape
    grid = get_coordinate_grid(H, W)
    X_train = grid.reshape(-1, 2)
    
    maps_transposed = np.moveaxis(maps_gt, 0, -1)
    maps_flat = maps_transposed.reshape(-1, num_coils)
    Y_train = np.concatenate([np.real(maps_flat), np.imag(maps_flat)], axis=1)
    
    return grid, X_train, Y_train


def predict_maps(model, device, grid, n_coils):
    """Predict sensitivity maps from trained model."""
    H, W, _ = grid.shape
    X = grid.reshape(-1, 2)
    
    with torch.no_grad():
        t_X = torch.tensor(X, dtype=torch.float32).to(device)
        out = model(t_X).cpu().numpy().reshape(H, W, 2*n_coils)
    
    out = out.reshape(H, W, n_coils, 2)
    sens_real = out[..., 0]
    sens_imag = out[..., 1]
    sens = sens_real + 1j * sens_imag
    sens = np.moveaxis(sens, -1, 0)
    
    # Normalize
    sens = normalize_sensitivity_maps(sens)
    return sens


def train_model(ModelClass, X_train, Y_train, n_outputs, 
                model_kwargs, model_name, epochs=100):
    """Train a model."""
    print(f"\n{'='*80}")
    print(f"TRAINING: {model_name}")
    print(f"{'='*80}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    t_X = torch.tensor(X_train, dtype=torch.float32).to(device)
    t_Y = torch.tensor(Y_train, dtype=torch.float32).to(device)
    
    dataset = TensorDataset(t_X, t_Y)
    dataloader = DataLoader(dataset, batch_size=8192, shuffle=True)
    
    model = ModelClass(**model_kwargs).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    
    import time
    start = time.time()
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = F.mse_loss(preds, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_x.size(0)
        epoch_loss /= len(dataset)
        scheduler.step()
        
        if epoch % 20 == 0 or epoch == epochs - 1:
            print(f"  Epoch {epoch:3d}/{epochs-1} | Loss: {epoch_loss:.6f}")
    
    train_time = time.time() - start
    print(f"Training time: {train_time:.1f}s")
    
    return model, device, train_time, n_params


def run_sense_safe(kspace_undersampled, maps, img_ref):
    """Run SENSE with error handling."""
    try:
        device_sp = sp.Device(-1)
        img_recon = mr.app.SenseRecon(
            kspace_undersampled,
            maps,
            lamda=0.01,
            device=device_sp,
            max_iter=30,
            show_pbar=False
        ).run()
        
        psnr_val, ssim_val = compute_recon_metrics(img_ref, img_recon)
        return img_recon, psnr_val, ssim_val
        
    except Exception as e:
        print(f"  ⚠️  SENSE reconstruction failed: {e}")
        return None, -999, 0.0


# ============================================================================
# Main Comparison
# ============================================================================

def main_automatic_comparison():
    """Run complete comparison with automatic diagnostics."""
    
    # 1. Load and prepare data
    kspace, maps_gt, img_ref, scaling_method = load_and_prepare_data(
        FILE_PATH, SLICE_IDX, ESPIRIT_CALIB_WIDTH
    )
    
    if img_ref.max() < 1e-6:
        print("\n❌ ERROR: Could not create valid reference image!")
        print("Please check your data file and slice index.")
        return None
    
    num_coils, H, W = maps_gt.shape
    
    # 2. Create undersampling mask
    mask = make_undersampling_mask(H, W, ACCEL_FACTOR, CENTER_FRACTION)
    kspace_undersampled = kspace * mask
    
    # 3. Prepare training data
    grid, X_train, Y_train = build_training_data(maps_gt)
    n_outputs = Y_train.shape[1]
    
    # 4. ESPIRiT baseline
    print(f"\n{'='*80}")
    print("ESPIRiT BASELINE")
    print(f"{'='*80}")
    img_esp, psnr_esp, ssim_esp = run_sense_safe(kspace_undersampled, maps_gt, img_ref)
    
    if psnr_esp < 0:
        print("⚠️  WARNING: ESPIRiT baseline failed. Continuing anyway...")
    else:
        print(f"✓ ESPIRiT: PSNR={psnr_esp:.2f} dB, SSIM={ssim_esp:.4f}")
    
    results = {
        'ESPIRiT': {'psnr': psnr_esp, 'ssim': ssim_esp, 'time': 0, 'params': 0}
    }
    
    # 5. Train models
    models_to_test = [
        ('MLP', VanillaMLP, {'in_features': 2, 'hidden_features': 128, 
                             'out_features': n_outputs, 'num_layers': 3}),
        ('SIREN', SIREN, {'in_features': 2, 'hidden_features': 128, 
                          'out_features': n_outputs}),
        ('KAN', KAN, {'layers_hidden': [2, 128, n_outputs], 
                      'grid_size': 20, 'spline_order': 3}),
    ]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for name, ModelClass, kwargs in models_to_test:
        try:
            model, device, train_time, n_params = train_model(
                ModelClass, X_train, Y_train, n_outputs, kwargs, name, epochs=100
            )
            
            maps_pred = predict_maps(model, device, grid, num_coils)
            img_recon, psnr_val, ssim_val = run_sense_safe(
                kspace_undersampled, maps_pred, img_ref
            )
            
            results[name] = {
                'psnr': psnr_val,
                'ssim': ssim_val,
                'time': train_time,
                'params': n_params
            }
            
            if psnr_val > 0:
                print(f"✓ {name}: PSNR={psnr_val:.2f} dB, SSIM={ssim_val:.4f}")
            else:
                print(f"⚠️  {name}: Reconstruction failed")
                
        except Exception as e:
            print(f"❌ {name}: Training/evaluation failed: {e}")
            results[name] = {'psnr': -999, 'ssim': 0, 'time': 0, 'params': 0}
    
    # 6. Print results
    print(f"\n{'='*80}")
    print("FINAL RESULTS")
    print(f"{'='*80}")
    print(f"Scaling method used: {scaling_method}")
    print(f"\n{'Method':<12} | {'PSNR (dB)':<10} | {'SSIM':<8} | {'Params':<10} | {'Time (s)':<10}")
    print("-" * 65)
    
    for method, m in results.items():
        params_str = f"{m['params']:,}" if m['params'] else "N/A"
        time_str = f"{m['time']:.1f}" if m['time'] else "N/A"
        psnr_str = f"{m['psnr']:>8.2f}" if m['psnr'] > -100 else "FAILED"
        ssim_str = f"{m['ssim']:>6.4f}" if m['ssim'] > 0 else "FAILED"
        print(f"{method:<12} | {psnr_str} | {ssim_str} | {params_str:<10} | {time_str:<10}")
    
    return results


if __name__ == "__main__":
    results = main_automatic_comparison()