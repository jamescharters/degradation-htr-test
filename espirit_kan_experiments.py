"""
FIXED: FastMRI comparison with proper ifftshift handling
This was the missing piece - fastMRI k-space needs ifftshift before/after FFT!
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
import time

# ============================================================================
# Configuration
# ============================================================================

FILE_PATH = "./data/fastMRI/multicoil_test/file1000136.h5"
SLICE_IDX = None  # Will use middle slice
ESPIRIT_CALIB_WIDTH = 24
ACCEL_FACTOR = 4
CENTER_FRACTION = 0.08

# ============================================================================
# CRITICAL FIX: Proper FFT functions for fastMRI
# ============================================================================

def fastmri_to_complex(tensor):
    """Convert fastMRI (..., 2) format to complex."""
    if tensor.shape[-1] == 2:
        return tensor[..., 0] + 1j * tensor[..., 1]
    return tensor

def ifft2c(kspace):
    """
    Centered inverse FFT for fastMRI data.
    This is what was missing!
    """
    return np.fft.ifftshift(
        np.fft.ifft2(
            np.fft.ifftshift(kspace, axes=(-2, -1))
        ), 
        axes=(-2, -1)
    )

def fft2c(image):
    """Centered forward FFT for fastMRI data."""
    return np.fft.fftshift(
        np.fft.fft2(
            np.fft.fftshift(image, axes=(-2, -1))
        ),
        axes=(-2, -1)
    )

# ============================================================================
# Helper Functions
# ============================================================================

def get_coordinate_grid(height, width):
    """Generate normalized coordinate grid."""
    x = np.linspace(-1, 1, width)
    y = np.linspace(-1, 1, height)
    X, Y = np.meshgrid(x, y)
    return np.stack([X, Y], axis=-1)

def normalize_sensitivity_maps(sens_maps):
    """Normalize sensitivity maps using sum-of-squares."""
    sos = np.sqrt(np.sum(np.abs(sens_maps)**2, axis=0) + 1e-8)
    return sens_maps / (sos[np.newaxis, :, :] + 1e-8)

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

def compute_recon_metrics(img_ref, img_recon):
    """Compute PSNR and SSIM."""
    ref_mag = np.abs(img_ref)
    rec_mag = np.abs(img_recon)
    
    d_range = ref_mag.max() - ref_mag.min()
    if d_range < 1e-6:
        return -999, 0.0
    
    val_psnr = psnr(ref_mag, rec_mag, data_range=d_range)
    val_ssim = ssim(ref_mag, rec_mag, data_range=d_range)
    return val_psnr, val_ssim

# ============================================================================
# Model Architectures
# ============================================================================

class VanillaMLP(nn.Module):
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
    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                k = np.sqrt(6 / self.in_features) / self.omega_0
                self.linear.weight.uniform_(-k, k)
    
    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

class SIREN(nn.Module):
    def __init__(self, in_features=2, hidden_features=128, out_features=30, 
                 first_omega_0=30.0, hidden_omega_0=30.0):
        super().__init__()
        self.net = []
        self.net.append(SineLayer(in_features, hidden_features, is_first=True, omega_0=first_omega_0))
        self.net.append(SineLayer(hidden_features, hidden_features, is_first=False, omega_0=hidden_omega_0))
        self.net.append(SineLayer(hidden_features, hidden_features, is_first=False, omega_0=hidden_omega_0))
        self.final_linear = nn.Linear(hidden_features, out_features)
        with torch.no_grad():
            self.final_linear.weight.uniform_(-np.sqrt(6/hidden_features)/hidden_omega_0, 
                                               np.sqrt(6/hidden_features)/hidden_omega_0)
        self.net = nn.Sequential(*self.net)
    
    def forward(self, x):
        x = self.net(x)
        return self.final_linear(x)

class KANLayer(nn.Module):
    def __init__(self, in_features, out_features, grid_size=5, spline_order=3):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = nn.Parameter(torch.Tensor(out_features, in_features, grid_size))
        self.register_buffer("grid", torch.linspace(0.0, 1.0, grid_size))
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_weight, a=np.sqrt(5))
        nn.init.zeros_(self.spline_weight)
    
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
        base_output = F.linear(x, self.base_weight)
        spline_basis = self.b_splines(x)
        spline_out = torch.einsum("nig,oig->no", spline_basis, self.spline_weight)
        out = base_output + spline_out
        return out.view(*original_shape[:-1], self.out_features)

class KAN(nn.Module):
    def __init__(self, layers_hidden, grid_size=5, spline_order=3):
        super().__init__()
        self.layers = nn.ModuleList()
        for in_dim, out_dim in zip(layers_hidden[:-1], layers_hidden[1:]):
            self.layers.append(KANLayer(in_dim, out_dim, grid_size=grid_size))
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# ============================================================================
# Data Loading with CORRECT FFT
# ============================================================================

def load_and_prepare_data(file_path, slice_idx):
    """Load data with proper ifftshift handling."""
    print("="*80)
    print("LOADING DATA (with corrected FFT)")
    print("="*80)
    
    with h5py.File(file_path, "r") as hf:
        kspace_raw = hf['kspace'][:]
        
        # Use middle slice if not specified
        if slice_idx is None:
            slice_idx = kspace_raw.shape[0] // 2
            print(f"Using middle slice: {slice_idx}")
        
        kspace_slice = fastmri_to_complex(kspace_raw[slice_idx])
    
    print(f"K-space shape: {kspace_slice.shape}")
    print(f"K-space magnitude: [{np.abs(kspace_slice).min():.2e}, {np.abs(kspace_slice).max():.2e}]")
    
    # FIXED: Use centered inverse FFT
    img_coils = ifft2c(kspace_slice)
    img_rss = np.sqrt(np.sum(np.abs(img_coils)**2, axis=0))
    
    print(f"RSS Image range: [{img_rss.min():.6f}, {img_rss.max():.6f}]")
    print(f"RSS Image mean: {img_rss.mean():.6f}")
    
    # Compute ESPIRiT maps
    print("\nComputing ESPIRiT maps...")
    device_sp = sp.Device(-1)
    maps_gt = mr.app.EspiritCalib(
        kspace_slice,
        calib_width=ESPIRIT_CALIB_WIDTH,
        device=device_sp,
        crop=0.95,
        kernel_width=6,
        thresh=0.02,
        show_pbar=False,
    ).run()
    
    maps_gt = normalize_sensitivity_maps(maps_gt)
    print(f"ESPIRiT maps: {maps_gt.shape}")
    
    return kspace_slice, maps_gt, img_rss, slice_idx

# ============================================================================
# Training Functions
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
    """Predict sensitivity maps from model."""
    H, W, _ = grid.shape
    X = grid.reshape(-1, 2)
    
    with torch.no_grad():
        t_X = torch.tensor(X, dtype=torch.float32).to(device)
        out = model(t_X).cpu().numpy().reshape(H, W, 2*n_coils)
    
    out = out.reshape(H, W, n_coils, 2)
    sens = out[..., 0] + 1j * out[..., 1]
    sens = np.moveaxis(sens, -1, 0)
    return normalize_sensitivity_maps(sens)

def train_model(ModelClass, X_train, Y_train, model_kwargs, name, epochs=100):
    """Train a model."""
    print(f"\n{'='*80}")
    print(f"TRAINING: {name}")
    print(f"{'='*80}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    t_X = torch.tensor(X_train, dtype=torch.float32).to(device)
    t_Y = torch.tensor(Y_train, dtype=torch.float32).to(device)
    
    dataset = TensorDataset(t_X, t_Y)
    dataloader = DataLoader(dataset, batch_size=8192, shuffle=True)
    
    model = ModelClass(**model_kwargs).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    
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
            print(f"  Epoch {epoch:3d} | Loss: {epoch_loss:.6f}")
    
    train_time = time.time() - start
    return model, device, train_time, n_params

def run_sense(kspace_undersampled, maps):
    """Run SENSE reconstruction."""
    device_sp = sp.Device(-1)
    img_recon = mr.app.SenseRecon(
        kspace_undersampled,
        maps,
        lamda=0.01,
        device=device_sp,
        max_iter=30,
        show_pbar=False
    ).run()
    return img_recon

# ============================================================================
# Main Comparison
# ============================================================================

def main():
    # 1. Load data
    kspace, maps_gt, img_ref, slice_idx = load_and_prepare_data(FILE_PATH, SLICE_IDX)
    num_coils, H, W = maps_gt.shape
    
    # 2. Create undersampling
    mask = make_undersampling_mask(H, W, ACCEL_FACTOR, CENTER_FRACTION)
    kspace_undersampled = kspace * mask
    
    # 3. Prepare training data
    grid, X_train, Y_train = build_training_data(maps_gt)
    n_outputs = Y_train.shape[1]
    
    # 4. ESPIRiT baseline
    print(f"\n{'='*80}")
    print("ESPIRiT BASELINE")
    print(f"{'='*80}")
    img_esp = run_sense(kspace_undersampled, maps_gt)
    psnr_esp, ssim_esp = compute_recon_metrics(img_ref, img_esp)
    print(f"PSNR: {psnr_esp:.2f} dB, SSIM: {ssim_esp:.4f}")
    
    results = {'ESPIRiT': {'psnr': psnr_esp, 'ssim': ssim_esp, 'time': 0, 'params': 0}}
    
    # 5. Train models
    models_config = [
        ('MLP', VanillaMLP, {'in_features': 2, 'hidden_features': 128, 
                             'out_features': n_outputs, 'num_layers': 3}),
        ('SIREN', SIREN, {'in_features': 2, 'hidden_features': 128, 
                          'out_features': n_outputs}),
        ('KAN', KAN, {'layers_hidden': [2, 128, n_outputs], 'grid_size': 20}),
    ]
    
    for name, ModelClass, kwargs in models_config:
        model, device, train_time, n_params = train_model(
            ModelClass, X_train, Y_train, kwargs, name, epochs=100
        )
        
        maps_pred = predict_maps(model, device, grid, num_coils)
        img_recon = run_sense(kspace_undersampled, maps_pred)
        psnr_val, ssim_val = compute_recon_metrics(img_ref, img_recon)
        
        print(f"✓ {name}: PSNR={psnr_val:.2f} dB, SSIM={ssim_val:.4f}")
        
        results[name] = {
            'psnr': psnr_val,
            'ssim': ssim_val,
            'time': train_time,
            'params': n_params
        }
    
    # 6. Print results
    print(f"\n{'='*80}")
    print("FINAL RESULTS")
    print(f"{'='*80}")
    print(f"File: {FILE_PATH}")
    print(f"Slice: {slice_idx}")
    print(f"\n{'Method':<12} | {'PSNR (dB)':<10} | {'SSIM':<8} | {'Params':<10} | {'Time (s)':<10}")
    print("-" * 65)
    
    for method, m in results.items():
        params_str = f"{m['params']:,}" if m['params'] else "N/A"
        time_str = f"{m['time']:.1f}" if m['time'] else "N/A"
        print(f"{method:<12} | {m['psnr']:>8.2f} | {m['ssim']:>6.4f} | {params_str:<10} | {time_str:<10}")
    
    return results

if __name__ == "__main__":
    results = main()