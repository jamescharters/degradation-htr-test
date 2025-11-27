import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.fft import fft2, ifft2
import nibabel as nib
from skimage.transform import resize

# ============================================================================
# 1. SHARED UTILS & PHYSICS
# ============================================================================

def load_ixi_slice(filepath, slice_idx=None):
    try:
        nii_img = nib.load(filepath)
        data = nii_img.get_fdata()
        if slice_idx is None: slice_idx = data.shape[2] // 2
        img_slice = data[:, :, slice_idx]
        img_slice = np.rot90(img_slice) 
        img_slice = (img_slice - img_slice.min()) / (img_slice.max() - img_slice.min())
        img_slice = resize(img_slice, (256, 256), anti_aliasing=True)
        return torch.tensor(img_slice, dtype=torch.float32)
    except: return None

def compute_psnr(pred, target, max_val=1.0):
    mse = torch.mean((pred - target) ** 2)
    if mse < 1e-10: return 100.0
    return 20 * np.log10(max_val / np.sqrt(mse.item()))

def create_motion_corrupted_image(clean_image, severity=10.0, motion_type='linear'):
    H, W = clean_image.shape
    t = torch.linspace(0, 1, H)
    
    if motion_type == 'linear':
        rotation = severity * t
        translation = severity * t
    
    kspace = fft2(clean_image)
    kspace_corrupted = kspace.clone()
    
    for line_idx in range(H):
        ky = (line_idx - H//2) / H
        kx = (torch.arange(W, dtype=torch.float32) - W//2) / W
        phase = 2 * np.pi * (translation[line_idx] * ky * 0.05 + rotation[line_idx] * kx * 0.1)
        kspace_corrupted[line_idx, :] *= torch.exp(1j * phase)
    
    corrupted = ifft2(kspace_corrupted).real
    
    return corrupted, {'rotation': rotation, 'translation': translation, 'timestamps': t}

# ============================================================================
# 2. MODEL DEFINITIONS (Must match training scripts exactly)
# ============================================================================

# --- KAN COMPONENTS ---
class DynamicChebyshevKAN(nn.Module):
    def __init__(self, degree=3):
        super().__init__()
        self.degree = degree
    def forward(self, x, coeffs):
        x_norm = 2 * x - 1
        T = [torch.ones_like(x), x_norm]
        for n in range(2, self.degree + 1):
            T.append(2 * x_norm * T[-1] - T[-2])
        return torch.matmul(coeffs, torch.stack(T))

class HyperNetworkMotionModel(nn.Module): # The KAN Model
    def __init__(self, image_size=256, kan_degree=3):
        super().__init__()
        self.dynamic_rot_kan = DynamicChebyshevKAN(degree=kan_degree)
        self.dynamic_trans_kan = DynamicChebyshevKAN(degree=kan_degree)
        self.enc1 = self._conv_block(1, 32)
        self.enc2 = self._conv_block(32, 64)
        self.enc3 = self._conv_block(64, 128)
        self.pool = nn.MaxPool2d(2)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.rot_predictor = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, kan_degree + 1))
        self.trans_predictor = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, kan_degree + 1))
        self.motion_map_encoder = nn.Sequential(nn.Linear(2, 64), nn.LayerNorm(64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU())
        self.bottleneck = self._conv_block(128 + 32, 128)
        self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec3 = self._conv_block(128, 64)
        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec2 = self._conv_block(64, 32)
        self.final = nn.Sequential(nn.Conv2d(32, 16, 3, padding=1), nn.ReLU(), nn.Conv2d(16, 1, 1), nn.Tanh())

    def _conv_block(self, in_ch, out_ch):
        return nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True), nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))

    def forward(self, corrupted_image, timestamps):
        H, W = corrupted_image.shape
        M = len(timestamps)
        x = corrupted_image.unsqueeze(0).unsqueeze(0)
        x1 = self.enc1(x); x2 = self.enc2(self.pool(x1)); x3 = self.enc3(self.pool(x2))
        global_features = self.global_pool(x3).flatten(1)
        rot_coeffs = self.rot_predictor(global_features)
        trans_coeffs = self.trans_predictor(global_features)
        est_rot = self.dynamic_rot_kan(timestamps, rot_coeffs).squeeze()
        est_trans = self.dynamic_trans_kan(timestamps, trans_coeffs).squeeze()
        
        motion_map = torch.zeros(H, W, 2, device=corrupted_image.device)
        for r in range(H):
            idx = min(int((r/H)*M), M-1)
            motion_map[r,:,0] = est_rot[idx]
            motion_map[r,:,1] = est_trans[idx]
            
        m_feat = self.motion_map_encoder(motion_map.reshape(-1,2)).reshape(H,W,32).permute(2,0,1)
        m_bottle = nn.functional.adaptive_avg_pool2d(m_feat.unsqueeze(0), (H//4, W//4))
        d = self.bottleneck(torch.cat([x3, m_bottle], dim=1))
        d = self.dec3(torch.cat([self.up3(d), x2], dim=1))
        d = self.dec2(torch.cat([self.up2(d), x1], dim=1))
        return corrupted_image + 0.05 * self.final(d).squeeze(), {'rotation': est_rot, 'translation': est_trans}

# --- MLP COMPONENTS ---
class MotionMLP(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=64, output_points=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_points))
    def forward(self, x, timestamps):
        mc = self.net(x)
        if mc.shape[1] != len(timestamps):
            mc = nn.functional.interpolate(mc.unsqueeze(1), size=len(timestamps), mode='linear').squeeze(1)
        return mc

class MLPMotionModel(nn.Module): # The MLP Model
    def __init__(self, image_size=256):
        super().__init__()
        self.rot_mlp = MotionMLP(input_dim=128, output_points=image_size)
        self.trans_mlp = MotionMLP(input_dim=128, output_points=image_size)
        self.enc1 = self._conv_block(1, 32); self.enc2 = self._conv_block(32, 64); self.enc3 = self._conv_block(64, 128)
        self.pool = nn.MaxPool2d(2); self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.motion_map_encoder = nn.Sequential(nn.Linear(2, 64), nn.LayerNorm(64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU())
        self.bottleneck = self._conv_block(128 + 32, 128)
        self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2); self.dec3 = self._conv_block(128, 64)
        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2); self.dec2 = self._conv_block(64, 32)
        self.final = nn.Sequential(nn.Conv2d(32, 16, 3, padding=1), nn.ReLU(), nn.Conv2d(16, 1, 1), nn.Tanh())

    def _conv_block(self, in_ch, out_ch):
        return nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True), nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))

    def forward(self, corrupted_image, timestamps):
        H, W = corrupted_image.shape
        M = len(timestamps)
        x = corrupted_image.unsqueeze(0).unsqueeze(0)
        x1 = self.enc1(x); x2 = self.enc2(self.pool(x1)); x3 = self.enc3(self.pool(x2))
        gf = self.global_pool(x3).flatten(1)
        est_rot = self.rot_mlp(gf, timestamps).squeeze()
        est_trans = self.trans_mlp(gf, timestamps).squeeze()
        
        motion_map = torch.zeros(H, W, 2, device=corrupted_image.device)
        for r in range(H):
            idx = min(int((r/H)*M), M-1)
            motion_map[r,:,0] = est_rot[idx]
            motion_map[r,:,1] = est_trans[idx]
            
        m_feat = self.motion_map_encoder(motion_map.reshape(-1,2)).reshape(H,W,32).permute(2,0,1)
        m_bottle = nn.functional.adaptive_avg_pool2d(m_feat.unsqueeze(0), (H//4, W//4))
        d = self.bottleneck(torch.cat([x3, m_bottle], dim=1))
        d = self.dec3(torch.cat([self.up3(d), x2], dim=1))
        d = self.dec2(torch.cat([self.up2(d), x1], dim=1))
        return corrupted_image + 0.05 * self.final(d).squeeze(), {'rotation': est_rot, 'translation': est_trans}

# ============================================================================
# 3. VISUALIZATION LOGIC
# ============================================================================

def run_comparison():
    print("1. Loading Data...")
    clean_image = load_ixi_slice("IXI002-Guys-0828-T2.nii")
    if clean_image is None: return
    
    # Generate ONE corrupted example (Severity 10 to show MLP saturation)
    print("2. Creating Corrupted Sample (Severity=10.0)...")
    corrupted, true_motion = create_motion_corrupted_image(clean_image, severity=10.0)
    t = true_motion['timestamps']
    
    # Load KAN
    print("3. Running KAN Model...")
    kan_model = HyperNetworkMotionModel(image_size=256)
    kan_model.load_state_dict(torch.load('best_kan_model.pth'))
    kan_model.eval()
    with torch.no_grad():
        kan_img, kan_est = kan_model(corrupted, t)
        
    # Load MLP
    print("4. Running MLP Model...")
    mlp_model = MLPMotionModel(image_size=256)
    mlp_model.load_state_dict(torch.load('best_mlp_model.pth'))
    mlp_model.eval()
    with torch.no_grad():
        mlp_img, mlp_est = mlp_model(corrupted, t)
        
    # --- PLOTTING ---
    print("5. Generating Plot...")
    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(2, 4)
    
    # Row 1: Images
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(clean_image, cmap='gray', vmin=0, vmax=1)
    ax1.set_title("Ground Truth", fontweight='bold')
    ax1.axis('off')
    
    ax2 = fig.add_subplot(gs[0, 1])
    psnr_in = compute_psnr(corrupted, clean_image)
    ax2.imshow(corrupted, cmap='gray', vmin=0, vmax=1)
    ax2.set_title(f"Corrupted Input\nPSNR: {psnr_in:.1f} dB", fontweight='bold')
    ax2.axis('off')
    
    ax3 = fig.add_subplot(gs[0, 2])
    psnr_mlp = compute_psnr(mlp_img, clean_image)
    ax3.imshow(mlp_img, cmap='gray', vmin=0, vmax=1)
    ax3.set_title(f"MLP Baseline\nPSNR: {psnr_mlp:.1f} dB", fontweight='bold')
    ax3.axis('off')
    
    ax4 = fig.add_subplot(gs[0, 3])
    psnr_kan = compute_psnr(kan_img, clean_image)
    ax4.imshow(kan_img, cmap='gray', vmin=0, vmax=1)
    ax4.set_title(f"HyperNet KAN (Ours)\nPSNR: {psnr_kan:.1f} dB", fontweight='bold', color='green')
    ax4.axis('off')
    
    # Row 2: Motion Curves
    
    # Rotation
    ax_rot = fig.add_subplot(gs[1, :2])
    rot_true = true_motion['rotation']
    rot_kan = kan_est['rotation']
    rot_mlp = mlp_est['rotation']
    
    r2_kan = np.corrcoef(rot_true, rot_kan)[0,1]**2
    r2_mlp = np.corrcoef(rot_true, rot_mlp)[0,1]**2
    
    ax_rot.plot(t, rot_true, 'k-', linewidth=3, label='Ground Truth', alpha=0.5)
    ax_rot.plot(t, rot_mlp, 'r--', linewidth=2, label=f'MLP (R²={r2_mlp:.2f})')
    ax_rot.plot(t, rot_kan, 'g-', linewidth=2, label=f'KAN (R²={r2_kan:.2f})')
    ax_rot.set_title("Comparative Analysis: Rotation Estimation", fontsize=14, fontweight='bold')
    ax_rot.set_ylabel("Rotation (Degrees)")
    ax_rot.set_xlabel("Acquisition Time")
    ax_rot.legend()
    ax_rot.grid(True, alpha=0.3)
    
    # Translation
    ax_trans = fig.add_subplot(gs[1, 2:])
    trans_true = true_motion['translation']
    trans_kan = kan_est['translation']
    trans_mlp = mlp_est['translation']
    
    r2_kan_t = np.corrcoef(trans_true, trans_kan)[0,1]**2
    r2_mlp_t = np.corrcoef(trans_true, trans_mlp)[0,1]**2
    
    ax_trans.plot(t, trans_true, 'k-', linewidth=3, label='Ground Truth', alpha=0.5)
    ax_trans.plot(t, trans_mlp, 'r--', linewidth=2, label=f'MLP (R²={r2_mlp_t:.2f})')
    ax_trans.plot(t, trans_kan, 'g-', linewidth=2, label=f'KAN (R²={r2_kan_t:.2f})')
    ax_trans.set_title("Comparative Analysis: Translation Estimation", fontsize=14, fontweight='bold')
    ax_trans.set_ylabel("Translation (mm)")
    ax_trans.set_xlabel("Acquisition Time")
    ax_trans.legend()
    ax_trans.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('final_thesis_comparison.png', dpi=300)
    plt.show()
    
    print(f"Comparison Saved to final_thesis_comparison.png")

if __name__ == '__main__':
    run_comparison()