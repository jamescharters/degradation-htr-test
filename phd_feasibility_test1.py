import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import h5py
import glob
import random
import os
import pandas as pd
from scipy.stats import ttest_rel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Running Quantitative Probe on: {device}")

# ==========================================
# 1. MODELS: KAN vs Linear Baseline
# ==========================================
class ComplexRBFLayer(nn.Module):
    def __init__(self, in_features, out_features, num_rbfs=16):
        super().__init__()
        # Centers initialized randomly
        self.centers = nn.Parameter(torch.randn(num_rbfs, in_features, dtype=torch.cfloat))
        self.log_gamma = nn.Parameter(torch.ones(num_rbfs) * -0.5)
        
        # FIX #1: Zero Initialization for RBF Weights
        # This ensures the model starts as a Linear model and slowly adds non-linearity
        self.weights = nn.Parameter(torch.zeros(num_rbfs, out_features, dtype=torch.cfloat))
        
        self.linear = nn.Linear(in_features, out_features, bias=False, dtype=torch.cfloat)

    def forward(self, x):
        z = x.unsqueeze(1)
        mu = self.centers.unsqueeze(0)
        dist_sq = ((z - mu) * (z - mu).conj()).real.sum(dim=2)
        phi = torch.exp(-torch.exp(self.log_gamma) * dist_sq)
        return self.linear(x) + torch.matmul(phi.type(torch.complex64), self.weights)

class CrossCoilKAN(nn.Module):
    def __init__(self):
        super().__init__()
        # 3x3 patch -> 1 pixel
        self.layer = ComplexRBFLayer(9, 1, num_rbfs=32)
        
    def forward(self, x):
        # FIX #2: Manual Complex Normalization
        # RBFs fail if values are huge (distance becomes massive, exp(-dist) -> 0)
        # We divide the patch by its maximum magnitude to keep it in the 0-1 range
        
        # Get max magnitude per patch [Batch, 1]
        scale = torch.abs(x).max(dim=1, keepdim=True)[0] + 1e-8
        x_norm = x / scale
        
        # Run KAN on normalized data
        out_norm = self.layer(x_norm)
        
        # Rescale output back to physical units
        return out_norm * scale

class LinearBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(9, 1, bias=True, dtype=torch.cfloat)
    
    def forward(self, x): 
        # Linear models usually handle scale fine, but to be fair,
        # let's give it the exact same advantage.
        scale = torch.abs(x).max(dim=1, keepdim=True)[0] + 1e-8
        x_norm = x / scale
        return self.linear(x_norm) * scale

# ==========================================
# 2. ROBUST DATA GENERATION
# ==========================================
def get_data_loader(folder_path, num_slices=10):
    files = glob.glob(os.path.join(folder_path, "*.h5"))
    if not files: raise ValueError("No files found")
    
    # Deterministic shuffle for reproducibility
    random.seed(42) 
    selected_files = random.sample(files, min(len(files), num_slices))
    
    data_buffer = []
    
    for f_path in selected_files:
        with h5py.File(f_path, 'r') as f:
            kspace = f['kspace'][()]
            # Pick middle slice
            slc = kspace[len(kspace)//2]
            
        # Reconstruct
        img_gt = np.fft.ifftshift(np.fft.ifft2(slc))
        img_gt = img_gt / (np.abs(img_gt).max() + 1e-8)
        
        # Physics Simulation
        H, W = img_gt.shape
        x = np.linspace(0, 1, W)
        y = np.linspace(0, 1, H)
        X, Y = np.meshgrid(x, y)
        
        # Coil 1: Sensitive Left
        S1 = np.exp(-((X - 0.2)**2 + (Y - 0.5)**2) * 5)
        # Coil 2: Sensitive Top + PHASE RAMP (Hard Physics)
        S2 = np.exp(-((X - 0.5)**2 + (Y - 0.2)**2) * 5) * np.exp(1j * 5 * X)
        
        C1 = torch.tensor(img_gt * S1).cfloat().to(device)
        C2 = torch.tensor(img_gt * S2).cfloat().to(device)
        
        data_buffer.append((C1, C2))
        
    return data_buffer

def extract_patches(img_tensor, kernel=3):
    H, W = img_tensor.shape
    x = img_tensor.view(1, 1, H, W)
    x_real = nn.Unfold(kernel, padding=kernel//2)(x.real).transpose(1,2)
    x_imag = nn.Unfold(kernel, padding=kernel//2)(x.imag).transpose(1,2)
    return torch.complex(x_real, x_imag).squeeze(0)

# ==========================================
# 3. METRICS
# ==========================================
def compute_nmse(pred, target):
    # Normalized Mean Squared Error
    return torch.norm(pred - target)**2 / torch.norm(target)**2

def compute_phase_error(pred, target):
    # Mean Absolute Phase Error (in Degrees)
    # Mask out background noise to avoid garbage phase stats
    mask = torch.abs(target) > 0.05 * torch.max(torch.abs(target))
    if mask.sum() == 0: return 0.0
    
    p_pred = torch.angle(pred[mask])
    p_targ = torch.angle(target[mask])
    
    # Handle wrapping
    diff = torch.abs(p_pred - p_targ)
    diff = torch.min(diff, 2*np.pi - diff)
    
    return torch.mean(diff).item() * (180 / np.pi) # Convert to degrees

# ==========================================
# 4. EXPERIMENT LOOP
# ==========================================
def run_rigorous_test():
    PATH = "./data/fastMRI/singlecoil_val" # UPDATE PATH
    slices = get_data_loader(PATH, num_slices=10)
    
    results = []
    
    print(f"{'Slice':<5} | {'Model':<8} | {'NMSE':<8} | {'Phase Err (Deg)':<15}")
    print("-" * 45)
    
    for i, (C1, C2) in enumerate(slices):
        # Prepare Data
        X = extract_patches(C1)
        Y = C2.flatten().unsqueeze(1)
        
        # Split: Train on Top 50%, Test on Bottom 50%
        split = len(X) // 2
        X_train, Y_train = X[:split], Y[:split]
        X_test, Y_test = X[split:], Y[split:]
        
        # --- MODEL 1: LINEAR (Control) ---
        lin_model = LinearBaseline().to(device)
        lin_opt = torch.optim.Adam(lin_model.parameters(), lr=0.01)
        for _ in range(300):
            lin_opt.zero_grad()
            loss = torch.abs(lin_model(X_train) - Y_train).mean()
            loss.backward()
            lin_opt.step()
            
        # --- MODEL 2: KAN (Experiment) ---
        kan_model = CrossCoilKAN().to(device)
        kan_opt = torch.optim.Adam(kan_model.parameters(), lr=0.01)
        for _ in range(300):
            kan_opt.zero_grad()
            loss = torch.abs(kan_model(X_train) - Y_train).mean()
            loss.backward()
            kan_opt.step()
            
        # --- EVALUATE ON TEST SET (Unseen Data) ---
        with torch.no_grad():
            # 1. Identity Baseline (Just copying Input to Output)
            # Input data needs to be center pixel of patch
            ident_pred = X_test[:, 4].unsqueeze(1) 
            nmse_id = compute_nmse(ident_pred, Y_test).item()
            phase_id = compute_phase_error(ident_pred, Y_test)
            
            # 2. Linear
            lin_pred = lin_model(X_test)
            nmse_lin = compute_nmse(lin_pred, Y_test).item()
            phase_lin = compute_phase_error(lin_pred, Y_test)
            
            # 3. KAN
            kan_pred = kan_model(X_test)
            nmse_kan = compute_nmse(kan_pred, Y_test).item()
            phase_kan = compute_phase_error(kan_pred, Y_test)
            
        print(f"{i:<5} | Linear   | {nmse_lin:.4f}   | {phase_lin:.2f}")
        print(f"{i:<5} | KAN      | {nmse_kan:.4f}   | {phase_kan:.2f}")
        
        results.append({'Slice': i, 'Model': 'Identity', 'NMSE': nmse_id, 'PhaseErr': phase_id})
        results.append({'Slice': i, 'Model': 'Linear', 'NMSE': nmse_lin, 'PhaseErr': phase_lin})
        results.append({'Slice': i, 'Model': 'KAN', 'NMSE': nmse_kan, 'PhaseErr': phase_kan})

    df = pd.DataFrame(results)
    
    # --- STATISTICAL SUMMARY ---
    print("\n=== AGGREGATE RESULTS (N=10 Slices) ===")
    summary = df.groupby('Model').mean()
    print(summary)
    
    # Paired T-Test (KAN vs Linear)
    kan_scores = df[df['Model']=='KAN']['NMSE'].values
    lin_scores = df[df['Model']=='Linear']['NMSE'].values
    t_stat, p_val = ttest_rel(lin_scores, kan_scores)
    
    print("\n=== HYPOTHESIS TEST ===")
    print(f"Paired T-Test (Linear vs KAN): p-value = {p_val:.5e}")
    if p_val < 0.05 and summary.loc['KAN','NMSE'] < summary.loc['Linear','NMSE']:
        print(">>> CONCLUSION: KAN statistically significantly outperforms Linear Baseline.")
        print(">>> PhD Track: VALIDATED.")
    else:
        print(">>> CONCLUSION: No significant difference found.")

    # --- VISUALIZE THE LAST SLICE ---
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    
    # Reshape preds to image
    H, W = C1.shape
    pred_lin_img = torch.zeros_like(C2); pred_lin_img.view(-1,1)[split:] = lin_pred
    pred_kan_img = torch.zeros_like(C2); pred_kan_img.view(-1,1)[split:] = kan_pred
    
    # Display Phase Difference (The hardest part)
    # GT Phase
    gt_phase = torch.angle(C2).cpu().numpy()
    # KAN Phase
    kan_phase = torch.angle(pred_kan_img).detach().cpu().numpy()
    # Linear Phase
    lin_phase = torch.angle(pred_lin_img).detach().cpu().numpy()
    
    # Plot Phase Error in Test Region
    mask = (np.abs(C2.cpu().numpy()) > 0.1) # Mask background
    
    # Ground Truth
    ax[0].imshow(np.angle(C2.cpu()), cmap='twilight')
    ax[0].set_title("Target Phase Structure")
    
    # Linear Error
    lin_err = np.abs(gt_phase - lin_phase) * mask
    ax[1].imshow(lin_err, vmin=0, vmax=np.pi/2, cmap='inferno')
    ax[1].set_title(f"Linear Phase Error\n(Avg: {phase_lin:.2f}°)")
    
    # KAN Error
    kan_err = np.abs(gt_phase - kan_phase) * mask
    ax[2].imshow(kan_err, vmin=0, vmax=np.pi/2, cmap='inferno')
    ax[2].set_title(f"KAN Phase Error\n(Avg: {phase_kan:.2f}°)")
    
    plt.show()

if __name__ == "__main__":
    run_rigorous_test()