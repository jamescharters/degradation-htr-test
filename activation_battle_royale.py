import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
import matplotlib.pyplot as plt
from torch.fft import fft2, ifft2
import nibabel as nib
from skimage.transform import resize
import glob

# ============================================================================
# 1. DATA LOADER
# ============================================================================
class MRIDataset(Dataset):
    def __init__(self, file_pattern="*.nii.gz", slices_per_vol=10):
        self.files = sorted(glob.glob(file_pattern))
        self.images = []
        if len(self.files) == 0:
            print("No files found. Using Synthetic Data.")
            self.images = [np.random.rand(128,128).astype(np.float32) for _ in range(100)]
        else:
            print(f"Loading {len(self.files)} volumes...")
            for f in self.files:
                try:
                    nii = nib.load(f)
                    data = nii.get_fdata()
                    mid = data.shape[2] // 2
                    for i in range(mid-5, mid+5):
                        slc = resize(np.rot90(data[:, :, i]), (128, 128), anti_aliasing=True)
                        slc = (slc - slc.min()) / (slc.max() - slc.min() + 1e-8)
                        self.images.append(slc)
                except Exception as e:
                    print(f"Warning: could not load {f}. Error: {e}")
        print(f"Dataset ready with {len(self.images)} slices.")

    def __len__(self): return len(self.images)
    
    def __getitem__(self, idx):
        img = torch.tensor(self.images[idx], dtype=torch.float32)
        H, W = img.shape
        # Create a new random mask every time (Data Augmentation)
        mask = torch.zeros(H, W)
        cw = int(H * 0.08)
        mask[H//2-cw:H//2+cw, :] = 1 
        idxs = np.concatenate([np.arange(0, H//2-cw), np.arange(H//2+cw, H)])
        keep = np.random.choice(idxs, int(len(idxs)*0.25), replace=False)
        mask[keep, :] = 1
        
        corrupted = ifft2(fft2(img) * mask).real
        corrupted = (corrupted - corrupted.min()) / (corrupted.max() - corrupted.min() + 1e-8)
        return corrupted.unsqueeze(0), img.unsqueeze(0)

# ============================================================================
# 2. MODELS (KAN vs Standard)
# ============================================================================
class RBFKANLayer(nn.Module):
    def __init__(self, channels, grid=8):
        super().__init__()
        self.grid = grid
        self.norm = nn.InstanceNorm2d(channels, affine=False)
        grid_centers = torch.linspace(-1, 1, grid)
        self.register_buffer("grid_centers", grid_centers)
        self.coef = nn.Parameter(torch.randn(1, channels, 1, 1, grid) * 0.01)
        self.base_scale = nn.Parameter(torch.ones(1, channels, 1, 1))

    def forward(self, x):
        x_norm = torch.tanh(self.norm(x)) 
        base = F.silu(x) * self.base_scale
        x_uns = x_norm.unsqueeze(-1)
        sigma = 2.0 / self.grid
        rbf = torch.exp(-((x_uns - self.grid_centers) ** 2) / (2 * sigma ** 2))
        return base + torch.sum(rbf * self.coef, dim=-1)

class DynamicUNet(nn.Module):
    def __init__(self, act_type="RELU"):
        super().__init__()
        self.enc1 = nn.Conv2d(1, 32, 3, padding=1)
        self.enc2 = nn.Conv2d(32, 64, 3, padding=1)
        self.dec1 = nn.Conv2d(64, 32, 3, padding=1)
        self.dec2 = nn.Conv2d(32, 1, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
        def make_act(ch):
            if act_type == "RELU": return nn.ReLU()
            if act_type == "GELU": return nn.GELU()
            if act_type == "SILU": return nn.SiLU()
            if act_type == "KAN":  return RBFKANLayer(ch, grid=8)
            raise ValueError(f"Unknown activation: {act_type}")

        self.act1 = make_act(32)
        self.act2 = make_act(64)
        self.act3 = make_act(32)

    def forward(self, x):
        x1 = self.act1(self.enc1(x))
        x2 = self.act2(self.enc2(self.pool(x1)))
        x3 = self.up(x2)
        x3 = self.dec1(x3)
        x3 = x3 + x1 
        x3 = self.act3(x3)
        return x + self.dec2(x3)

# ============================================================================
# 3. TRAINING & VALIDATION
# ============================================================================
def train_candidate(act_type, train_loader, val_loader):
    print(f"Training {act_type}...")
    model = DynamicUNet(act_type)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    best_psnr = 0
    
    for ep in range(20): # Train for 20 epochs
        model.train()
        for inp, gt in train_loader:
            opt.zero_grad()
            loss = F.l1_loss(model(inp), gt)
            loss.backward()
            opt.step()
            
        model.eval()
        psnrs = []
        with torch.no_grad():
            for inp, gt in val_loader:
                mse = torch.mean((model(inp) - gt)**2)
                if mse > 0: psnrs.append(20 * torch.log10(1.0 / torch.sqrt(mse)).item())
        
        avg_psnr = np.mean(psnrs) if psnrs else 0
        if avg_psnr > best_psnr: best_psnr = avg_psnr
        
    return best_psnr, model

# ============================================================================
# 4. FINAL VISUALIZATION
# ============================================================================
def generate_final_report(models, val_set, results):
    print("Generating Final Composite Report...")
    
    # Get a validation sample
    inp, gt = val_set[0] # Shape: (1, 128, 128)
    inp_batch = inp.unsqueeze(0)
    
    # Run Inference
    outs = {}
    residuals = {}
    for name, model in models.items():
        model.eval()
        with torch.no_grad():
            out = model(inp_batch)
            outs[name] = out[0,0].numpy()
            residuals[name] = np.abs(out[0,0].numpy() - gt[0].numpy())

    # Extract KAN Shapes
    kan_model = models['KAN']
    layer = kan_model.act1
    x_range = torch.linspace(-3, 3, 100).view(1, 1, 1, 1, 100)
    
    x_norm = torch.tanh(x_range)
    x_uns = x_norm.unsqueeze(-1)
    centers = layer.grid_centers.view(1, 1, 1, 1, 1, -1)
    sigma = 2.0 / layer.grid
    rbf = torch.exp(-((x_uns - centers) ** 2) / (2 * sigma ** 2))
    y_splines = torch.sum(rbf * layer.coef, dim=-1) 
    y_base = F.silu(x_range) * layer.base_scale
    y_total = y_base + y_splines
    
    # PLOTTING
    fig = plt.figure(figsize=(20, 10))
    
    # Row 1: Images
    ax1 = plt.subplot2grid((3, 5), (0, 0))
    ax1.imshow(inp[0], cmap='gray'); ax1.set_title("Input (Aliased)")
    
    activations = ['RELU', 'GELU', 'KAN']
    for i, name in enumerate(activations):
        ax = plt.subplot2grid((3, 5), (0, i+1))
        ax.imshow(outs[name], cmap='gray')
        ax.set_title(f"{name}\n{results[name]:.2f} dB")
        
    ax_gt = plt.subplot2grid((3, 5), (0, 4))
    ax_gt.imshow(gt[0], cmap='gray'); ax_gt.set_title("Ground Truth")

    # Row 2: Error Maps
    for i, name in enumerate(activations):
        ax = plt.subplot2grid((3, 5), (1, i+1))
        ax.imshow(residuals[name], cmap='inferno', vmin=0, vmax=0.5)
        ax.set_title(f"{name} Error")
    
    # Row 3: What KAN learned
    ax_plot = plt.subplot2grid((3, 5), (2, 0), colspan=5)
    x_np = x_range.flatten().numpy()
    
    for i in range(min(5, y_total.shape[1])): # Plot first 5 channels
        y_np = y_total[0, i, 0, 0, :].detach().numpy()
        ax_plot.plot(x_np, y_np, linewidth=2, label=f'KAN Ch {i}')
    
    ax_plot.plot(x_np, np.maximum(0, x_np), 'k--', linewidth=3, label='ReLU')
    def gelu(x): return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
    ax_plot.plot(x_np, gelu(x_np), 'b:', linewidth=3, label='GELU')
    
    ax_plot.set_title("Learned KAN Activations vs Fixed Standards")
    ax_plot.legend(); ax_plot.grid(True, alpha=0.3)
    ax_plot.set_ylim(-1, 4) # Clip for visibility
    
    # Clean up axes
    for ax in fig.get_axes(): ax.axis('off')
    ax_plot.axis('on')

    plt.tight_layout()
    plt.savefig("final_paper_figure.png")
    plt.show()

# ============================================================================
# 5. MAIN
# ============================================================================
def main():
    dataset = MRIDataset("*.nii.gz")
    train_size = int(0.8*len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=1)
    
    candidates = ["RELU", "GELU", "SILU", "KAN"]
    results = {}
    models = {}
    
    print("\n--- ACTIVATION FUNCTION BATTLE ROYALE ---")
    
    for cand in candidates:
        score, model = train_candidate(cand, train_loader, val_loader)
        results[cand] = score
        models[cand] = model
        print(f"Candidate {cand}: Best PSNR = {score:.2f} dB")
        
    # Final Verdict
    sorted_res = sorted(results.items(), key=lambda x: x[1], reverse=True)
    print("\nFINAL RANKING:")
    for rank, (name, score) in enumerate(sorted_res):
        print(f"{rank+1}. {name}: {score:.2f} dB")

    # Generate the conclusive figure
    generate_final_report(models, val_set, results)

if __name__ == "__main__":
    main()