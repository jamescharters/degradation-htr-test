import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
from torch.fft import fft2, ifft2
import nibabel as nib
from skimage.transform import resize
import glob
import os

# ============================================================================
# 1. ROBUST DATASET LOADING
# ============================================================================
class MRIDataset(Dataset):
    def __init__(self, file_pattern="*.nii.gz", slices_per_vol=10):
        self.files = sorted(glob.glob(file_pattern))
        self.images = []
        
        print(f"Found {len(self.files)} MRI volumes.")
        
        for f in self.files:
            try:
                nii = nib.load(f)
                data = nii.get_fdata()
                # Extract center slices (most interesting anatomy)
                mid = data.shape[2] // 2
                start = mid - (slices_per_vol // 2)
                
                for i in range(start, start + slices_per_vol):
                    # Rotate and Resize to standard 128x128
                    slc = resize(np.rot90(data[:, :, i]), (128, 128), anti_aliasing=True)
                    
                    # Normalize 0-1 per slice
                    slc = (slc - slc.min()) / (slc.max() - slc.min() + 1e-8)
                    self.images.append(slc)
            except Exception as e:
                print(f"Skipping {f}: {e}")
                
        print(f"Total Dataset Size: {len(self.images)} slices.")
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img = torch.tensor(self.images[idx], dtype=torch.float32)
        
        # Dynamic Physics Augmentation
        # We generate a new mask every time to force learning general patterns
        H, W = img.shape
        kspace = fft2(img)
        
        mask = torch.zeros(H, W)
        center = H // 2
        cw = int(H * 0.08)
        mask[center-cw:center+cw, :] = 1 
        
        # 4x Acceleration
        idxs = np.concatenate([np.arange(0, center-cw), np.arange(center+cw, H)])
        keep = np.random.choice(idxs, int(len(idxs)*0.25), replace=False)
        mask[keep, :] = 1
        
        corrupted_k = kspace * mask
        corrupted_img = ifft2(corrupted_k).real
        
        # Re-normalize input to ensure stability
        corrupted_img = (corrupted_img - corrupted_img.min()) / (corrupted_img.max() - corrupted_img.min() + 1e-8)
        
        return corrupted_img.unsqueeze(0), img.unsqueeze(0)

# ============================================================================
# 2. STABLE RBF-KAN LAYER
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
        centers = self.grid_centers
        sigma = 2.0 / self.grid
        rbf_bases = torch.exp(-((x_uns - centers) ** 2) / (2 * sigma ** 2))
        
        kan_out = torch.sum(rbf_bases * self.coef, dim=-1)
        return base + kan_out

# ============================================================================
# 3. UNET ARCHITECTURE
# ============================================================================
class SimpleUNet(nn.Module):
    def __init__(self, use_kan=False):
        super().__init__()
        self.enc1 = nn.Conv2d(1, 32, 3, padding=1)
        self.enc2 = nn.Conv2d(32, 64, 3, padding=1)
        self.dec1 = nn.Conv2d(64, 32, 3, padding=1)
        self.dec2 = nn.Conv2d(32, 1, 3, padding=1)
        
        if use_kan:
            self.act1 = RBFKANLayer(32, grid=8)
            self.act2 = RBFKANLayer(64, grid=8)
            self.act3 = RBFKANLayer(32, grid=8)
        else:
            self.act1 = nn.ReLU()
            self.act2 = nn.ReLU()
            self.act3 = nn.ReLU()
            
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, x):
        x1 = self.enc1(x)
        x1 = self.act1(x1)
        x2 = self.pool(x1)
        x2 = self.enc2(x2)
        x2 = self.act2(x2)
        
        x3 = self.up(x2)
        x3 = self.dec1(x3)
        x3 = x3 + x1 # Residual connection
        x3 = self.act3(x3)
        
        out = self.dec2(x3)
        return x + out

# ============================================================================
# 4. TRAINING ROUTINE
# ============================================================================
def train_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0
    for inp, gt in loader:
        optimizer.zero_grad()
        pred = model(inp)
        loss = F.l1_loss(pred, gt) # L1 for sharpness
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def validate(model, loader):
    model.eval()
    total_psnr = 0
    with torch.no_grad():
        for inp, gt in loader:
            pred = model(inp)
            mse = torch.mean((pred - gt)**2)
            psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
            total_psnr += psnr.item()
    return total_psnr / len(loader)







def visualize_paper_figure(cnn_model, kan_model, val_loader):
    print("Generating High-Res Figure...")
    cnn_model.eval()
    kan_model.eval()
    
    # Get a specific image that looks interesting (e.g., index 0 or 1)
    # We iterate to find one with high contrast
    iterator = iter(val_loader)
    inp, gt = next(iterator)
    
    with torch.no_grad():
        out_c = cnn_model(inp)
        out_k = kan_model(inp)
    
    # Calculate Residuals (The Error Maps)
    # We amplify them by 5x so they are visible to the human eye
    res_c = torch.abs(out_c - gt) * 5 
    res_k = torch.abs(out_k - gt) * 5
    
    # Prepare Plot
    fig, ax = plt.subplots(2, 3, figsize=(15, 10))
    
    # Row 1: Images
    ax[0,0].imshow(inp[0,0], cmap='gray'); ax[0,0].set_title("Aliased Input (4x)")
    ax[0,1].imshow(out_c[0,0], cmap='gray'); ax[0,1].set_title("Standard U-Net")
    ax[0,2].imshow(out_k[0,0], cmap='gray'); ax[0,2].set_title("U-KAN (Ours)")
    
    # Row 2: Error Maps (The "Scientific Proof")
    ax[1,0].imshow(gt[0,0], cmap='gray'); ax[1,0].set_title("Ground Truth")
    
    # Use 'inferno' for errors - Black means perfect, Yellow means big error
    im1 = ax[1,1].imshow(res_c[0,0], cmap='inferno', vmin=0, vmax=1)
    ax[1,1].set_title("U-Net Error (Brighter = Worse)")
    
    im2 = ax[1,2].imshow(res_k[0,0], cmap='inferno', vmin=0, vmax=1)
    ax[1,2].set_title("U-KAN Error (Darker = Better)")
    
    # Remove axes
    for a in ax.ravel(): a.axis('off')
    
    plt.tight_layout()
    plt.savefig("paper_figure.png", dpi=300) # High DPI for report
    plt.show()
    
    # Print numerical gap
    err_c_sum = torch.sum(res_c).item()
    err_k_sum = torch.sum(res_k).item()
    print(f"Total Error Mass - CNN: {err_c_sum:.2f}")
    print(f"Total Error Mass - KAN: {err_k_sum:.2f}")
    print(f"Error Reduction: {((err_c_sum - err_k_sum)/err_c_sum)*100:.1f}%")







def visualize_features(cnn_model, kan_model, val_loader):
    print("X-Raying the Models...")
    
    # Get one sample
    iterator = iter(val_loader)
    inp, gt = next(iterator)
    
    # --- 1. CNN INTERNALS ---
    # Run through first Conv
    c_feat = cnn_model.enc1(inp)
    # Run through ReLU
    c_act = cnn_model.act1(c_feat)
    
    # --- 2. KAN INTERNALS ---
    # Run through first Conv
    k_feat = kan_model.enc1(inp)
    # Run through RBF-KAN
    k_act = kan_model.act1(k_feat)
    
    # --- VISUALIZATION ---
    # We pick the channel with the highest variance (most activity)
    c_idx = torch.argmax(torch.std(c_act, dim=(2,3))).item()
    k_idx = torch.argmax(torch.std(k_act, dim=(2,3))).item()
    
    plt.figure(figsize=(12, 8))
    
    # Input
    plt.subplot(2, 2, 1)
    plt.imshow(inp[0,0], cmap='gray')
    plt.title("Input Image")
    plt.axis('off')
    
    # CNN Feature Map
    plt.subplot(2, 2, 2)
    # Use 'gray' but clip negative to black to show ReLU effect
    plt.imshow(c_act[0, c_idx].detach(), cmap='gray') 
    plt.title(f"CNN Feature (ReLU)\nChannel {c_idx}")
    plt.axis('off')
    
    # KAN Feature Map
    plt.subplot(2, 2, 4)
    plt.imshow(k_act[0, k_idx].detach(), cmap='gray')
    plt.title(f"KAN Feature (RBF)\nChannel {k_idx}")
    plt.axis('off')
    
    # Histogram of Activations (The Math Proof)
    plt.subplot(2, 2, 3)
    
    c_vals = c_act.detach().flatten().numpy()
    k_vals = k_act.detach().flatten().numpy()
    
    plt.hist(c_vals, bins=50, alpha=0.5, label='CNN (ReLU)', color='gray', density=True)
    plt.hist(k_vals, bins=50, alpha=0.5, label='KAN (RBF)', color='red', density=True)
    plt.legend()
    plt.title("Activation Histogram\n(Note ReLU Spike at 0)")
    
    plt.tight_layout()
    plt.savefig("internal_features.png")
    plt.show()



# ============================================================================
# 5. MAIN EXPERIMENT
# ============================================================================
def main():
    # 1. Prepare Data
    full_dataset = MRIDataset("*.nii.gz", slices_per_vol=10)
    if len(full_dataset) == 0:
        print("No files found! Generating synthetic data for demo...")
        # Synthetic fallback...
        full_dataset.images = [np.random.rand(128,128).astype(np.float32) for _ in range(50)]
    
    # 80/20 Split
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_set, val_set = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)
    
    print(f"\nTraining on {train_size} slices, Validating on {val_size} slices.")
    
    # 2. Setup Models
    cnn = SimpleUNet(use_kan=False)
    kan = SimpleUNet(use_kan=True)
    
    opt_cnn = torch.optim.Adam(cnn.parameters(), lr=1e-3)
    opt_kan = torch.optim.Adam(kan.parameters(), lr=1e-3)
    
    # 3. Training Loop
    epochs = 30 # Fast check
    
    print("\n--- Starting Training ---")
    print(f"{'Epoch':<5} | {'CNN Loss':<10} | {'KAN Loss':<10} | {'CNN PSNR':<10} | {'KAN PSNR':<10}")
    print("-" * 55)
    
    for ep in range(epochs):
        loss_c = train_epoch(cnn, train_loader, opt_cnn)
        loss_k = train_epoch(kan, train_loader, opt_kan)
        
        # Validate every 5 epochs
        if (ep+1) % 5 == 0:
            psnr_c = validate(cnn, val_loader)
            psnr_k = validate(kan, val_loader)
            print(f"{ep+1:<5} | {loss_c:.6f}   | {loss_k:.6f}   | {psnr_c:.2f}dB    | {psnr_k:.2f}dB")
    
    # 4. Visual Comparison on One Test Image
    cnn.eval(); kan.eval()
    sample_inp, sample_gt = val_set[0] # Get first validation item
    sample_inp = sample_inp.unsqueeze(0)
    sample_gt = sample_gt.unsqueeze(0)
    
    with torch.no_grad():
        out_c = cnn(sample_inp)
        out_k = kan(sample_inp)
        
    p_c = 20 * torch.log10(1.0 / torch.sqrt(torch.mean((out_c - sample_gt)**2))).item()
    p_k = 20 * torch.log10(1.0 / torch.sqrt(torch.mean((out_k - sample_gt)**2))).item()
    
    # Plot
    plt.figure(figsize=(16, 6))
    plt.subplot(1,4,1); plt.imshow(sample_inp[0,0], cmap='gray'); plt.title("Aliased Input")
    plt.subplot(1,4,2); plt.imshow(out_c[0,0], cmap='gray'); plt.title(f"Standard U-Net\n{p_c:.2f}dB")
    plt.subplot(1,4,3); plt.imshow(out_k[0,0], cmap='gray'); plt.title(f"U-KAN (RBF)\n{p_k:.2f}dB")
    plt.subplot(1,4,4); plt.imshow(sample_gt[0,0], cmap='gray'); plt.title("Ground Truth")
    plt.tight_layout()
    plt.savefig("validation_comparison.png")
    plt.show()
    
    print("\nFINAL VALIDATION RESULTS:")
    if p_k > p_c:
        print(f"✅ U-KAN Wins by +{p_k - p_c:.2f} dB on unseen data.")
    else:
        print(f"❌ Standard U-Net Wins by +{p_c - p_k:.2f} dB.")


    visualize_features(cnn, kan, val_loader)
    
    visualize_paper_figure(cnn, kan, val_loader)

if __name__ == "__main__":
    main()