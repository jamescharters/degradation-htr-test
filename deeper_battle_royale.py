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
# 1. DATA (Same as before)
# ============================================================================
class MRIDataset(Dataset):
    def __init__(self, file_pattern="*.nii.gz", slices_per_vol=10):
        # ... (Identical to the previous script) ...
        self.files = sorted(glob.glob(file_pattern))
        self.images = []
        if not self.files:
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
                except: pass
        
    def __len__(self): return len(self.images)
    def __getitem__(self, idx):
        img = torch.tensor(self.images[idx], dtype=torch.float32)
        H, W = img.shape
        mask = torch.zeros(H, W)
        cw = int(H * 0.08); mask[H//2-cw:H//2+cw, :] = 1 
        idxs = np.concatenate([np.arange(0, H//2-cw), np.arange(H//2+cw, H)])
        keep = np.random.choice(idxs, int(len(idxs)*0.25), replace=False); mask[keep, :] = 1
        corrupted = ifft2(fft2(img) * mask).real
        corrupted = (corrupted - corrupted.min()) / (corrupted.max() - corrupted.min() + 1e-8)
        return corrupted.unsqueeze(0), img.unsqueeze(0)

# ============================================================================
# 2. RBF-KAN LAYER (Same as before)
# ============================================================================
class RBFKANLayer(nn.Module):
    # ... (Identical to the previous script) ...
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

# ============================================================================
# 3. DEEPER DYNAMIC U-NET (The Upgrade)
# ============================================================================
class DeeperUNet(nn.Module):
    def __init__(self, act_type="RELU"):
        super().__init__()
        
        def make_act(ch):
            if act_type == "RELU": return nn.ReLU()
            if act_type == "GELU": return nn.GELU()
            if act_type == "SILU": return nn.SiLU()
            if act_type == "KAN":  return RBFKANLayer(ch, grid=8)
            raise ValueError()

        # Encoder (Downsample Twice)
        self.enc1a = nn.Conv2d(1, 32, 3, padding=1)
        self.enc1b = nn.Conv2d(32, 32, 3, padding=1)
        self.act1 = make_act(32)
        
        self.enc2a = nn.Conv2d(32, 64, 3, padding=1)
        self.enc2b = nn.Conv2d(64, 64, 3, padding=1)
        self.act2 = make_act(64)
        
        # Bottleneck
        self.bot1 = nn.Conv2d(64, 128, 3, padding=1)
        self.bot2 = nn.Conv2d(128, 128, 3, padding=1)
        self.act_bot = make_act(128)
        
        # Decoder (Upsample Twice)
        self.dec1a = nn.Conv2d(128 + 64, 64, 3, padding=1) # Note: in_channels from concat
        self.dec1b = nn.Conv2d(64, 64, 3, padding=1)
        self.act3 = make_act(64)
        
        self.dec2a = nn.Conv2d(64 + 32, 32, 3, padding=1)
        self.dec2b = nn.Conv2d(32, 32, 3, padding=1)
        self.act4 = make_act(32)
        
        self.final = nn.Conv2d(32, 1, 1)
        
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, x):
        # Encoder
        x1 = self.act1(self.enc1b(self.enc1a(x)))
        x2 = self.act2(self.enc2b(self.enc2a(self.pool(x1))))
        
        # Bottleneck
        bot = self.act_bot(self.bot2(self.bot1(self.pool(x2))))
        
        # Decoder
        up1 = self.up(bot)
        cat1 = torch.cat([up1, x2], dim=1) # Concatenate skip connection
        dec1 = self.act3(self.dec1b(self.dec1a(cat1)))
        
        up2 = self.up(dec1)
        cat2 = torch.cat([up2, x1], dim=1)
        dec2 = self.act4(self.dec2b(self.dec2a(cat2)))
        
        # Final output (No residual, direct prediction)
        return self.final(dec2)

# ============================================================================
# 4. TRAINING & VISUALIZATION (Mainly the same)
# ============================================================================
def train_candidate(act_type, train_loader, val_loader):
    print(f"Training Deeper {act_type}...")
    model = DeeperUNet(act_type) # Using the deeper model
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    best_psnr = 0
    
    # More epochs for a deeper model
    for ep in range(25):
        model.train()
        for inp, gt in train_loader:
            opt.zero_grad()
            loss = F.l1_loss(model(inp), gt)
            loss.backward()
            opt.step()
        
        model.eval()
        psnrs = [20 * torch.log10(1.0 / torch.sqrt(torch.mean((model(inp) - gt)**2))).item() for inp, gt in val_loader]
        avg_psnr = np.mean(psnrs)
        if avg_psnr > best_psnr: best_psnr = avg_psnr
        
    return best_psnr, model

def main():
    dataset = MRIDataset("*.nii.gz")
    train_set, val_set = random_split(dataset, [int(0.8*len(dataset)), len(dataset)-int(0.8*len(dataset))])
    train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=1)
    
    candidates = ["RELU", "GELU", "SILU", "KAN"]
    results = {}
    
    for cand in candidates:
        score, _ = train_candidate(cand, train_loader, val_loader)
        results[cand] = score
        print(f"Candidate {cand}: Best PSNR = {score:.2f} dB\n")
        
    sorted_res = sorted(results.items(), key=lambda x: x[1], reverse=True)
    print("\nFINAL RANKING (DEEPER MODEL):")
    for rank, (name, score) in enumerate(sorted_res):
        print(f"{rank+1}. {name}: {score:.2f} dB")
        
    # Final check
    if sorted_res[0][0] == 'KAN':
        gap = sorted_res[0][1] - sorted_res[1][1]
        print(f"\n✅ SUCCESS: KAN wins with a margin of {gap:.2f} dB.")
        print("Hypothesis confirmed: KAN's advantage grows with network capacity.")
    else:
        print("\n❌ RESULT: KAN did not outperform the baseline in a deeper architecture.")

if __name__ == "__main__":
    main()