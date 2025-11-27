import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.fft import fft2, ifft2
import nibabel as nib
from skimage.transform import resize

# ============================================================================
# 1. DATA SETUP
# ============================================================================
def get_data():
    try:
        nii = nib.load("IXI002-Guys-0828-T2.nii")
        data = nii.get_fdata()[:, :, nii.shape[2]//2]
        data = resize(np.rot90(data), (128, 128))
        img = torch.tensor(data, dtype=torch.float32)
        img = (img - img.min()) / (img.max() - img.min())
    except:
        x, y = np.meshgrid(np.linspace(-1,1,128), np.linspace(-1,1,128))
        img = torch.tensor((x**2 + y**2 < 0.6).astype(float), dtype=torch.float32)
    return img

def get_training_pair(img):
    H, W = img.shape
    kspace = fft2(img)
    mask = torch.zeros(H, W)
    center = H // 2
    cw = int(H * 0.08)
    mask[center-cw:center+cw, :] = 1 
    
    idxs = np.concatenate([np.arange(0, center-cw), np.arange(center+cw, H)])
    keep = np.random.choice(idxs, int(len(idxs)*0.25), replace=False)
    mask[keep, :] = 1
    
    corrupted_k = kspace * mask
    corrupted_img = ifft2(corrupted_k).real
    corrupted_img = (corrupted_img - corrupted_img.min()) / (corrupted_img.max() - corrupted_img.min())
    
    return corrupted_img.unsqueeze(0).unsqueeze(0), img.unsqueeze(0).unsqueeze(0)

# ============================================================================
# 2. STABLE RBF-KAN LAYER
# ============================================================================
class RBFKANLayer(nn.Module):
    def __init__(self, channels, grid=8):
        super().__init__()
        self.grid = grid
        
        # 1. Normalize to [-1, 1]
        self.norm = nn.InstanceNorm2d(channels, affine=False)
        
        # 2. Create Grid Centers (mu)
        # Fixed centers from -1 to 1
        grid_centers = torch.linspace(-1, 1, grid)
        self.register_buffer("grid_centers", grid_centers)
        
        # 3. Learnable Weights for each RBF basis
        # We use a 1x1 Conv to weight the basis functions
        # Input: Grid Size. Output: 1 (Summed up). 
        # We do this PER CHANNEL (groups=channels) implies we need careful shaping.
        # Alternative: Just learn parameters directly.
        self.coef = nn.Parameter(torch.randn(1, channels, 1, 1, grid) * 0.01) # Init small!
        
        # Base weight
        self.base_scale = nn.Parameter(torch.ones(1, channels, 1, 1))

    def forward(self, x):
        # x: (B, C, H, W)
        
        # Normalize and Clamp to ensure we stay on the grid
        x_norm = torch.tanh(self.norm(x)) 
        
        # Base Path (Residual)
        base = F.silu(x) * self.base_scale
        
        # RBF Expansion
        # (B, C, H, W, 1) - (Grid,) -> (B, C, H, W, Grid)
        x_uns = x_norm.unsqueeze(-1)
        centers = self.grid_centers
        
        # Gaussian RBF: exp(-(x - mu)^2 / sigma)
        # sigma = 2/grid roughly
        sigma = 2.0 / self.grid
        rbf_bases = torch.exp(-((x_uns - centers) ** 2) / (2 * sigma ** 2))
        
        # Weighted Sum
        # (B, C, H, W, Grid) * (1, C, 1, 1, Grid) -> Sum over Grid
        kan_out = torch.sum(rbf_bases * self.coef, dim=-1)
        
        return base + kan_out

# ============================================================================
# 3. ARCHITECTURES
# ============================================================================

class SimpleUNet(nn.Module):
    def __init__(self, use_kan=False):
        super().__init__()
        
        self.enc1 = nn.Conv2d(1, 32, 3, padding=1)
        self.enc2 = nn.Conv2d(32, 64, 3, padding=1)
        self.dec1 = nn.Conv2d(64, 32, 3, padding=1)
        self.dec2 = nn.Conv2d(32, 1, 3, padding=1)
        
        if use_kan:
            # Using the Stable RBF Layer
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
        # Encoder
        x1 = self.enc1(x)
        x1 = self.act1(x1)
        
        x2 = self.pool(x1)
        x2 = self.enc2(x2)
        x2 = self.act2(x2)
        
        # Decoder
        x3 = self.up(x2)
        x3 = self.dec1(x3)
        
        # Skip Connection
        x3 = x3 + x1
        x3 = self.act3(x3)
        
        out = self.dec2(x3)
        return x + out

# ============================================================================
# 4. TRAIN
# ============================================================================
def train_model(type_name, clean_img):
    if type_name == "CNN": model = SimpleUNet(use_kan=False)
    else: model = SimpleUNet(use_kan=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    corrupted, clean = get_training_pair(clean_img)
    
    print(f"Training {type_name}...")
    
    # Using L1 Loss for sharper edges (better for deblurring)
    loss_fn = nn.L1Loss()
    
    for i in range(400): # Slightly longer training
        optimizer.zero_grad()
        pred = model(corrupted)
        loss = loss_fn(pred, clean)
        loss.backward()
        optimizer.step()
        
        if i % 100 == 0: print(f"  {i}: {loss.item():.6f}")
        
    return pred.detach(), model







def plot_learned_functions(model):
    print("Extracting Learned Activation Functions...")
    
    # Create a dummy input range from -3 to 3 (where normalized data lives)
    x = torch.linspace(-3, 3, 100).view(1, 1, 1, 1, 100)
    
    # Extract the first KAN layer
    # In SimpleUNet -> self.act1 is an RBFKANLayer
    layer = model.act1
    
    # We want to see what the first 5 channels learned
    num_channels_to_plot = 5
    
    # Prepare plot
    plt.figure(figsize=(15, 5))
    
    # Run the RBF logic manually for visualization
    # 1. Normalize range
    # Note: We skip the InstanceNorm for viz because we are sweeping x manually
    x_norm = torch.tanh(x) 
    
    # 2. RBF Expansion
    centers = layer.grid_centers.view(1, 1, 1, 1, 1, -1) # (1,1,1,1,1, Grid)
    x_uns = x_norm.unsqueeze(-1)
    sigma = 2.0 / layer.grid
    rbf_bases = torch.exp(-((x_uns - centers) ** 2) / (2 * sigma ** 2))
    
    # 3. Weighted Sum
    # coef shape: (1, C, 1, 1, Grid)
    # We pick specific channels
    
    for i in range(num_channels_to_plot):
        # Get coeffs for channel i
        coef = layer.coef[0, i, 0, 0, :] # (Grid,)
        base_scale = layer.base_scale[0, i, 0, 0]
        
        # Calculate curve
        # Base path: SiLU * scale
        y_base = F.silu(x) * base_scale
        
        # Spline path: Sum(basis * coef)
        # rbf_bases: (..., 100, Grid)
        y_spline = torch.sum(rbf_bases * coef, dim=-1)
        
        y_total = y_base + y_spline
        
        # Plot
        y_np = y_total.detach().flatten().numpy()
        x_np = x.flatten().numpy()
        
        plt.subplot(1, 5, i+1)
        plt.plot(x_np, y_np, 'r-', linewidth=3, label='Learned KAN')
        plt.plot(x_np, F.relu(torch.tensor(x_np)).numpy(), 'k--', alpha=0.3, label='Std ReLU')
        plt.title(f"Channel {i}")
        plt.grid(True, alpha=0.3)
        if i == 0: plt.legend()

    plt.suptitle("What did the U-KAN learn? (Red = KAN Act, Grey = ReLU)", fontsize=14)
    plt.tight_layout()
    plt.savefig("learned_activations.png")
    plt.show()




# ============================================================================
# 5. RUN
# ============================================================================
def main():
    img = get_data()
    
    cnn_out, cnn_model = train_model("CNN", img)
    kan_out, kan_model = train_model("KAN", img)
    
    corrupted, gt = get_training_pair(img)
    
    def psnr(pred, gt):
        mse = torch.mean((pred - gt)**2)
        return 20 * np.log10(1.0 / np.sqrt(mse.item()))
    
    p_in = psnr(corrupted, gt)
    p_cnn = psnr(cnn_out, gt)
    p_kan = psnr(kan_out, gt)
    
    plt.figure(figsize=(16, 6))
    plt.subplot(1,4,1); plt.imshow(corrupted[0,0], cmap='gray'); plt.title(f"Input\n{p_in:.1f}dB")
    plt.subplot(1,4,2); plt.imshow(cnn_out[0,0], cmap='gray'); plt.title(f"Standard U-Net\n{p_cnn:.1f}dB")
    plt.subplot(1,4,3); plt.imshow(kan_out[0,0], cmap='gray'); plt.title(f"U-KAN (RBF)\n{p_kan:.1f}dB")
    
    plt.subplot(1,4,4)
    diff = torch.abs(kan_out - cnn_out)[0,0]
    plt.imshow(diff, cmap='inferno')
    plt.title("Diff (KAN - CNN)")
    
    plt.tight_layout()
    plt.savefig("stable_ukan_result.png")
    plt.show()
    
    print("\nFINAL VERDICT:")
    print(f"CNN Params: {sum(p.numel() for p in cnn_model.parameters())}")
    print(f"KAN Params: {sum(p.numel() for p in kan_model.parameters())}")
    print("-" * 30)
    if p_kan > p_cnn:
        print("✅ SUCCESS: U-KAN (RBF) outperformed Standard U-Net.")
    else:
        print("RESULT: Comparable. Check if KAN edges are sharper.")


if __name__ == "__main__":
    main()