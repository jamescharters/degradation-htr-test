# import h5py
# import numpy as np
# import sigpy as sp
# import sigpy.mri as mr
# import matplotlib.pyplot as plt
# from skimage.metrics import peak_signal_noise_ratio as psnr
# from skimage.metrics import structural_similarity as ssim

# # ==========================================
# # 1. Helper Functions
# # ==========================================

# def fastmri_to_complex(tensor):
#     """Converts fastMRI (..., 2) format to complex numpy array."""
#     if tensor.shape[-1] == 2:
#         return tensor[..., 0] + 1j * tensor[..., 1]
#     return tensor

# def get_coordinate_grid(height, width):
#     """
#     Generates a normalized coordinate grid (x, y) in range [-1, 1].
#     Shape: [Height, Width, 2]
#     """
#     x = np.linspace(-1, 1, width)
#     y = np.linspace(-1, 1, height)
#     X, Y = np.meshgrid(x, y)  # Create 2D grids
    
#     # Stack them to get (H, W, 2)
#     grid = np.stack([X, Y], axis=-1)
#     return grid

# def compute_metrics(gt, pred):
#     """
#     Computes NMSE, PSNR, SSIM between Ground Truth and Prediction.
#     Expects 2D Complex Arrays.
#     """
#     # 1. NMSE (Complex)
#     # Norm of difference squared / Norm of GT squared
#     nmse = np.linalg.norm(gt - pred)**2 / np.linalg.norm(gt)**2

#     # 2. Magnitude for visual metrics
#     gt_mag = np.abs(gt)
#     pred_mag = np.abs(pred)
    
#     # Dynamic range (usually ~1.0 for sensitivity maps)
#     d_range = gt_mag.max() - gt_mag.min()

#     # 3. PSNR
#     val_psnr = psnr(gt_mag, pred_mag, data_range=d_range)

#     # 4. SSIM
#     val_ssim = ssim(gt_mag, pred_mag, data_range=d_range)

#     return nmse, val_psnr, val_ssim

# # ==========================================
# # 2. Main Processing Pipeline
# # ==========================================

# def process_slice_for_kan(file_path, slice_idx):
#     print(f"--- Processing Slice {slice_idx} ---")
    
#     # --- A. Load Data ---
#     with h5py.File(file_path, 'r') as hf:
#         # Load raw k-space
#         kspace_raw = hf['kspace'][slice_idx] 
#         kspace = fastmri_to_complex(kspace_raw) # Shape: [Coils, H, W]

#     print(f"Original K-space Shape: {kspace.shape}")

#     # --- B. Compute ESPIRiT Maps (Ground Truth) ---
#     # We use SigPy to extract the center ACS (Auto-Calibration Signal)
#     # and compute the maps.
#     print("Computing ESPIRiT Maps (this may take a moment)...")
    
#     device = sp.Device(-1) # CPU. Use sp.Device(0) for GPU
    
#     # Standard fastMRI calibration settings
#     # calib_width=24 uses the center 24 lines for maps
#     maps = mr.app.EspiritCalib(
#         kspace, 
#         calib_width=24, 
#         device=device,
#         crop=0.95,  # Mask background noise
#         kernel_width=6,
#         thresh=0.02,
#         show_pbar=False
#     ).run()
    
#     # Move to CPU numpy
#     maps = sp.to_device(maps, sp.cpu_device)
#     print(f"Sensitivity Map Shape: {maps.shape}") # [Coils, H, W]

#     # --- C. Create Input Coordinates ---
#     num_coils, H, W = maps.shape
#     grid = get_coordinate_grid(H, W)
#     print(f"Coordinate Grid Shape: {grid.shape}") # [H, W, 2]

#     # --- D. Prepare Training Pairs (X, y) ---
#     # For a Neural Network (KAN), we usually flatten the images
#     # Input X: (N_pixels, 2) -> (x, y) coordinates
#     # Target Y: (N_pixels, 2 * Coils) -> Real/Imag parts of maps
    
#     # Flatten grid
#     X_train = grid.reshape(-1, 2)
    
#     # Flatten maps and separate Re/Im
#     # Transpose maps to [H, W, Coils] first to align with pixels
#     maps_reshaped = np.moveaxis(maps, 0, -1).reshape(-1, num_coils)
    
#     # Create Real/Imag targets
#     Y_train_real = np.real(maps_reshaped)
#     Y_train_imag = np.imag(maps_reshaped)
#     Y_train = np.concatenate([Y_train_real, Y_train_imag], axis=1)

#     print(f"Training Input X shape: {X_train.shape}")
#     print(f"Training Target Y shape: {Y_train.shape}")
    
#     return maps, grid, X_train, Y_train

# # ==========================================
# # 3. Execution & Evaluation Demo
# # ==========================================

# # CHANGE THIS to your actual file path
# file_path = './data/fastMRI/multicoil_test/file1000082.h5' 

# try:
#     # 1. Run Pipeline
#     # Using slice 20 (middle slices usually have best anatomy)
#     gt_maps, grid, train_x, train_y = process_slice_for_kan(file_path, slice_idx=20)

#     # 2. SIMULATE a "Bad" KAN Prediction
#     # Since we don't have the network yet, let's create a fake prediction
#     # by adding noise to the ground truth.
#     noise = np.random.normal(0, 0.1, gt_maps.shape) + 1j * np.random.normal(0, 0.1, gt_maps.shape)
#     fake_kan_prediction = gt_maps + noise

#     # 3. Evaluate (Your Step 3)
#     print("\n--- Map Quality Evaluation (Simulated) ---")
#     print(f"{'Coil':<5} | {'NMSE':<10} | {'PSNR':<10} | {'SSIM':<10}")
#     print("-" * 50)

#     num_coils = gt_maps.shape[0]
    
#     # Loop through each coil to calculate metrics
#     for i in range(num_coils):
#         nmse, val_psnr, val_ssim = compute_metrics(gt_maps[i], fake_kan_prediction[i])
#         print(f"{i:<5} | {nmse:.4f}     | {val_psnr:.2f} dB   | {val_ssim:.4f}")

#     # 4. Visualization
#     coil_idx = 0
#     plt.figure(figsize=(12, 6))

#     # Ground Truth Magnitude
#     plt.subplot(2, 3, 1)
#     plt.imshow(np.abs(gt_maps[coil_idx]), cmap='gray')
#     plt.title(f"GT Coil {coil_idx} Mag")
#     plt.axis('off')

#     # Fake Prediction Magnitude
#     plt.subplot(2, 3, 2)
#     plt.imshow(np.abs(fake_kan_prediction[coil_idx]), cmap='gray')
#     plt.title(f"Pred Coil {coil_idx} Mag")
#     plt.axis('off')
    
#     # Error Map
#     plt.subplot(2, 3, 3)
#     plt.imshow(np.abs(gt_maps[coil_idx] - fake_kan_prediction[coil_idx]), cmap='hot')
#     plt.title("Error Magnitude")
#     plt.axis('off')

#     # Ground Truth Phase
#     plt.subplot(2, 3, 4)
#     plt.imshow(np.angle(gt_maps[coil_idx]), cmap='jet')
#     plt.title(f"GT Coil {coil_idx} Phase")
#     plt.axis('off')

#     # Fake Prediction Phase
#     plt.subplot(2, 3, 5)
#     plt.imshow(np.angle(fake_kan_prediction[coil_idx]), cmap='jet')
#     plt.title(f"Pred Coil {coil_idx} Phase")
#     plt.axis('off')

#     plt.tight_layout()
#     plt.show()

# except OSError:
#     print(f"File not found: {file_path}")
#     print("Please set 'file_path' to a valid fastMRI multi-coil .h5 file.")
# except Exception as e:
#     print(f"An error occurred: {e}")









########## KAN





import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class KANLayer(nn.Module):
    """
    An efficient B-Spline KAN Layer.
    """
    def __init__(self, in_features, out_features, grid_size=5, spline_order=3, scale_noise=0.1, scale_base=1.0, scale_spline=1.0, base_activation=torch.nn.SiLU, grid_eps=0.02, grid_range=[-1, 1]):
        super(KANLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = ((torch.arange(-spline_order, grid_size + spline_order + 1) * h) + grid_range[0]).expand(in_features, -1).contiguous()
        self.register_buffer("grid", grid)

        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = nn.Parameter(torch.Tensor(out_features, in_features * (grid_size + spline_order)))
        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_weight, a=np.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (torch.rand(self.grid_size + 1, self.in_features, self.out_features) - 1 / 2) * self.scale_noise / self.grid_size
            self.spline_weight.data.copy_((self.scale_spline if self.scale_spline is not None else 1.0) * self.curve2coeff(self.grid.T[self.spline_order : -self.spline_order], noise))

    def b_splines(self, x: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features
        grid: torch.Tensor = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (x - grid[:, : -(k + 1)]) / (grid[:, k:-1] - grid[:, : -(k + 1)]) * bases[:, :, :-1] + \
                    (grid[:, k + 1 :] - x) / (grid[:, k + 1 :] - grid[:, 1:(-k)]) * bases[:, :, 1:]
        assert bases.size() == (x.size(0), self.in_features, self.grid_size + self.spline_order)
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        A = self.b_splines(x).transpose(0, 1)
        B = y.transpose(0, 1)
        solution = torch.linalg.lstsq(A, B).solution
        result = solution.permute(2, 0, 1)
        return result.contiguous().view(self.out_features, -1)

    def forward(self, x: torch.Tensor):
        original_shape = x.shape
        x = x.view(-1, self.in_features)

        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(self.b_splines(x).view(x.size(0), -1), self.spline_weight)
        output = base_output + spline_output
        
        output = output.view(*original_shape[:-1], self.out_features)
        return output

class KAN(nn.Module):
    def __init__(self, layers_hidden, grid_size=5, spline_order=3):
        super(KAN, self).__init__()
        self.layers = nn.ModuleList()
        for in_dim, out_dim in zip(layers_hidden[:-1], layers_hidden[1:]):
            self.layers.append(
                KANLayer(in_dim, out_dim, grid_size=grid_size, spline_order=spline_order)
            )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x





########## END KAN















import h5py
import numpy as np
import sigpy as sp
import sigpy.mri as mr
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# ==========================================
# 1. Helper Functions
# ==========================================

def fastmri_to_complex(tensor):
    """Converts fastMRI (..., 2) format to complex numpy array."""
    if tensor.shape[-1] == 2:
        return tensor[..., 0] + 1j * tensor[..., 1]
    return tensor

def get_coordinate_grid(height, width):
    """
    Generates a normalized coordinate grid (x, y) in range [-1, 1].
    Returns shape: [Height, Width, 2]
    """
    x = np.linspace(-1, 1, width)
    y = np.linspace(-1, 1, height)
    X, Y = np.meshgrid(x, y) 
    grid = np.stack([X, Y], axis=-1)
    return grid

def compute_metrics(gt, pred):
    """
    Computes NMSE (Complex), PSNR (Mag), SSIM (Mag).
    gt, pred: 2D Complex Arrays [Height, Width]
    """
    # 1. NMSE (Normalized Mean Squared Error) - Complex
    # Formula: || GT - Pred ||^2 / || GT ||^2
    nmse = np.linalg.norm(gt - pred)**2 / np.linalg.norm(gt)**2

    # 2. Prepare Magnitude for visual metrics
    gt_mag = np.abs(gt)
    pred_mag = np.abs(pred)
    
    # Dynamic range (Sensitivity maps are usually 0.0 to 1.0)
    d_range = gt_mag.max() - gt_mag.min()

    # 3. PSNR
    val_psnr = psnr(gt_mag, pred_mag, data_range=d_range)

    # 4. SSIM
    val_ssim = ssim(gt_mag, pred_mag, data_range=d_range)

    return nmse, val_psnr, val_ssim

# ==========================================
# 2. Main Processing Pipeline
# ==========================================

def prepare_data_and_evaluate(file_path, slice_idx=20):
    print(f"--- Processing Slice {slice_idx} ---")
    
    # ---------------------------------------------------------
    # PART A: Load Data & Compute Ground Truth (ESPIRiT)
    # ---------------------------------------------------------
    with h5py.File(file_path, 'r') as hf:
        kspace_raw = hf['kspace'][slice_idx] 
        kspace = fastmri_to_complex(kspace_raw) # [Coils, H, W]

    print(f"Original K-space Shape: {kspace.shape}")
    print("Computing ESPIRiT Maps (Ground Truth)...")
    
    # Use SigPy to compute maps from the center calibration region
    # device=-1 is CPU. If you have a GPU/CUDA, use device=0 for speed.
    device = sp.Device(-1) 
    
    # 'calib_width' extracts the center lines (ACS) automatically
    maps_gt = mr.app.EspiritCalib(
        kspace, 
        calib_width=24, # DEVNOTE: 24 is standard/default
        device=device,
        crop=0.95,  # Masks background noise (important for clear maps)
        kernel_width=6,
        thresh=0.02,
        show_pbar=False
    ).run()
    
    maps_gt = sp.to_device(maps_gt, sp.cpu_device)
    print(f"Ground Truth Maps Shape: {maps_gt.shape}") # [Coils, H, W]

    # ---------------------------------------------------------
    # PART B: Prepare Training Tensors for KAN
    # ---------------------------------------------------------
    # Neural Networks expect inputs (X) and targets (Y).
    # X = (x,y) coordinates
    # Y = Sensitivity values (Real, Imag)
    
    num_coils, H, W = maps_gt.shape
    
    # 1. Create Grid Input (X)
    grid = get_coordinate_grid(H, W) # [H, W, 2]
    X_train = grid.reshape(-1, 2)    # Flatten to [N_pixels, 2]

    # 2. Create Target Output (Y)
    # Move coils to last dim: [H, W, Coils]
    maps_transposed = np.moveaxis(maps_gt, 0, -1)
    maps_flat = maps_transposed.reshape(-1, num_coils) # [N_pixels, Coils]
    
    # Split Complex into Real/Imag parts (KANs usually output Real numbers)
    # Shape becomes [N_pixels, Coils * 2]
    Y_train = np.concatenate([np.real(maps_flat), np.imag(maps_flat)], axis=1)

    print(f"KAN Training Input X: {X_train.shape}")
    print(f"KAN Training Target Y: {Y_train.shape}")
    
    from torch.utils.data import TensorDataset, DataLoader

    print("\n[!] Training KAN Model (Batch Optimized)...")

    # 1. Setup Device
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    if device.type == 'cpu' and torch.cuda.is_available(): device = torch.device('cuda')
    print(f"    Using device: {device}")

    # 2. Convert Data to Torch Tensors
    t_X = torch.tensor(X_train, dtype=torch.float32).to(device)
    t_Y = torch.tensor(Y_train, dtype=torch.float32).to(device)

    # 3. Create DataLoader (The Speed Fix)
    # Batch size of 4096 or 8192 is sweet spot for M1/M2 chips
    dataset = TensorDataset(t_X, t_Y)
    dataloader = DataLoader(dataset, batch_size=8192, shuffle=True)
    
    # 4. Initialize KAN Model
    # grid_size=20 gives the detail you need. 128 width gives capacity.
    model = KAN(layers_hidden=[2, 128, Y_train.shape[1]], grid_size=20, spline_order=3).to(device)

    # 5. Optimizer
    # We use a slightly higher LR because we are batching
    optimizer = optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99) # Smooth decay

    # 6. Training Loop
    epochs = 100 # Reduced from 2000. Your loss logs show 300 is enough!
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        
        # Mini-batch loop
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = nn.MSELoss()(pred, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        scheduler.step()
        
        if epoch % 10 == 0:
            avg_loss = epoch_loss / len(dataloader)
            print(f"    Epoch {epoch}/{epochs} | Avg Loss: {avg_loss:.6f}")

    print("    Training complete.")

    # 7. Inference (Get final maps)
    # For inference, we can do full batch usually, but let's be safe on RAM
    model.eval()
    with torch.no_grad():
        # Process in chunks to avoid memory spike on inference
        predictions = []
        for i in range(0, len(t_X), 10000):
            batch = t_X[i : i+10000]
            predictions.append(model(batch).cpu().numpy())
        
        predicted_tensor = np.concatenate(predictions, axis=0)
    
    # 8. Reshape back to Images
    n_out = predicted_tensor.shape[1] // 2
    pred_real = predicted_tensor[:, :n_out]
    pred_imag = predicted_tensor[:, n_out:]
    pred_complex_flat = pred_real + 1j * pred_imag
    
    maps_predicted_HWC = pred_complex_flat.reshape(H, W, num_coils)
    maps_predicted = np.moveaxis(maps_predicted_HWC, -1, 0)
    
    print(f"    Prediction Shape: {maps_predicted.shape}")

    # ---------------------------------------------------------
    # PART D: Evaluation (Step 3)
    # ---------------------------------------------------------
    print("\n--- Map Quality Evaluation ---")
    print(f"{'Coil':<5} | {'NMSE':<10} | {'PSNR':<10} | {'SSIM':<10}")
    print("-" * 50)

    for i in range(num_coils):
        nmse, val_psnr, val_ssim = compute_metrics(maps_gt[i], maps_predicted[i])
        print(f"{i:<5} | {nmse:.4f}     | {val_psnr:.2f} dB   | {val_ssim:.4f}")

    return kspace, maps_gt, maps_predicted, grid

# ==========================================
# 3. Visualization
# ==========================================

def visualize_results(gt, pred, coil_idx=0):
    """Visualizes Magnitude, Phase, and Error for a specific coil."""
    plt.figure(figsize=(12, 6))

    # GT Mag
    plt.subplot(2, 3, 1)
    plt.imshow(np.abs(gt[coil_idx]), cmap='gray')
    plt.title(f"GT Coil {coil_idx} Mag")
    plt.axis('off')

    # Pred Mag
    plt.subplot(2, 3, 2)
    plt.imshow(np.abs(pred[coil_idx]), cmap='gray')
    plt.title(f"Pred Coil {coil_idx} Mag")
    plt.axis('off')

    # Error Mag (Difference)
    plt.subplot(2, 3, 3)
    # Use same scale as image for fair comparison, or dynamic for detail
    err = np.abs(gt[coil_idx] - pred[coil_idx])
    plt.imshow(err, cmap='inferno') 
    plt.title("Error Magnitude")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis('off')

    # GT Phase
    plt.subplot(2, 3, 4)
    plt.imshow(np.angle(gt[coil_idx]), cmap='twilight') # twilight is good for phase
    plt.title(f"GT Coil {coil_idx} Phase")
    plt.axis('off')

    # Pred Phase
    plt.subplot(2, 3, 5)
    plt.imshow(np.angle(pred[coil_idx]), cmap='twilight')
    plt.title(f"Pred Coil {coil_idx} Phase")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

def visualize_masked(gt, pred, coil_idx=0):
    # 1. Create a binary mask from Ground Truth Magnitude
    # Threshold: typically 10% of the max intensity, but can also try 5% for knees
    thresh = 0.05 * np.abs(gt).max()
    mask = np.abs(gt) > thresh

    # 2. Apply Mask to Prediction
    # We only keep the prediction where the anatomy actually exists
    pred_masked = pred * mask
    gt_masked = gt * mask

    plt.figure(figsize=(12, 6))

    # GT Mag
    plt.subplot(2, 3, 1)
    plt.imshow(np.abs(gt_masked[coil_idx]), cmap='gray')
    plt.title(f"GT Coil {coil_idx} Mag (Masked)")
    plt.axis('off')

    # Pred Mag
    plt.subplot(2, 3, 2)
    plt.imshow(np.abs(pred_masked[coil_idx]), cmap='gray')
    plt.title(f"Pred Coil {coil_idx} Mag (Masked)")
    plt.axis('off')

    # Error Mag (Difference)
    plt.subplot(2, 3, 3)
    # We boost the contrast of the error map to see subtle flaws
    err = np.abs(gt_masked[coil_idx] - pred_masked[coil_idx])
    plt.imshow(err, cmap='inferno', vmin=0, vmax=0.1*np.abs(gt).max()) 
    plt.title("Error Magnitude (Contrast Boosted)")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis('off')

    # GT Phase
    plt.subplot(2, 3, 4)
    plt.imshow(np.angle(gt_masked[coil_idx]), cmap='twilight') 
    plt.title(f"GT Coil {coil_idx} Phase")
    plt.axis('off')

    # Pred Phase
    plt.subplot(2, 3, 5)
    plt.imshow(np.angle(pred_masked[coil_idx]), cmap='twilight')
    plt.title(f"Pred Coil {coil_idx} Phase")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# ==========================================
# 4. Run Script
# ==========================================

# CHANGE THIS PATH
file_path = './data/fastMRI/multicoil_test/file1000082.h5' 

try:
    # Run the pipeline
    kspace, gt_maps, pred_maps, _ = prepare_data_and_evaluate(file_path, slice_idx=18)
    
    # Visualize Coil 0
    visualize_masked(gt_maps, pred_maps, coil_idx=0)

except OSError:
    print(f"Error: Could not open file '{file_path}'. Check path.")
except Exception as e:
    print(f"An error occurred: {e}")




# ==========================================
# STEP 4: SENSE Reconstruction & Comparison
# ==========================================

print("\n--- STEP 4: SENSE Reconstruction ---")

# 1. Create Undersampling Mask (4x Acceleration)
# Simulates a scan that is 4x faster
C, H, W = kspace.shape
mask = np.zeros((H, W), dtype=np.float32)
accel_factor = 4
center_fraction = 0.08 # Keep 8% of center lines for contrast

# Calculate center region
num_low_freq = int(round(W * center_fraction))
pad = (W - num_low_freq + 1) // 2
mask[:, pad:pad + num_low_freq] = 1 # Fully sample center

# Undersample the rest (outer k-space)
outer_indices = np.concatenate([np.arange(0, pad), np.arange(pad + num_low_freq, W)])
# Select every 4th line in the outer region
mask[:, outer_indices[::accel_factor]] = 1

print(f"Mask created: {accel_factor}x Acceleration")

# 2. Generate Undersampled K-Space (The Input Data)
kspace_undersampled = kspace * mask

# 3. Define Helper for SENSE Reconstruction
def run_sense(y, mps, lamda=0.01):
    """
    Runs SENSE reconstruction using SigPy.
    y: Undersampled k-space [Coils, H, W]
    mps: Sensitivity Maps [Coils, H, W]
    """
    print("  Running SENSE solver...")
    # SigPy SenseRecon App:
    # Solves: min_x || P F S x - y ||^2 + lamda || x ||^2
    # We use CPU device (-1) to be safe, as SigPy CUDA/MPS mixing can be tricky
    return mr.app.SenseRecon(
        y, 
        mps, 
        lamda=lamda, 
        device=sp.Device(-1), # CPU
        show_pbar=False
    ).run()

# 4. Run Reconstructions
print("A. Reconstructing with GT (ESPIRiT) maps...")
img_espirit = run_sense(kspace_undersampled, gt_maps)

print("B. Reconstructing with KAN (Predicted) maps...")
img_kan = run_sense(kspace_undersampled, pred_maps)

# 5. Create "Gold Standard" Reference
# This is the Root-Sum-of-Squares of the FULLY sampled data
print("C. Computing Reference (RSS of full data)...")
img_ref = sp.rss(sp.ifft(kspace, axes=(-2, -1)), axes=0)
# Normalize images for fair metric comparison
img_ref = img_ref / np.abs(img_ref).max()
img_espirit = img_espirit / np.abs(img_espirit).max()
img_kan = img_kan / np.abs(img_kan).max()

# 6. Compute Image Metrics
def get_img_metrics(ref, img):
    ref_mag = np.abs(ref)
    img_mag = np.abs(img)
    d_range = ref_mag.max()
    p = psnr(ref_mag, img_mag, data_range=d_range)
    s = ssim(ref_mag, img_mag, data_range=d_range)
    return p, s

psnr_esp, ssim_esp = get_img_metrics(img_ref, img_espirit)
psnr_kan, ssim_kan = get_img_metrics(img_ref, img_kan)

print("\n--- Final Reconstruction Results ---")
print(f"Method   | PSNR (dB) | SSIM")
print("-" * 30)
print(f"ESPIRiT  | {psnr_esp:.2f}     | {ssim_esp:.4f}")
print(f"KAN      | {psnr_kan:.2f}     | {ssim_kan:.4f}")

# 7. Visualization
plt.figure(figsize=(12, 8))

# Reference
plt.subplot(2, 3, 1)
plt.imshow(np.abs(img_ref), cmap='gray', vmin=0, vmax=0.8)
plt.title("Reference (Fully Sampled)")
plt.axis('off')

# ESPIRiT Recon
plt.subplot(2, 3, 2)
plt.imshow(np.abs(img_espirit), cmap='gray', vmin=0, vmax=0.8)
plt.title(f"ESPIRiT Recon\nPSNR: {psnr_esp:.2f}")
plt.axis('off')

# KAN Recon
plt.subplot(2, 3, 3)
plt.imshow(np.abs(img_kan), cmap='gray', vmin=0, vmax=0.8)
plt.title(f"KAN Recon\nPSNR: {psnr_kan:.2f}")
plt.axis('off')

# Undersampling Mask
plt.subplot(2, 3, 4)
plt.imshow(mask, cmap='gray')
plt.title("Undersampling Mask (4x)")
plt.axis('off')

# Error: ESPIRiT
plt.subplot(2, 3, 5)
plt.imshow(np.abs(img_ref - img_espirit), cmap='inferno', vmin=0, vmax=0.1)
plt.title("Error: ESPIRiT")
plt.axis('off')

# Error: KAN
plt.subplot(2, 3, 6)
plt.imshow(np.abs(img_ref - img_kan), cmap='inferno', vmin=0, vmax=0.1)
plt.title("Error: KAN")
plt.axis('off')

plt.tight_layout()
plt.show()