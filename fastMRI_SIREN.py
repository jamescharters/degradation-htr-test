# --- START OF FILE visualise_h5.py ---

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import trange

# --- 1. SIREN Model Definition ---
# (This section is unchanged and correct)
class SineLayer(nn.Module):
    """A linear layer followed by a sine activation function."""
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
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

class Siren(nn.Module):
    """The full SIREN model."""
    def __init__(self, in_features, hidden_features, hidden_layers, out_features,
                 outermost_linear=True, omega_0=30):
        super().__init__()
        
        self.net = []
        self.net.append(SineLayer(in_features, hidden_features, 
                                  is_first=True, omega_0=omega_0))
        for _ in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features, 
                                      is_first=False, omega_0=omega_0))
        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / omega_0, 
                                              np.sqrt(6 / hidden_features) / omega_0)
            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features, 
                                      is_first=False, omega_0=omega_0))
        
        self.net = nn.Sequential(*self.net)
    
    def forward(self, coords):
        return self.net(coords)

# --- 2. Helper Functions ---
def get_mgrid(shape, dim=2):
    """Generates a 2D grid of coordinates normalized to [-1, 1]."""
    h, w = shape
    tensors = (torch.linspace(-1, 1, steps=h), torch.linspace(-1, 1, steps=w))
    mgrid = torch.stack(torch.meshgrid(*tensors, indexing="ij"), dim=-1)
    return mgrid.reshape(-1, dim)

def create_undersampling_mask(shape, acceleration, num_center_lines):
    """Creates a 1D Cartesian undersampling mask."""
    num_cols = shape[-1]
    center_fraction = num_center_lines / num_cols
    
    num_sampled_lines = int(num_cols / acceleration)
    
    mask = np.zeros(num_cols, dtype=np.float32)
    
    center_start = (num_cols - num_center_lines) // 2
    center_end = center_start + num_center_lines
    mask[center_start:center_end] = 1
    
    num_outer_lines_to_sample = num_sampled_lines - num_center_lines
    if num_outer_lines_to_sample > 0:
        outer_indices = np.setdiff1d(np.arange(num_cols), np.arange(center_start, center_end))
        sampled_indices = np.random.choice(outer_indices, num_outer_lines_to_sample, replace=False)
        mask[sampled_indices] = 1
        
    return torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0)

def apply_mask(kspace, mask):
    """Applies the undersampling mask to k-space data."""
    return kspace * mask

def fft_2d(image):
    """2D FFT with appropriate shifts."""
    return torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(image, dim=(-2, -1))), dim=(-2, -1))

def ifft_2d(kspace):
    """2D iFFT with appropriate shifts."""
    return torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(kspace, dim=(-2, -1))), dim=(-2, -1))

# --- 3. Main Script ---

if __name__ == '__main__':
    # --- Device Configuration ---
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    print(f"Using device: {device}")

    # --- Data Loading ---
    try:
        with h5py.File('./data/fastMRI/multicoil_test/file1000136.h5', 'r') as f:
            kspace = f['kspace'][:]
            mid_slice = kspace.shape[0] // 2
            kspace_slice_np = kspace[mid_slice]
    except FileNotFoundError:
        print("Error: H5 file not found. Make sure 'file1000136.h5' is in './data/fastMRI/multicoil_test/'")
        exit()


    # --- Prepare Data for PyTorch ---
    kspace_slice_torch = torch.from_numpy(kspace_slice_np).to(torch.complex64).to(device)
    # We scale the k-space to prevent exploding gradients.
    k_max = torch.max(torch.abs(kspace_slice_torch))
    kspace_slice_torch_normalized = kspace_slice_torch / k_max

    shape = kspace_slice_torch_normalized.shape
    num_coils, height, width = shape

    # --- A. Original Fully-Sampled Reconstruction ---
    coil_images_np = np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(kspace_slice_np, axes=(-2, -1))), axes=(-2, -1))
    ground_truth_rss = np.sqrt(np.sum(np.abs(coil_images_np) ** 2, axis=0))

    # --- B. Create Undersampled Data ---
    ACCELERATION = 4
    CENTER_LINES = 24
    mask = create_undersampling_mask(shape, ACCELERATION, CENTER_LINES).to(device)
    #undersampled_kspace = apply_mask(kspace_slice_torch, mask)
    undersampled_kspace_normalized = apply_mask(kspace_slice_torch_normalized, mask)
    
    zero_filled_images = ifft_2d(undersampled_kspace_normalized) #ifft_2d(undersampled_kspace)
    zero_filled_rss = torch.sqrt(torch.sum(torch.abs(zero_filled_images) ** 2, axis=0)).cpu().numpy()

    # --- C. SIREN Reconstruction ---
    LEARNING_RATE = 1e-4
    EPOCHS = 500
    
    model = Siren(in_features=2, 
                  out_features=num_coils * 2, 
                  hidden_features=256, 
                  hidden_layers=3, 
                  outermost_linear=True).to(device)
                  
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    model_input = get_mgrid((height, width)).to(device)

    print("\nStarting SIREN training...")
    tbar = trange(EPOCHS)
    for epoch in tbar:
        model_output = model(model_input)
        
        model_output = model_output.view(height, width, num_coils, 2)
        model_output = model_output.permute(2, 0, 1, 3)
        
        complex_image = torch.complex(model_output[..., 0], model_output[..., 1])

        predicted_kspace = fft_2d(complex_image)
        predicted_kspace_masked = apply_mask(predicted_kspace, mask)

        # ***** FIX: Calculate loss on real and imaginary parts separately *****
        # mse_loss does not support complex numbers directly.
        # The correct loss is the sum of MSE on real and imaginary components.
        loss_real = torch.nn.functional.mse_loss(predicted_kspace_masked.real, undersampled_kspace_normalized.real)
        loss_imag = torch.nn.functional.mse_loss(predicted_kspace_masked.imag, undersampled_kspace_normalized.imag)
        loss = loss_real + loss_imag
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            tbar.set_description(f"Epoch {epoch}, Loss: {loss.item():.6f}")

    # --- Get Final SIREN Result ---
    model.eval()
    with torch.no_grad():
        final_output = model(model_input)
        final_output = final_output.view(height, width, num_coils, 2).permute(2, 0, 1, 3)
        final_complex_image = torch.complex(final_output[..., 0], final_output[..., 1])
        siren_rss = torch.sqrt(torch.sum(torch.abs(final_complex_image) ** 2, axis=0)).cpu().numpy()

    # --- Display All Results ---
    plt.figure(figsize=(20, 10))
    
    plt.subplot(2, 3, 1)
    plt.imshow(np.log(1 + np.abs(kspace_slice_np[0])), cmap='gray')
    plt.title('K-space (coil 0)')
    plt.axis('off')

    plt.subplot(2, 3, 2)
    # Display the original (non-normalized) undersampled k-space for visual consistency
    plt.imshow(np.abs(apply_mask(kspace_slice_torch, mask)[0].cpu().numpy()), cmap='gray')
    plt.title(f'Undersampled K-space ({ACCELERATION}x)')
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.imshow(np.abs(ifft_2d(kspace_slice_torch)[0].cpu().numpy()), cmap='gray')
    plt.title('Original Single Coil Image')
    plt.axis('off')

    plt.subplot(2, 3, 4)
    plt.imshow(ground_truth_rss, cmap='gray')
    plt.title('Ground Truth (Fully Sampled RSS)')
    plt.axis('off')
    
    plt.subplot(2, 3, 5)
    # We can plot the un-normalized zero-filled recon for a direct comparison with ground-truth scale
    zero_filled_rss_unnormalized = torch.sqrt(torch.sum(torch.abs(ifft_2d(apply_mask(kspace_slice_torch, mask))) ** 2, axis=0)).cpu().numpy()
    plt.imshow(zero_filled_rss_unnormalized, cmap='gray')
    plt.title(f'Zero-Filled Recon ({ACCELERATION}x)')
    plt.axis('off')
    
    plt.subplot(2, 3, 6)
    plt.imshow(siren_rss, cmap='gray')
    plt.title('SIREN Reconstruction')
    plt.axis('off')
    
    plt.tight_layout()
    plt.suptitle("Comparison of MRI Reconstruction Methods", fontsize=16)
    plt.subplots_adjust(top=0.92)
    plt.show()