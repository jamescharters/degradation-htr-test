#!/usr/bin/env python3
"""
Physics-Informed Neural Network for a SIMPLE TARGET (an expanding square)
This script is a simplified version of V3 to test the core PINN logic.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

print("=" * 60)
print("QUILL PHYSICS PINN - SIMPLE TARGET TRAINING")
print("=" * 60)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"✓ Using device: {device}\n")

# ============================================
# MODEL AND GRADIENT FUNCTIONS (Identical to V3)
# ============================================

class QuillPINN_V3(nn.Module):
    def __init__(self, hidden_dim=256, num_layers=8, time_encoding_dim=32):
        super().__init__()
        self.time_encoding_dim = time_encoding_dim
        input_dim = 2 + time_encoding_dim
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.Tanh())
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_dim, 1))
        self.network = nn.Sequential(*layers)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
        print(f"✓ Model created (Params: {sum(p.numel() for p in self.parameters()):,})")
    
    def encode_time(self, t):
        freqs = torch.arange(self.time_encoding_dim // 2, device=t.device, dtype=torch.float32)
        freqs = 2.0 ** freqs
        t_expanded = t.unsqueeze(-1) * freqs.unsqueeze(0)
        encodings = torch.cat([torch.sin(2 * np.pi * t_expanded), torch.cos(2 * np.pi * t_expanded)], dim=-1)
        return encodings
    
    def forward(self, x, y, t):
        t_encoded = self.encode_time(t)
        inputs = torch.cat([x.unsqueeze(-1) if x.dim() == 1 else x,
                            y.unsqueeze(-1) if y.dim() == 1 else y,
                            t_encoded], dim=-1)
        h = self.network(inputs)
        h = torch.relu(h.squeeze(-1))
        return h

def compute_gradient(output, input_tensor, create_graph=True):
    if output is None or input_tensor is None: return None
    grad = torch.autograd.grad(outputs=output, inputs=input_tensor,
                               grad_outputs=torch.ones_like(output),
                               create_graph=create_graph, retain_graph=True, allow_unused=True)[0]
    return grad

def physics_loss_v3(model, x, y, t, verbose=False):
    x = x.requires_grad_(True)
    y = y.requires_grad_(True)
    t = t.requires_grad_(True)
    h = model(x, y, t)
    
    h_t = compute_gradient(h, t)
    if h_t is None: return torch.tensor(0.0, device=device)
    
    negative_growth = torch.relu(-h_t)
    loss_monotonic = (negative_growth ** 2).mean()
    
    max_rate = 2.0
    excessive_growth = torch.relu(h_t - max_rate)
    loss_bounded = (excessive_growth ** 2).mean()
    
    h_x = compute_gradient(h, x)
    h_y = compute_gradient(h, y)
    loss_smooth = torch.tensor(0.0, device=device)
    if h_x is not None and h_y is not None:
        h_xx = compute_gradient(h_x, x)
        h_yy = compute_gradient(h_y, y)
        if h_xx is not None and h_yy is not None:
            laplacian = h_xx + h_yy
            loss_smooth = (laplacian ** 2).mean() * 0.01

    total = loss_monotonic + loss_bounded + loss_smooth
    return total

# ============================================
# NEW: SIMPLIFIED DATASET
# ============================================

class SimpleTargetDataset:
    def __init__(self, width=100, height=100):
        self.width = width
        self.height = height
        
        # Create a simple target: a soft-edged square
        image = np.zeros((height, width), dtype=np.float32)
        # Define square boundaries
        x_start, x_end = int(width * 0.3), int(width * 0.7)
        y_start, y_end = int(height * 0.3), int(height * 0.7)
        image[y_start:y_end, x_start:x_end] = 1.0
        
        # Apply Gaussian blur for soft edges (helps the network)
        self.image = gaussian_filter(image, sigma=3)
        
        print(f"✓ Created simple target dataset: {self.width}x{self.height}")
    
    def sample_points(self, n_points):
        x = np.random.rand(n_points)
        y = np.random.rand(n_points)
        t = np.random.rand(n_points)
        return x, y, t
    
    def get_final_image_points(self, n_points):
        x_idx = np.random.randint(0, self.width, n_points)
        y_idx = np.random.randint(0, self.height, n_points)
        x = x_idx / (self.width - 1)
        y = y_idx / (self.height - 1)
        h_true = self.image[y_idx, x_idx]
        return x, y, h_true
    
    def get_initial_condition_points(self, n_points):
        x = np.random.rand(n_points)
        y = np.random.rand(n_points)
        return x, y

# ============================================
# SIMPLIFIED TRAINING LOOP
# ============================================

def train_pinn_simple(model, dataset, epochs=5000, 
                      n_physics=1000, n_boundary=1000, n_initial=800, 
                      lr=1e-3, patience=500):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=patience, factor=0.5, verbose=True)
    
    # Simplified loss weights
    lambda_physics = 1.0       # Physics constraints
    lambda_boundary = 10.0     # Final image match
    lambda_initial = 10.0      # Blank at t=0 (critical!)
    lambda_monotonic = 5.0     # Enforce h(t2) ≥ h(t1)

    print("\n" + "=" * 60)
    print("TRAINING CONFIGURATION (SIMPLE TARGET)")
    print("=" * 60)
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # Physics Loss
        x_phys, y_phys, t_phys = dataset.sample_points(n_physics)
        x_phys = torch.tensor(x_phys, dtype=torch.float32, device=device)
        y_phys = torch.tensor(y_phys, dtype=torch.float32, device=device)
        t_phys = torch.tensor(t_phys, dtype=torch.float32, device=device)
        loss_phys = physics_loss_v3(model, x_phys, y_phys, t_phys)
        
        # Boundary Condition (t=1)
        x_bc, y_bc, h_true_bc = dataset.get_final_image_points(n_boundary)
        x_bc = torch.tensor(x_bc, dtype=torch.float32, device=device)
        y_bc = torch.tensor(y_bc, dtype=torch.float32, device=device)
        t_bc = torch.ones_like(x_bc, device=device)
        h_true_bc = torch.tensor(h_true_bc, dtype=torch.float32, device=device)
        h_pred_bc = model(x_bc, y_bc, t_bc)
        loss_bc = ((h_pred_bc - h_true_bc) ** 2).mean()
        
        # Initial Condition (t=0)
        x_ic, y_ic = dataset.get_initial_condition_points(n_initial)
        x_ic = torch.tensor(x_ic, dtype=torch.float32, device=device)
        y_ic = torch.tensor(y_ic, dtype=torch.float32, device=device)
        t_ic = torch.zeros_like(x_ic, device=device)
        h_pred_ic = model(x_ic, y_ic, t_ic)
        loss_ic = (h_pred_ic ** 2).mean()
        
        # Monotonicity Loss
        x_mono, y_mono, _ = dataset.sample_points(n_physics // 2)
        x_mono = torch.tensor(x_mono, dtype=torch.float32, device=device)
        y_mono = torch.tensor(y_mono, dtype=torch.float32, device=device)
        t1 = torch.rand(n_physics // 2, device=device)
        t2 = torch.clamp(t1 + torch.rand(n_physics // 2, device=device) * 0.1, 0, 1)
        h1 = model(x_mono, y_mono, t1)
        h2 = model(x_mono, y_mono, t2)
        violation = torch.relu(h1 - h2)
        loss_mono = (violation ** 2).mean()
        
        # Total Loss
        loss = (lambda_physics * loss_phys + 
                lambda_boundary * loss_bc + 
                lambda_initial * loss_ic +
                lambda_monotonic * loss_mono)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step(loss)
        
        if (epoch + 1) % 200 == 0:
            print(f"\nEpoch {epoch+1}/{epochs} | Total: {loss.item():.6f}")
            print(f"  Physics: {loss_phys.item():.6f} | Initial: {loss_ic.item():.6f} | "
                  f"Monotonic: {loss_mono.item():.6f} | Boundary: {loss_bc.item():.6f}")

# ============================================
# VISUALIZATION (Identical to V3)
# ============================================

def visualize_reconstruction_simple(model, dataset, output_path='reconstruction_simple_target.png'):
    model.eval()
    time_points = [0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.0]
    
    fig, axes = plt.subplots(2, len(time_points) + 1, figsize=(22, 5))
    
    # Add the ground truth final image for comparison
    axes[0, 0].imshow(1.0 - dataset.image, cmap='gray', vmin=0, vmax=1)
    axes[0, 0].set_title('Ground Truth (t=1.0)')
    axes[0, 0].axis('off')
    axes[1, 0].imshow(dataset.image, cmap='hot', vmin=0, vmax=1)
    axes[1, 0].axis('off')

    with torch.no_grad():
        for i, t_val in enumerate(time_points):
            x_grid = np.linspace(0, 1, dataset.width)
            y_grid = np.linspace(0, 1, dataset.height)
            X, Y = np.meshgrid(x_grid, y_grid)
            
            x_flat = torch.tensor(X.flatten(), dtype=torch.float32, device=device)
            y_flat = torch.tensor(Y.flatten(), dtype=torch.float32, device=device)
            t_flat = torch.ones_like(x_flat) * t_val
            
            h_pred = model(x_flat, y_flat, t_flat)
            h_pred = h_pred.cpu().numpy().reshape(dataset.height, dataset.width)
            
            ax_idx = i + 1
            axes[0, ax_idx].imshow(1.0 - np.clip(h_pred, 0, 1), cmap='gray', vmin=0, vmax=1)
            axes[0, ax_idx].set_title(f't={t_val:.2f}')
            axes[0, ax_idx].axis('off')
            
            axes[1, ax_idx].imshow(h_pred, cmap='hot', vmin=0, vmax=h_pred.max() if h_pred.max() > 0 else 1)
            axes[1, ax_idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"✓ Saved visualization: {output_path}")
    plt.close()

# ============================================
# MAIN EXECUTION
# ============================================

if __name__ == "__main__":
    print("\n[1] Creating simple dataset...")
    dataset = SimpleTargetDataset()
    
    print("\n[2] Creating V3 model...")
    model = QuillPINN_V3(
        hidden_dim=256, 
        num_layers=8,
        time_encoding_dim=32
    ).to(device)
    
    print("\n[3] Starting simplified training...")
    train_pinn_simple(model, dataset)
    
    print("\n[4] Generating visualizations...")
    visualize_reconstruction_simple(model, dataset)
    
    print("\n[5] Saving model...")
    torch.save(model.state_dict(), 'quill_pinn_model_simple_target.pt')
    
    print("\n" + "=" * 60)
    print("✓ SIMPLE TARGET TRAINING COMPLETE")
    print("\nCheck the file 'reconstruction_simple_target.png'.")
    print("It should show a square smoothly appearing over time.")
    print("=" * 60)
