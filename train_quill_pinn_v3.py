#!/usr/bin/env python3
"""
Physics-Informed Neural Network for Quill Writing Reconstruction V3
Multiple strategies to force temporal learning
"""

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import json
import matplotlib.pyplot as plt
from pathlib import Path
import sys

print("=" * 60)
print("QUILL PHYSICS PINN TRAINING V3")
print("=" * 60)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"✓ Using device: {device}\n")

# ============================================
# IMPROVED PINN ARCHITECTURE WITH TIME ENCODING
# ============================================

class QuillPINN_V3(nn.Module):
    """
    Enhanced PINN with sinusoidal time encoding to force temporal awareness
    """
    def __init__(self, hidden_dim=256, num_layers=8, time_encoding_dim=32):
        super().__init__()
        
        self.time_encoding_dim = time_encoding_dim
        
        # Input: (x, y) [2] + time_encoding [time_encoding_dim] → hidden
        input_dim = 2 + time_encoding_dim
        
        # Network
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.Tanh())
        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
        
        layers.append(nn.Linear(hidden_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
        
        print(f"✓ Model created:")
        print(f"  - Time encoding: {time_encoding_dim} dimensions")
        print(f"  - Hidden dim: {hidden_dim}")
        print(f"  - Layers: {num_layers}")
        print(f"  - Total params: {sum(p.numel() for p in self.parameters()):,}")
    
    def encode_time(self, t):
        """
        Sinusoidal positional encoding for time
        Makes network sensitive to temporal variations
        """
        # Create frequency bands
        freqs = torch.arange(
            self.time_encoding_dim // 2, 
            device=t.device, 
            dtype=torch.float32
        )
        freqs = 2.0 ** freqs  # Exponential frequencies
        
        # Expand time with frequencies
        t_expanded = t.unsqueeze(-1) * freqs.unsqueeze(0)
        
        # Sine and cosine encodings
        encodings = torch.cat([
            torch.sin(2 * np.pi * t_expanded),
            torch.cos(2 * np.pi * t_expanded)
        ], dim=-1)
        
        return encodings
    
    def forward(self, x, y, t):
        """
        Forward pass with time encoding
        
        Args:
            x, y: spatial coordinates [0, 1]
            t: time [0, 1]
        Returns:
            h: ink height (non-negative)
        """
        # Encode time
        t_encoded = self.encode_time(t)
        
        # Concatenate spatial + temporal features
        inputs = torch.cat([
            x.unsqueeze(-1) if x.dim() == 1 else x,
            y.unsqueeze(-1) if y.dim() == 1 else y,
            t_encoded
        ], dim=-1)
        
        # Network prediction
        h = self.network(inputs)
        
        # Non-negative constraint (ink height ≥ 0)
        h = torch.relu(h.squeeze(-1))
        
        return h

# ============================================
# ROBUST GRADIENT COMPUTATION
# ============================================

def compute_gradient(output, input_tensor, create_graph=True):
    """
    Robust gradient computation with error checking
    """
    if output is None or input_tensor is None:
        return None
    
    grad = torch.autograd.grad(
        outputs=output,
        inputs=input_tensor,
        grad_outputs=torch.ones_like(output),
        create_graph=create_graph,
        retain_graph=True,
        allow_unused=True
    )[0]
    
    return grad

# ============================================
# MULTI-COMPONENT PHYSICS LOSS
# ============================================

def physics_loss_v3(model, x, y, t, verbose=False):
    """
    Comprehensive physics-based loss with multiple constraints
    """
    # Ensure gradient tracking
    x = x.requires_grad_(True)
    y = y.requires_grad_(True)
    t = t.requires_grad_(True)
    
    # Forward pass
    h = model(x, y, t)
    
    # === TEMPORAL DERIVATIVE ===
    h_t = compute_gradient(h, t)
    
    if h_t is None:
        if verbose:
            print("WARNING: h_t gradient is None!")
        return torch.tensor(0.0, device=device)
    
    # Check if meaningful
    if torch.all(torch.abs(h_t) < 1e-8):
        if verbose:
            print(f"WARNING: h_t is near zero! Range: [{h_t.min():.2e}, {h_t.max():.2e}]")
    
    # === CONSTRAINT 1: Monotonic Accumulation ===
    # Ink only accumulates: ∂h/∂t ≥ 0
    negative_growth = torch.relu(-h_t)
    loss_monotonic = (negative_growth ** 2).mean()
    
    # === CONSTRAINT 2: Bounded Growth Rate ===
    # Ink shouldn't appear instantly
    max_rate = 2.0  # Maximum reasonable growth rate
    excessive_growth = torch.relu(h_t - max_rate)
    loss_bounded = (excessive_growth ** 2).mean()
    
    # === CONSTRAINT 3: Spatial Smoothness ===
    # Ink spreads smoothly in space
    h_x = compute_gradient(h, x)
    h_y = compute_gradient(h, y)
    
    if h_x is not None and h_y is not None:
        h_xx = compute_gradient(h_x, x)
        h_yy = compute_gradient(h_y, y)
        
        if h_xx is not None and h_yy is not None:
            laplacian = h_xx + h_yy
            # Penalize sharp spatial gradients
            loss_smooth = (laplacian ** 2).mean() * 0.01
        else:
            loss_smooth = torch.tensor(0.0, device=device)
    else:
        loss_smooth = torch.tensor(0.0, device=device)
    
    # === CONSTRAINT 4: Temporal Smoothness ===
    # Second derivative in time should be small (gradual changes)
    h_tt = compute_gradient(h_t, t)
    if h_tt is not None:
        loss_temporal_smooth = (h_tt ** 2).mean() * 0.1
    else:
        loss_temporal_smooth = torch.tensor(0.0, device=device)
    
    # Total physics loss
    total = loss_monotonic + loss_bounded + loss_smooth + loss_temporal_smooth
    
    if verbose and total.item() > 0:
        print(f"  Physics components:")
        print(f"    Monotonic:   {loss_monotonic.item():.6f}")
        print(f"    Bounded:     {loss_bounded.item():.6f}")
        print(f"    Spatial:     {loss_smooth.item():.6f}")
        print(f"    Temporal:    {loss_temporal_smooth.item():.6f}")
    
    return total

# ============================================
# DATA LOADING (SAME AS V2)
# ============================================

class QuillDataset:
    def __init__(self, letter_path, metadata_path):
        self.image = np.array(Image.open(letter_path).convert('L'))
        self.image = self.image.astype(np.float32) / 255.0
        self.image = 1.0 - self.image
        
        self.height, self.width = self.image.shape
        
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        self.total_time = self.metadata['total_time']
        
        self.stroke_times = []
        for stroke in self.metadata['strokes']:
            start_t = stroke['start_time'] / self.total_time
            end_t = stroke['end_time'] / self.total_time
            self.stroke_times.append((start_t, end_t))
        
        print(f"✓ Loaded letter: {self.metadata['letter']}")
        print(f"  Image: {self.width}x{self.height}")
        print(f"  Strokes: {len(self.stroke_times)}")
    
    def sample_points(self, n_points):
        x_idx = np.random.randint(0, self.width, n_points)
        y_idx = np.random.randint(0, self.height, n_points)
        t = np.random.uniform(0, 1, n_points)
        
        x = x_idx / self.width
        y = y_idx / self.height
        
        return x, y, t
    
    def get_final_image_points(self, n_points):
        x_idx = np.random.randint(0, self.width, n_points)
        y_idx = np.random.randint(0, self.height, n_points)
        
        x = x_idx / self.width
        y = y_idx / self.height
        h_true = self.image[y_idx, x_idx]
        
        return x, y, h_true
    
    def get_initial_condition_points(self, n_points):
        x_idx = np.random.randint(0, self.width, n_points)
        y_idx = np.random.randint(0, self.height, n_points)
        
        x = x_idx / self.width
        y = y_idx / self.height
        
        return x, y
    
    def get_early_time_points(self, n_points):
        """Points before first stroke should be blank"""
        if len(self.stroke_times) == 0:
            return None
        
        x_idx = np.random.randint(0, self.width, n_points)
        y_idx = np.random.randint(0, self.height, n_points)
        
        x = x_idx / self.width
        y = y_idx / self.height
        
        # Sample times before first stroke
        t = np.random.uniform(0, self.stroke_times[0][0] * 0.9, n_points)
        h_true = np.zeros(n_points)
        
        return x, y, t, h_true
    
    def get_late_time_points(self, n_points):
        """Points after last stroke should match final image at inked regions"""
        if len(self.stroke_times) == 0:
            return None
        
        x_idx = np.random.randint(0, self.width, n_points)
        y_idx = np.random.randint(0, self.height, n_points)
        
        x = x_idx / self.width
        y = y_idx / self.height
        
        # Sample times after last stroke
        t = np.random.uniform(self.stroke_times[-1][1], 1.0, n_points)
        h_true = self.image[y_idx, x_idx]
        
        return x, y, t, h_true

# ============================================
# ENHANCED TRAINING WITH DIAGNOSTICS
# ============================================

def train_pinn_v3(model, dataset, epochs=10000, 
                  n_physics=1000, n_boundary=1000, n_initial=800, 
                  n_early=400, n_late=400,
                  lr=1e-3, patience=1000):
    """
    Training with comprehensive constraints and diagnostics
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=patience, factor=0.5, verbose=True
    )
    
    # Loss weights (tuned for temporal learning)
    lambda_physics = 50.0       # Physics constraints
    lambda_boundary = 20.0      # Final image match
    lambda_initial = 100.0      # Blank at t=0 (critical!)
    lambda_early = 50.0         # Blank before first stroke
    lambda_late = 30.0          # Match final after last stroke
    lambda_monotonic = 80.0     # Enforce h(t2) ≥ h(t1)
    
    loss_history = {
        'total': [], 'physics': [], 'boundary': [], 
        'initial': [], 'early': [], 'late': [], 'monotonic': []
    }
    
    print("\n" + "=" * 60)
    print("TRAINING CONFIGURATION")
    print("=" * 60)
    print(f"Epochs: {epochs}")
    print(f"Learning rate: {lr}")
    print(f"Loss weights:")
    print(f"  Physics:    {lambda_physics}")
    print(f"  Initial:    {lambda_initial} (blank at t=0)")
    print(f"  Monotonic:  {lambda_monotonic} (accumulation)")
    print(f"  Boundary:   {lambda_boundary} (final match)")
    print("=" * 60 + "\n")
    
    # Diagnostic: Check if model uses time
    def check_temporal_sensitivity():
        with torch.no_grad():
            x_test = torch.tensor([0.5], device=device)
            y_test = torch.tensor([0.5], device=device)
            
            t_values = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], device=device)
            h_values = []
            
            for t in t_values:
                t_tensor = torch.tensor([t.item()], device=device)
                h = model(x_test, y_test, t_tensor)
                h_values.append(h.item())
            
            variance = np.var(h_values)
            range_val = max(h_values) - min(h_values)
            
            return h_values, variance, range_val
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # === PHYSICS LOSS ===
        x_phys, y_phys, t_phys = dataset.sample_points(n_physics)
        x_phys = torch.tensor(x_phys, dtype=torch.float32, device=device)
        y_phys = torch.tensor(y_phys, dtype=torch.float32, device=device)
        t_phys = torch.tensor(t_phys, dtype=torch.float32, device=device)
        
        verbose_physics = (epoch % 500 == 0)
        loss_phys = physics_loss_v3(model, x_phys, y_phys, t_phys, verbose=verbose_physics)
        
        # === BOUNDARY CONDITION (t=1) ===
        x_bc, y_bc, h_true_bc = dataset.get_final_image_points(n_boundary)
        x_bc = torch.tensor(x_bc, dtype=torch.float32, device=device)
        y_bc = torch.tensor(y_bc, dtype=torch.float32, device=device)
        t_bc = torch.ones_like(x_bc, device=device)
        h_true_bc = torch.tensor(h_true_bc, dtype=torch.float32, device=device)
        
        h_pred_bc = model(x_bc, y_bc, t_bc)
        loss_bc = ((h_pred_bc - h_true_bc) ** 2).mean()
        
        # === INITIAL CONDITION (t=0, blank canvas) ===
        x_ic, y_ic = dataset.get_initial_condition_points(n_initial)
        x_ic = torch.tensor(x_ic, dtype=torch.float32, device=device)
        y_ic = torch.tensor(y_ic, dtype=torch.float32, device=device)
        t_ic = torch.zeros_like(x_ic, device=device)
        
        h_pred_ic = model(x_ic, y_ic, t_ic)
        loss_ic = (h_pred_ic ** 2).mean()
        
        # === EARLY TIME SUPERVISION ===
        early_data = dataset.get_early_time_points(n_early)
        if early_data is not None:
            x_early, y_early, t_early, h_early = early_data
            x_early = torch.tensor(x_early, dtype=torch.float32, device=device)
            y_early = torch.tensor(y_early, dtype=torch.float32, device=device)
            t_early = torch.tensor(t_early, dtype=torch.float32, device=device)
            h_early = torch.tensor(h_early, dtype=torch.float32, device=device)
            
            h_pred_early = model(x_early, y_early, t_early)
            loss_early = ((h_pred_early - h_early) ** 2).mean()
        else:
            loss_early = torch.tensor(0.0, device=device)
        
        # === LATE TIME SUPERVISION ===
        late_data = dataset.get_late_time_points(n_late)
        if late_data is not None:
            x_late, y_late, t_late, h_late = late_data
            x_late = torch.tensor(x_late, dtype=torch.float32, device=device)
            y_late = torch.tensor(y_late, dtype=torch.float32, device=device)
            t_late = torch.tensor(t_late, dtype=torch.float32, device=device)
            h_late = torch.tensor(h_late, dtype=torch.float32, device=device)
            
            h_pred_late = model(x_late, y_late, t_late)
            loss_late = ((h_pred_late - h_late) ** 2).mean()
        else:
            loss_late = torch.tensor(0.0, device=device)
        
        # === MONOTONICITY (h(t2) ≥ h(t1) for t2 > t1) ===
        x_mono, y_mono, _ = dataset.sample_points(n_physics // 2)
        x_mono = torch.tensor(x_mono, dtype=torch.float32, device=device)
        y_mono = torch.tensor(y_mono, dtype=torch.float32, device=device)
        
        t1 = torch.rand(n_physics // 2, device=device)
        t2 = t1 + torch.rand(n_physics // 2, device=device) * 0.1
        t2 = torch.clamp(t2, 0, 1)
        
        h1 = model(x_mono, y_mono, t1)
        h2 = model(x_mono, y_mono, t2)
        
        violation = torch.relu(h1 - h2)
        loss_mono = (violation ** 2).mean()
        
        # === TOTAL LOSS ===
        loss = (lambda_physics * loss_phys + 
                lambda_boundary * loss_bc + 
                lambda_initial * loss_ic +
                lambda_early * loss_early +
                lambda_late * loss_late +
                lambda_monotonic * loss_mono)
        
        # Backward
        loss.backward()
        
        # Gradient clipping (stability)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step(loss)
        
        # Record
        loss_history['total'].append(loss.item())
        loss_history['physics'].append(loss_phys.item())
        loss_history['boundary'].append(loss_bc.item())
        loss_history['initial'].append(loss_ic.item())
        loss_history['early'].append(loss_early.item())
        loss_history['late'].append(loss_late.item())
        loss_history['monotonic'].append(loss_mono.item())
        
        # Progress
        if (epoch + 1) % 200 == 0:
            print(f"\nEpoch {epoch+1}/{epochs}")
            print(f"  Total:     {loss.item():.6f}")
            print(f"  Physics:   {loss_phys.item():.6f}")
            print(f"  Initial:   {loss_ic.item():.6f}")
            print(f"  Monotonic: {loss_mono.item():.6f}")
            print(f"  Boundary:  {loss_bc.item():.6f}")
            print(f"  Early:     {loss_early.item():.6f}")
            print(f"  Late:      {loss_late.item():.6f}")
            print(f"  LR:        {optimizer.param_groups[0]['lr']:.2e}")
            
            # Diagnostic
            h_vals, var, rng = check_temporal_sensitivity()
            print(f"  Temporal check: variance={var:.6f}, range={rng:.6f}")
            if var < 1e-4:
                print(f"  ⚠️  WARNING: Model not using time! h values: {h_vals}")
            else:
                print(f"  ✓ Model is temporally sensitive")
        
        # Early stopping check
        if (epoch + 1) % 500 == 0:
            if loss_phys.item() < 1e-6:
                print("\n⚠️  WARNING: Physics loss near zero - may indicate gradient issues")
    
    return loss_history

# ============================================
# VISUALIZATION
# ============================================

def visualize_reconstruction_v3(model, dataset, output_path='reconstruction_v3.png'):
    model.eval()
    time_points = [0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.0]
    
    fig, axes = plt.subplots(2, len(time_points), figsize=(20, 5))
    
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
            
            h_display = 1.0 - np.clip(h_pred, 0, 1)
            
            axes[0, i].imshow(h_display, cmap='gray', vmin=0, vmax=1)
            axes[0, i].set_title(f't={t_val:.2f}', fontsize=10)
            axes[0, i].axis('off')
            
            axes[1, i].imshow(h_pred, cmap='hot', vmin=0, vmax=h_pred.max() if h_pred.max() > 0 else 1)
            axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()

def plot_loss_curves_v3(loss_history, output_path='loss_curves_v3.png'):
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    for idx, (key, ax) in enumerate(zip(
        ['total', 'physics', 'initial', 'monotonic', 'boundary', 'early', 'late'],
        axes.flatten()
    )):
        if key in loss_history and len(loss_history[key]) > 0:
            ax.plot(loss_history[key])
            ax.set_title(key.capitalize())
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('Epoch')
    
    # Hide unused subplot
    axes.flatten()[-1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"✓ Saved: {output_path}")
    plt.close()

# ============================================
# MAIN EXECUTION
# ============================================

if __name__ == "__main__":
    if not Path("synthetic_letters").exists():
        print("✗ Run quill_simulator.py first!")
        sys.exit(1)
    
    print("\n[1] Loading data...")
    dataset = QuillDataset(
        "synthetic_letters/CANONE_letter_0_C.png",
        "synthetic_letters/CANONE_letter_0_C_metadata.json"
    )
    
    print("\n[2] Creating V3 model with time encoding...")
    model = QuillPINN_V3(
        hidden_dim=256, 
        num_layers=8,
        time_encoding_dim=32
    ).to(device)
    
    print("\n[3] Starting training...")
    print("This will take 30-45 minutes on M4...")
    
    loss_history = train_pinn_v3(
        model, 
        dataset,
        epochs=10000,
        n_physics=1000,
        n_boundary=1000,
        n_initial=800,
        n_early=400,
        n_late=400,
        lr=1e-3,
        patience=1000
    )
    
    print("\n[4] Generating visualizations...")
    visualize_reconstruction_v3(model, dataset)
    plot_loss_curves_v3(loss_history)
    
    print("\n[5] Saving model...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'loss_history': loss_history,
        'config': {
            'hidden_dim': 256,
            'num_layers': 8,
            'time_encoding_dim': 32
        }
    }, 'quill_pinn_model_v3.pt')
    
    print("\n" + "=" * 60)
    print("✓ TRAINING COMPLETE (V3)")
    print("\nCheck these files:")
    print("  - reconstruction_v3.png (should show temporal progression!)")
    print("  - loss_curves_v3.png (physics should be non-zero)")
    print("  - quill_pinn_model_v3.pt (trained model)")
    print("=" * 60)
