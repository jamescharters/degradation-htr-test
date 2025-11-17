#!/usr/bin/env python3
"""
Physics-Informed Neural Network for Quill Writing Reconstruction
REVISED: Strong temporal constraints to learn writing process
"""

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import json
import matplotlib.pyplot as plt
from pathlib import Path

print("=" * 60)
print("QUILL PHYSICS PINN TRAINING (REVISED)")
print("=" * 60)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"✓ Using device: {device}\n")

# ============================================
# PINN ARCHITECTURE (Same)
# ============================================

class QuillPINN(nn.Module):
    def __init__(self, hidden_dim=256, num_layers=8):
        super().__init__()
        
        layers = [nn.Linear(3, hidden_dim), nn.Tanh()]
        
        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh()
            ])
        
        layers.append(nn.Linear(hidden_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x, y, t):
        inputs = torch.stack([x, y, t], dim=-1)
        h = self.network(inputs)
        # Apply sigmoid to constrain output to [0, 1]
        return torch.sigmoid(h.squeeze(-1))

# ============================================
# PHYSICS EQUATIONS (Improved)
# ============================================

def compute_gradients(output, input_tensor, order=1):
    grad = output
    for _ in range(order):
        grad = torch.autograd.grad(
            grad.sum(), 
            input_tensor,
            create_graph=True,
            retain_graph=True,
            allow_unused=True
        )[0]
        if grad is None:
            return torch.zeros_like(input_tensor)
    return grad

def physics_loss(model, x, y, t):
    """
    Enforces: ∂h/∂t ≥ 0 (ink only accumulates, never disappears)
    And: ∂h/∂t should be smooth (gradual changes)
    """
    x = x.requires_grad_(True)
    y = y.requires_grad_(True)
    t = t.requires_grad_(True)
    
    h = model(x, y, t)
    h_t = compute_gradients(h, t, order=1)
    
    # Physical constraint 1: Ink accumulates (∂h/∂t ≥ 0)
    violation = torch.relu(-h_t)  # Penalize negative time derivative
    accumulation_loss = (violation ** 2).mean()
    
    # Physical constraint 2: Smooth temporal evolution
    h_tt = compute_gradients(h_t, t, order=1)
    smoothness_loss = (h_tt ** 2).mean()
    
    # Physical constraint 3: Spatial diffusion (ink spreads)
    h_x = compute_gradients(h, x, order=1)
    h_y = compute_gradients(h, y, order=1)
    h_xx = compute_gradients(h_x, x, order=1)
    h_yy = compute_gradients(h_y, y, order=1)
    
    laplacian = h_xx + h_yy
    diffusion_term = 0.01 * laplacian
    
    # PDE: ∂h/∂t = diffusion
    pde_residual = h_t - diffusion_term
    pde_loss = (pde_residual ** 2).mean()
    
    return accumulation_loss + 0.1 * smoothness_loss + 0.01 * pde_loss

# ============================================
# IMPROVED DATA LOADING
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
        
        # Extract stroke timings for temporal supervision
        self.stroke_times = []
        for stroke in self.metadata['strokes']:
            start_t = stroke['start_time'] / self.total_time
            end_t = stroke['end_time'] / self.total_time
            self.stroke_times.append((start_t, end_t))
        
        print(f"✓ Loaded letter: {self.metadata['letter']}")
        print(f"  Strokes: {len(self.stroke_times)}")
        print(f"  Stroke timings: {self.stroke_times}")
    
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
    
    def get_intermediate_time_points(self, n_points):
        """
        Sample points at intermediate times where we know ink status
        Before first stroke: no ink
        After last stroke: full ink
        """
        x_idx = np.random.randint(0, self.width, n_points)
        y_idx = np.random.randint(0, self.height, n_points)
        
        x = x_idx / self.width
        y = y_idx / self.height
        
        # Sample time before first stroke (should be blank)
        if len(self.stroke_times) > 0:
            t_before = np.random.uniform(0, self.stroke_times[0][0], n_points // 2)
            h_before = np.zeros(n_points // 2)
            
            # Sample time after all strokes (should match final for inked regions)
            t_after = np.random.uniform(self.stroke_times[-1][1], 1.0, n_points // 2)
            h_after = self.image[y_idx[n_points//2:], x_idx[n_points//2:]]
            
            t = np.concatenate([t_before, t_after])
            h_true = np.concatenate([h_before, h_after])
            
            return x, y, t, h_true
        
        return None

# ============================================
# TRAINING LOOP (REVISED)
# ============================================

def train_pinn(model, dataset, epochs=10000, 
               n_physics=1000, n_boundary=1000, n_initial=500, n_intermediate=500,
               lr=1e-3):
    """
    Train PINN with much stronger temporal constraints
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=500, factor=0.5)
    
    # REVISED: Much stronger physics weight
    lambda_physics = 100.0      # Was 1.0
    lambda_boundary = 10.0      # Final image match
    lambda_initial = 50.0       # Was 5.0 - enforce blank start
    lambda_intermediate = 20.0  # NEW - intermediate supervision
    lambda_monotonic = 50.0     # NEW - enforce monotonic increase
    
    loss_history = {
        'total': [],
        'physics': [],
        'boundary': [],
        'initial': [],
        'intermediate': [],
        'monotonic': []
    }
    
    print("\n" + "=" * 60)
    print("TRAINING WITH STRONG TEMPORAL CONSTRAINTS")
    print("=" * 60)
    print(f"Physics weight:      {lambda_physics}")
    print(f"Initial cond weight: {lambda_initial}")
    print(f"Monotonic weight:    {lambda_monotonic}")
    print("=" * 60 + "\n")
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # Physics Loss
        x_phys, y_phys, t_phys = dataset.sample_points(n_physics)
        x_phys = torch.tensor(x_phys, dtype=torch.float32, device=device)
        y_phys = torch.tensor(y_phys, dtype=torch.float32, device=device)
        t_phys = torch.tensor(t_phys, dtype=torch.float32, device=device)
        
        loss_phys = physics_loss(model, x_phys, y_phys, t_phys)
        
        # Boundary Condition (final image)
        x_bc, y_bc, h_true = dataset.get_final_image_points(n_boundary)
        x_bc = torch.tensor(x_bc, dtype=torch.float32, device=device)
        y_bc = torch.tensor(y_bc, dtype=torch.float32, device=device)
        t_bc = torch.ones_like(x_bc, device=device)
        h_true = torch.tensor(h_true, dtype=torch.float32, device=device)
        
        h_pred = model(x_bc, y_bc, t_bc)
        loss_bc = ((h_pred - h_true) ** 2).mean()
        
        # Initial Condition (blank at t=0)
        x_ic, y_ic = dataset.get_initial_condition_points(n_initial)
        x_ic = torch.tensor(x_ic, dtype=torch.float32, device=device)
        y_ic = torch.tensor(y_ic, dtype=torch.float32, device=device)
        t_ic = torch.zeros_like(x_ic, device=device)
        
        h_pred_ic = model(x_ic, y_ic, t_ic)
        loss_ic = (h_pred_ic ** 2).mean()
        
        # NEW: Intermediate time supervision
        loss_inter = torch.tensor(0.0, device=device)
        intermediate_data = dataset.get_intermediate_time_points(n_intermediate)
        if intermediate_data is not None:
            x_int, y_int, t_int, h_int = intermediate_data
            x_int = torch.tensor(x_int, dtype=torch.float32, device=device)
            y_int = torch.tensor(y_int, dtype=torch.float32, device=device)
            t_int = torch.tensor(t_int, dtype=torch.float32, device=device)
            h_int = torch.tensor(h_int, dtype=torch.float32, device=device)
            
            h_pred_int = model(x_int, y_int, t_int)
            loss_inter = ((h_pred_int - h_int) ** 2).mean()
        
        # NEW: Monotonicity constraint (h(t2) ≥ h(t1) for t2 > t1)
        x_mono, y_mono, _ = dataset.sample_points(n_physics)
        x_mono = torch.tensor(x_mono, dtype=torch.float32, device=device)
        y_mono = torch.tensor(y_mono, dtype=torch.float32, device=device)
        
        t1 = torch.rand(n_physics, device=device)
        t2 = t1 + torch.rand(n_physics, device=device) * 0.1  # t2 slightly > t1
        t2 = torch.clamp(t2, 0, 1)
        
        h1 = model(x_mono, y_mono, t1)
        h2 = model(x_mono, y_mono, t2)
        
        # h2 should be >= h1 (ink accumulates)
        monotonic_violation = torch.relu(h1 - h2)
        loss_mono = (monotonic_violation ** 2).mean()
        
        # Total Loss
        loss = (lambda_physics * loss_phys + 
                lambda_boundary * loss_bc + 
                lambda_initial * loss_ic +
                lambda_intermediate * loss_inter +
                lambda_monotonic * loss_mono)
        
        loss.backward()
        optimizer.step()
        scheduler.step(loss)
        
        # Record
        loss_history['total'].append(loss.item())
        loss_history['physics'].append(loss_phys.item())
        loss_history['boundary'].append(loss_bc.item())
        loss_history['initial'].append(loss_ic.item())
        loss_history['intermediate'].append(loss_inter.item())
        loss_history['monotonic'].append(loss_mono.item())
        
        if (epoch + 1) % 200 == 0:
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  Total:        {loss.item():.6f}")
            print(f"  Physics:      {loss_phys.item():.6f}")
            print(f"  Boundary:     {loss_bc.item():.6f}")
            print(f"  Initial:      {loss_ic.item():.6f}")
            print(f"  Intermediate: {loss_inter.item():.6f}")
            print(f"  Monotonic:    {loss_mono.item():.6f}")
            print(f"  LR:           {optimizer.param_groups[0]['lr']:.2e}")
    
    return loss_history

# ============================================
# VISUALIZATION (Same but fixed)
# ============================================

def visualize_reconstruction(model, dataset, output_path='reconstruction_v2.png'):
    model.eval()
    time_points = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    
    fig, axes = plt.subplots(2, len(time_points), figsize=(18, 6))
    
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
            
            h_display = 1.0 - h_pred
            h_display = np.clip(h_display, 0, 1)
            
            axes[0, i].imshow(h_display, cmap='gray', vmin=0, vmax=1)
            axes[0, i].set_title(f't={t_val:.1f}')
            axes[0, i].axis('off')
            
            axes[1, i].imshow(h_pred, cmap='hot', vmin=0, vmax=1)
            axes[1, i].set_title(f'Ink height')
            axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()

def plot_loss_curves(loss_history, output_path='loss_curves_v2.png'):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    axes[0, 0].plot(loss_history['total'])
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].set_yscale('log')
    axes[0, 0].grid(True)
    
    axes[0, 1].plot(loss_history['physics'])
    axes[0, 1].set_title('Physics Loss')
    axes[0, 1].set_yscale('log')
    axes[0, 1].grid(True)
    
    axes[0, 2].plot(loss_history['boundary'])
    axes[0, 2].set_title('Boundary Loss')
    axes[0, 2].set_yscale('log')
    axes[0, 2].grid(True)
    
    axes[1, 0].plot(loss_history['initial'])
    axes[1, 0].set_title('Initial Condition')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True)
    
    axes[1, 1].plot(loss_history['intermediate'])
    axes[1, 1].set_title('Intermediate Time')
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True)
    
    axes[1, 2].plot(loss_history['monotonic'])
    axes[1, 2].set_title('Monotonicity')
    axes[1, 2].set_yscale('log')
    axes[1, 2].grid(True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"✓ Saved: {output_path}")
    plt.close()

# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    import sys
    
    if not Path("synthetic_letters").exists():
        print("✗ Run quill_simulator.py first!")
        sys.exit(1)
    
    print("\n[1] Loading data...")
    dataset = QuillDataset(
        "synthetic_letters/CANONE_letter_0_C.png",
        "synthetic_letters/CANONE_letter_0_C_metadata.json"
    )
    
    print("\n[2] Creating model...")
    model = QuillPINN(hidden_dim=256, num_layers=8).to(device)
    print(f"✓ Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    print("\n[3] Training...")
    loss_history = train_pinn(
        model, 
        dataset,
        epochs=10000,  # More epochs
        n_physics=1000,
        n_boundary=1000,
        n_initial=500,
        n_intermediate=500,
        lr=1e-3
    )
    
    print("\n[4] Visualizing...")
    visualize_reconstruction(model, dataset, 'pinn_reconstruction_v2.png')
    plot_loss_curves(loss_history, 'pinn_loss_curves_v2.png')
    
    print("\n[5] Saving...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'loss_history': loss_history
    }, 'quill_pinn_model_v2.pt')
    
    print("\n" + "=" * 60)
    print("✓ TRAINING COMPLETE (V2)")
    print("=" * 60)
