#!/usr/bin/env python3
"""
Fast stroke order visualization using vectorized operations
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from train_quill_pinn import QuillPINN, QuillDataset

device = torch.device("mps")

# Load
model = QuillPINN().to(device)
model.load_state_dict(torch.load('quill_pinn_model.pt')['model_state_dict'])
model.eval()

dataset = QuillDataset(
    "synthetic_letters/CANONE_letter_0_C.png",
    "synthetic_letters/CANONE_letter_0_C_metadata.json"
)

print("Computing activation times (this may take 1-2 minutes)...")

# Sample fewer time points and use batching
time_samples = np.linspace(0, 1, 20)  # Reduced from 50
activation_time = np.zeros((dataset.height, dataset.width))

# Vectorized: process entire spatial grid at once for each time
x_grid = np.linspace(0, 1, dataset.width)
y_grid = np.linspace(0, 1, dataset.height)
X, Y = np.meshgrid(x_grid, y_grid)

x_flat = torch.tensor(X.flatten(), dtype=torch.float32, device=device)
y_flat = torch.tensor(Y.flatten(), dtype=torch.float32, device=device)

with torch.no_grad():
    for t_idx, t_val in enumerate(time_samples):
        print(f"  Time step {t_idx+1}/{len(time_samples)}: t={t_val:.2f}")
        
        t_flat = torch.ones_like(x_flat) * t_val
        
        # Predict all pixels at this time (vectorized!)
        h = model(x_flat, y_flat, t_flat)
        h = h.cpu().numpy().reshape(dataset.height, dataset.width)
        
        # Mark pixels that activated at this time
        mask = (activation_time == 0) & (h > 0.1)
        activation_time[mask] = t_val

# Plot
fig, ax = plt.subplots(figsize=(8, 8))
im = ax.imshow(activation_time, cmap='jet', vmin=0, vmax=1)
ax.set_title('Stroke Order Recovery\n(Color = Time of Writing)')
ax.axis('off')
plt.colorbar(im, ax=ax, label='Normalized Time')
plt.tight_layout()
plt.savefig('stroke_order_visualization_fast.png', dpi=150)
print("✓ Saved: stroke_order_visualization_fast.png")
plt.show()
