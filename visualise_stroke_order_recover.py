#!/usr/bin/env python3
"""
Visualize: Which parts of letter appear at what time?
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

# Create time-coded visualization
fig, ax = plt.subplots(figsize=(8, 8))

with torch.no_grad():
    # For each pixel, find when it first gets ink
    x_grid = np.linspace(0, 1, dataset.width)
    y_grid = np.linspace(0, 1, dataset.height)
    X, Y = np.meshgrid(x_grid, y_grid)
    
    activation_time = np.zeros((dataset.height, dataset.width))
    
    time_samples = np.linspace(0, 1, 50)
    
    for y_idx in range(dataset.height):
        for x_idx in range(dataset.width):
            x = X[y_idx, x_idx]
            y = Y[y_idx, x_idx]
            
            # Sample over time
            for t_val in time_samples:
                x_t = torch.tensor([x], dtype=torch.float32, device=device)
                y_t = torch.tensor([y], dtype=torch.float32, device=device)
                t_t = torch.tensor([t_val], dtype=torch.float32, device=device)
                
                h = model(x_t, y_t, t_t).item()
                
                if h > 0.1:  # Ink appeared
                    activation_time[y_idx, x_idx] = t_val
                    break

# Plot: color = time of activation
im = ax.imshow(activation_time, cmap='jet', vmin=0, vmax=1)
ax.set_title('Stroke Order Recovery\n(Color = Time of Writing)')
ax.axis('off')
plt.colorbar(im, ax=ax, label='Normalized Time')

plt.tight_layout()
plt.savefig('stroke_order_visualization.png', dpi=150)
print("✓ Saved: stroke_order_visualization.png")
