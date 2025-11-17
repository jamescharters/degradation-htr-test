#!/usr/bin/env python3
"""
Evaluate PINN reconstruction accuracy
"""

import torch
import numpy as np
from PIL import Image
from train_quill_pinn import QuillPINN, QuillDataset

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Load trained model
checkpoint = torch.load('quill_pinn_model.pt')
model = QuillPINN(hidden_dim=128, num_layers=8).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Load dataset
dataset = QuillDataset(
    "synthetic_letters/CANONE_letter_0_C.png",
    "synthetic_letters/CANONE_letter_0_C_metadata.json"
)

# Evaluate final reconstruction accuracy
with torch.no_grad():
    x_grid = np.linspace(0, 1, dataset.width)
    y_grid = np.linspace(0, 1, dataset.height)
    X, Y = np.meshgrid(x_grid, y_grid)
    
    x_flat = torch.tensor(X.flatten(), dtype=torch.float32, device=device)
    y_flat = torch.tensor(Y.flatten(), dtype=torch.float32, device=device)
    t_flat = torch.ones_like(x_flat)  # t=1.0 (final)
    
    h_pred = model(x_flat, y_flat, t_flat)
    h_pred = h_pred.cpu().numpy().reshape(dataset.height, dataset.width)

# Ground truth
h_true = dataset.image

# Metrics
mse = np.mean((h_pred - h_true) ** 2)
mae = np.mean(np.abs(h_pred - h_true))
correlation = np.corrcoef(h_pred.flatten(), h_true.flatten())[0, 1]

print("=" * 60)
print("RECONSTRUCTION METRICS")
print("=" * 60)
print(f"MSE:         {mse:.6f}")
print(f"MAE:         {mae:.6f}")
print(f"Correlation: {correlation:.4f}")
print()

# Pixel-wise accuracy (within threshold)
threshold = 0.1
accurate_pixels = np.sum(np.abs(h_pred - h_true) < threshold)
accuracy = accurate_pixels / h_true.size * 100
print(f"Accuracy (±{threshold}): {accuracy:.2f}%")
print("=" * 60)
