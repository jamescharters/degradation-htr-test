#!/usr/bin/env python3
"""
Test: Can PINN recover stroke parameters?
"""

import torch
import numpy as np
import json
import matplotlib.pyplot as plt
from train_quill_pinn import QuillPINN, QuillDataset

device = torch.device("mps")

# Load model
checkpoint = torch.load('quill_pinn_model.pt')
model = QuillPINN(hidden_dim=128, num_layers=8).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Load ground truth metadata
with open("synthetic_letters/CANONE_letter_0_C_metadata.json") as f:
    metadata = json.load(f)

print("=" * 60)
print("STROKE PARAMETER RECOVERY TEST")
print("=" * 60)

# For each stroke, analyze temporal activation
strokes = metadata['strokes']

for i, stroke in enumerate(strokes):
    start = stroke['start']
    end = stroke['end']
    true_start_time = stroke['start_time']
    true_end_time = stroke['end_time']
    true_pressure = stroke['pressure']
    true_speed = stroke['speed']
    
    print(f"\nStroke {i+1}:")
    print(f"  Ground truth:")
    print(f"    Time: {true_start_time:.3f}s - {true_end_time:.3f}s")
    print(f"    Pressure: {true_pressure:.2f}")
    print(f"    Speed: {true_speed:.2f}")
    
    # Sample points along stroke path
    n_samples = 20
    x_stroke = np.linspace(start[0]/256, end[0]/256, n_samples)
    y_stroke = np.linspace(start[1]/256, end[1]/256, n_samples)
    
    # Find when ink appears at these locations
    time_samples = np.linspace(0, 1, 100)
    
    with torch.no_grad():
        activations = []
        for t_val in time_samples:
            x_t = torch.tensor(x_stroke, dtype=torch.float32, device=device)
            y_t = torch.tensor(y_stroke, dtype=torch.float32, device=device)
            t_t = torch.ones_like(x_t) * t_val
            
            h = model(x_t, y_t, t_t)
            avg_h = h.mean().item()
            activations.append(avg_h)
    
    # Find activation time (when ink first appears)
    activations = np.array(activations)
    threshold = 0.1 * activations.max()
    activation_idx = np.where(activations > threshold)[0]
    
    if len(activation_idx) > 0:
        inferred_start = time_samples[activation_idx[0]]
        inferred_end = time_samples[activation_idx[-1]]
        
        print(f"  Inferred:")
        print(f"    Time: {inferred_start:.3f}s - {inferred_end:.3f}s")
        
        # Check if stroke order is correct
        if i > 0:
            prev_end = strokes[i-1]['end_time']
            if inferred_start > prev_end:
                print(f"    ✓ Stroke order correct (after stroke {i})")
            else:
                print(f"    ✗ Stroke order wrong")

print("\n" + "=" * 60)
