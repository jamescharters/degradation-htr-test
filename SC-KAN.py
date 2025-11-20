import torch
import matplotlib.pyplot as plt
import numpy as np

# --- 1. The Setup ---
x = torch.linspace(0, 1, 200)

# Task A: The left side should be 0
mask_A = (x < 0.5)
y_A_target = torch.zeros_like(x)

# Task B: The right side should have a bump
mask_B = (x >= 0.5)
y_B_target = torch.zeros_like(x)
y_B_target[mask_B] = torch.exp(-100 * (x[mask_B] - 0.75)**2) # A bump at 0.75

# --- 2. The Models ---

# MODEL 1: The "Standard" approach (Global Polynomials / MLP style)
def global_model(x, weights):
    # Uses x, x^2, x^3... altering one weight affects the WHOLE curve.
    basis = torch.stack([x**i for i in range(len(weights))], dim=1)
    return basis @ weights

# MODEL 2: The "KAN" approach (Local Basis Functions)
def local_kan_model(x, weights, centers, width=0.03):
    # Radial Basis Functions (similar behavior to B-Splines)
    # Unsqueeze for broadcasting: [200, 1] vs [10] -> [200, 10]
    basis = torch.exp(-(x.unsqueeze(1) - centers.unsqueeze(0))**2 / (2 * width**2))
    return basis @ weights

# Parameters
num_params = 10
epochs = 500
w_global = torch.zeros(num_params, requires_grad=True)
w_local  = torch.zeros(num_params, requires_grad=True)
centers  = torch.linspace(0, 1, num_params) # Evenly spaced control points

optimizer_g = torch.optim.Adam([w_global], lr=0.1)
optimizer_l = torch.optim.Adam([w_local],  lr=0.1)

# --- 3. The Experiment with Logging ---

print("--- PHASE 1: Training on Task A (Left Side = 0) ---")

for i in range(epochs):
    # Global Train
    optimizer_g.zero_grad()
    pred_g = global_model(x[mask_A], w_global)
    loss_g = torch.mean((pred_g - y_A_target[mask_A])**2)
    loss_g.backward()
    optimizer_g.step()
    
    # Local Train
    optimizer_l.zero_grad()
    pred_l = local_kan_model(x[mask_A], w_local, centers)
    loss_l = torch.mean((pred_l - y_A_target[mask_A])**2)
    loss_l.backward()
    optimizer_l.step()
    
    if i % (epochs//5) == 0:
        print(f"Epoch {i:3d} | Global Loss: {loss_g.item():.6f} | Local KAN Loss: {loss_l.item():.6f}")

# --- DIAGNOSTIC: Snapshot weights after Phase 1 ---
w_global_after_A = w_global.clone().detach()
w_local_after_A  = w_local.clone().detach()
print("\nPhase 1 complete. Weights have been snapshotted.")


print("\n--- PHASE 2: Training on Task B (Right Side = Spike) ---")
print("Crucial: We are NO LONGER showing the models data from Task A.")

for i in range(epochs):
    # Global Train
    optimizer_g.zero_grad()
    pred_g = global_model(x[mask_B], w_global)
    loss_g = torch.mean((pred_g - y_B_target[mask_B])**2)
    loss_g.backward()
    optimizer_g.step()

    # Local Train
    optimizer_l.zero_grad()
    pred_l = local_kan_model(x[mask_B], w_local, centers)
    loss_l = torch.mean((pred_l - y_B_target[mask_B])**2)
    loss_l.backward()
    optimizer_l.step()
    
    if i % (epochs//5) == 0:
        print(f"Epoch {i:3d} | Global Loss: {loss_g.item():.6f} | Local KAN Loss: {loss_l.item():.6f}")

print("\nPhase 2 complete.")


# --- 4. Final Analysis & Quantitative Report ---
print("\n--- FINAL DIAGNOSTIC REPORT ---")

with torch.no_grad():
    # --- Metric 1: The Forgetting Score ---
    # How badly did the models mess up Task A after learning Task B? (Lower is better)
    final_pred_g_A = global_model(x[mask_A], w_global)
    forget_score_g = torch.mean((final_pred_g_A - y_A_target[mask_A])**2)

    final_pred_l_A = local_kan_model(x[mask_A], w_local, centers)
    forget_score_l = torch.mean((final_pred_l_A - y_A_target[mask_A])**2)

    print("\n1. Forgetting Metric (MSE on Task A after learning Task B):")
    print(f"   - Standard Model: {forget_score_g.item():.6f}")
    print(f"   - Local KAN Model:  {forget_score_l.item():.6f} (<< Should be near zero)")

    # --- Metric 2: Weight Change Analysis ---
    # How much did the weights change during Phase 2?
    delta_w_global = torch.norm(w_global - w_global_after_A)
    
    # For KAN, we analyze left and right weights separately
    left_mask = centers < 0.5
    right_mask = centers >= 0.5
    
    delta_w_local_left = torch.norm(w_local[left_mask] - w_local_after_A[left_mask])
    delta_w_local_right = torch.norm(w_local[right_mask] - w_local_after_A[right_mask])

    print("\n2. Weight Change (L2 Norm) during Phase 2:")
    print(f"   - Standard Model (All Weights): {delta_w_global.item():.4f}")
    print(f"   - Local KAN Model (Left Weights):  {delta_w_local_left.item():.4f} (<< Should be zero)")
    print(f"   - Local KAN Model (Right Weights): {delta_w_local_right.item():.4f} (>> Should be large)")

    # --- 5. Visualization ---
    y_pred_global = global_model(x, w_global)
    y_pred_local  = local_kan_model(x, w_local, centers)

    plt.figure(figsize=(12, 6))

    # Plot Global (Failure)
    plt.subplot(1, 2, 1)
    title_g = f"Standard Model (Global)\nCatastrophic Forgetting!\nTask A MSE: {forget_score_g.item():.4f}"
    plt.title(title_g)
    plt.plot(x.numpy(), y_pred_global.detach().numpy(), 'r-', label="Prediction")
    plt.plot(x[mask_A].numpy(), y_A_target[mask_A].numpy(), 'k--', label="Task A (Forgotten)", alpha=0.5)
    plt.plot(x[mask_B].numpy(), y_B_target[mask_B].numpy(), 'b--', label="Task B (Current)", alpha=0.5)
    plt.axvline(0.5, color='gray', linestyle=':')
    plt.legend()
    plt.ylim(-1.5, 1.5)

    # Plot Local (Success)
    plt.subplot(1, 2, 2)
    title_l = f"KAN Model (Local)\nMemory Preserved\nTask A MSE: {forget_score_l.item():.4f}"
    plt.title(title_l)
    plt.plot(x.numpy(), y_pred_local.detach().numpy(), 'g-', label="Prediction")
    plt.plot(x[mask_A].numpy(), y_A_target[mask_A].numpy(), 'k--', label="Task A (Safe)", alpha=0.5)
    plt.plot(x[mask_B].numpy(), y_B_target[mask_B].numpy(), 'b--', label="Task B (Learned)", alpha=0.5)
    plt.scatter(centers.numpy(), w_local.detach().numpy(), s=20, color='green', marker='x', label="Control Points")
    plt.axvline(0.5, color='gray', linestyle=':')
    plt.legend()
    # Auto-scaling for Y-axis on this plot is fine
    
    plt.tight_layout()
    plt.show()