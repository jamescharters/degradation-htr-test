import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Define the Problem ---
t_begin, t_end = 0.0, 10.0
y1_0, y2_0 = 1.0, 0.0

# --- 2. Define the Ground Truth ---
t_eval = np.linspace(t_begin, t_end, 500)
y1_true = np.cos(t_eval)

# --- 3. The PINN with a Clean, Corrected Structure ---
class FinalPINN(nn.Module):
    def __init__(self, n_features=128, scale=5.0):
        super(FinalPINN, self).__init__()
        # Fourier feature mapping matrix B
        self.B = nn.Parameter(torch.randn(1, n_features) * scale, requires_grad=False)
        self.mlp = nn.Sequential(
            nn.Linear(2 * n_features, 64), nn.SiLU(),
            nn.Linear(64, 64), nn.SiLU(),
            nn.Linear(64, 2) # Output for y₁ and y₂
        )

    def get_features(self, t):
        """Computes the Fourier features for a given time tensor."""
        t_proj = t @ self.B
        return torch.cat([torch.sin(t_proj), torch.cos(t_proj)], dim=-1)

    def forward(self, t):
        """
        Computes the network's output y(t).
        This forward pass has a simple, clean computational graph.
        """
        features = self.get_features(t)
        return self.mlp(features)

    def compute_derivatives(self, t):
        """
        Computes the analytical derivatives dy/dt using the chain rule.
        This is a more complex operation, kept separate to avoid graph issues.
        """
        t.requires_grad_(True) # Ensure t requires grad for this operation
        features = self.get_features(t)
        features.requires_grad_(True)

        y_pred = self.mlp(features)
        
        # d(MLP)/d(features)
        d_y_d_features = torch.autograd.grad(
            outputs=y_pred,
            inputs=features,
            grad_outputs=torch.ones_like(y_pred),
            create_graph=True
        )[0]
        
        # d(features)/dt
        t_proj = t @ self.B
        d_features_d_t = torch.cat(
            [torch.cos(t_proj) * self.B, -torch.sin(t_proj) * self.B], dim=-1
        )
        
        # Chain rule: dy/dt = d(MLP)/d(features) * d(features)/dt
        dy_dt = (d_y_d_features * d_features_d_t).sum(dim=1, keepdim=True)
        
        # We need the derivatives for y₁ and y₂ separately.
        dy1_dt = torch.autograd.grad(y_pred[:, 0].sum(), features, create_graph=True)[0]
        dy1_dt_analytic = (dy1_dt * d_features_d_t).sum(dim=1, keepdim=True)

        dy2_dt = torch.autograd.grad(y_pred[:, 1].sum(), features, create_graph=True)[0]
        dy2_dt_analytic = (dy2_dt * d_features_d_t).sum(dim=1, keepdim=True)

        return dy1_dt_analytic, dy2_dt_analytic

# --- 4. The Final Training Loop ---
pinn = FinalPINN()
optimizer = torch.optim.Adam(pinn.parameters(), lr=1e-3)
epochs = 10000

t_initial = torch.tensor([[t_begin]], dtype=torch.float32)
t_physics = torch.linspace(t_begin, t_end, 200).view(-1, 1)

print("--- Starting training with Corrected Analytical Derivatives ---")
for epoch in range(epochs):
    optimizer.zero_grad()
    
    # --- Loss 1: Initial Conditions ---
    y_init_pred = pinn(t_initial)
    y1_init_pred, y2_init_pred = y_init_pred.unbind(1)
    loss_data = torch.mean((y1_init_pred - y1_0)**2) + torch.mean((y2_init_pred - y2_0)**2)

    # --- Loss 2: Physics (using the dedicated derivative method) ---
    y_pred = pinn(t_physics)
    y1_pred, y2_pred = y_pred.unbind(1)
    dy1_dt_analytic, dy2_dt_analytic = pinn.compute_derivatives(t_physics)
    
    residual_1 = dy1_dt_analytic - y2_pred.view(-1, 1)
    residual_2 = dy2_dt_analytic - (-y1_pred.view(-1, 1))
    loss_physics = torch.mean(residual_1**2) + torch.mean(residual_2**2)

    total_loss = 10 * loss_data + loss_physics
    total_loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 1000 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss.item():.8f}')

print("Training finished.")

# --- 5. Visualization ---
pinn.eval()
with torch.no_grad():
    t_plot_tensor = torch.tensor(t_eval, dtype=torch.float32).view(-1, 1)
    y_pinn_pred = pinn(t_plot_tensor)
    y1_pinn_pred = y_pinn_pred[:, 0]

plt.figure(figsize=(10, 6))
plt.plot(t_eval, y1_true, 'b-', linewidth=3, alpha=0.8, label='Ground Truth y(t) = cos(t)')
plt.plot(t_eval, y1_pinn_pred.numpy(), 'r--', linewidth=2, label='PINN Prediction y(t)')
plt.title('Harmonic Oscillator - Solved with Final Corrected PINN')
plt.xlabel('Time'); plt.ylabel('Position y(t)'); plt.legend(); plt.grid(True)
plt.show()