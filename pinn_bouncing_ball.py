import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import imageio
import os

# --- Step 1: Generate Synthetic "Ground Truth" Data ---
# We create a perfect simulation of a bouncing ball.

# Physics parameters
g = 9.8       # Gravity
vx0 = 2.0     # Initial horizontal velocity
vy0 = 10.0    # Initial vertical velocity
x0, y0 = 0, 5 # Initial position
e = 0.85      # Coefficient of restitution (how bouncy)
dt = 0.01     # Time step
T = 4.0       # Total simulation time

# Simulation loop
t_full = np.arange(0, T, dt)
x_full, y_full = [], []
vx, vy = vx0, vy0
x, y = x0, y0

for t in t_full:
    # Update position
    x += vx * dt
    y += vy * dt
    # Update velocity
    vy -= g * dt
    # Check for bounce
    if y < 0:
        y = 0
        vy = -vy * e
    x_full.append(x)
    y_full.append(y)

x_full = np.array(x_full)
y_full = np.array(y_full)

# --- Step 2: Create a Sparse Training Dataset ---
# We pick only a few points from the perfect simulation.
# This mimics a real-world scenario with limited data.

num_training_points = 8
idx = np.linspace(0, len(t_full) - 1, num_training_points).astype(int)
t_train = torch.tensor(t_full[idx], dtype=torch.float32).view(-1, 1)
x_train = torch.tensor(x_full[idx], dtype=torch.float32).view(-1, 1)
y_train = torch.tensor(y_full[idx], dtype=torch.float32).view(-1, 1)
pos_train = torch.cat([x_train, y_train], dim=1)

# --- Step 3: Define the Neural Network Architecture ---
# A simple multi-layer perceptron (MLP)
# Input: time (t)
# Output: position (x, y)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 32)
        self.fc4 = nn.Linear(32, 2) # Output is 2D: (x, y)

    def forward(self, t):
        x = torch.tanh(self.fc1(t))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = self.fc4(x)
        return x

# --- Step 4: Train the "Naive" Model ---
# This model only tries to match the sparse data points.

print("Training the Naive Model...")
naive_model = Net()
optimizer = torch.optim.Adam(naive_model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

for epoch in range(15000):
    optimizer.zero_grad()
    pos_pred = naive_model(t_train)
    loss = loss_fn(pos_pred, pos_train)
    loss.backward()
    optimizer.step()
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# --- Step 5: Train the Physically Informed Model (PINN) ---
# This model minimizes both the data error AND the physics error.

print("\nTraining the Physically Informed Model (PINN)...")
pinn_model = Net()
optimizer = torch.optim.Adam(pinn_model.parameters(), lr=1e-3)

# Hyperparameter to balance data vs. physics loss
lambda_physics = 0.01

for epoch in range(15000):
    optimizer.zero_grad()

    # 1. Data Loss (same as the naive model)
    pos_pred_data = pinn_model(t_train)
    data_loss = loss_fn(pos_pred_data, pos_train)

    # 2. Physics Loss
    # We check the physics at random "collocation" points in time
    t_physics = torch.linspace(0, T, 100, requires_grad=True).view(-1, 1)
    pos_pred_physics = pinn_model(t_physics)
    x_pred, y_pred = pos_pred_physics[:, 0], pos_pred_physics[:, 1]

    # Use autograd to get derivatives
    # First derivatives (velocity)
    vx_pred = torch.autograd.grad(x_pred, t_physics, torch.ones_like(x_pred), create_graph=True)[0]
    vy_pred = torch.autograd.grad(y_pred, t_physics, torch.ones_like(y_pred), create_graph=True)[0]
    
    # Second derivatives (acceleration)
    ax_pred = torch.autograd.grad(vx_pred, t_physics, torch.ones_like(vx_pred), create_graph=True)[0]
    ay_pred = torch.autograd.grad(vy_pred, t_physics, torch.ones_like(vy_pred), create_graph=True)[0]
    
    # The physical laws are:
    # ax = 0 (no horizontal force)
    # ay = -g (gravity)
    # We calculate the residuals (how much the model violates the laws)
    physics_loss_x = loss_fn(ax_pred, torch.zeros_like(ax_pred))
    physics_loss_y = loss_fn(ay_pred, -g * torch.ones_like(ay_pred))
    physics_loss = physics_loss_x + physics_loss_y

    # 3. Total Loss
    total_loss = data_loss + lambda_physics * physics_loss
    total_loss.backward()
    optimizer.step()

    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Total Loss: {total_loss.item():.4f}, Data Loss: {data_loss.item():.4f}, Physics Loss: {physics_loss.item():.4f}")

# --- Step 6: Generate Predictions and Create the GIF ---

print("\nGenerating GIF...")
t_eval = torch.tensor(t_full, dtype=torch.float32).view(-1, 1)
with torch.no_grad():
    naive_pred = naive_model(t_eval).numpy()
    pinn_pred = pinn_model(t_eval).numpy()

filenames = []
frame_dir = "bouncing_ball_frames"
os.makedirs(frame_dir, exist_ok=True)

for i in range(0, len(t_full), 4): # Create a frame every 4 time steps
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot Ground Truth
    ax.plot(x_full, y_full, 'k--', label='Ground Truth', alpha=0.4)
    
    # Plot Training Data
    ax.plot(x_train.numpy(), y_train.numpy(), 'ro', markersize=10, label='Training Data')

    # Plot Naive Model Trajectory and Ball
    ax.plot(naive_pred[:i, 0], naive_pred[:i, 1], 'b-', label='Naive Model')
    ax.plot(naive_pred[i, 0], naive_pred[i, 1], 'bo', markersize=15)
    
    # Plot PINN Trajectory and Ball
    ax.plot(pinn_pred[:i, 0], pinn_pred[:i, 1], 'g-', label='PINN')
    ax.plot(pinn_pred[i, 0], pinn_pred[i, 1], 'go', markersize=15)

    # Formatting
    ax.set_xlim(min(x_full) - 1, max(x_full) + 1)
    ax.set_ylim(-1, max(y_full) + 1)
    ax.set_title(f"Bouncing Ball Simulation | Time: {t_full[i]:.2f}s")
    ax.set_xlabel("Horizontal Position (x)")
    ax.set_ylabel("Vertical Position (y)")
    ax.legend(loc='upper right')
    ax.grid(True)
    
    filename = os.path.join(frame_dir, f'frame_{i:04d}.png')
    filenames.append(filename)
    plt.savefig(filename)
    plt.close()

# Create GIF
with imageio.get_writer('bouncing_ball_comparison.gif', mode='I', duration=0.04) as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)

# Clean up frames
for filename in filenames:
    os.remove(filename)
os.rmdir(frame_dir)

print("\nDone! Check for 'bouncing_ball_comparison.gif' in your directory.")
