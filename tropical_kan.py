import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time

# ============================================================================
# MODEL DEFINITIONS
# ============================================================================

class RigorousTropicalKAN(nn.Module):
    """Hardware-efficient piecewise-linear KAN using tropical algebra"""
    def __init__(self, in_dim, out_dim, degree=5):
        super().__init__()
        self.degree = degree
        
        # Fixed integer slopes (bit-shift friendly)
        #self.register_buffer('slopes', torch.arange(degree + 1, dtype=torch.float32))
        self.slopes = nn.Parameter(torch.arange(degree + 1, dtype=torch.float32))
        
        # Learnable intercepts
        scale = 0.5 / (degree + 1)
        self.weights_p = nn.Parameter(torch.randn(in_dim, out_dim, degree + 1) * scale)
        self.weights_q = nn.Parameter(torch.randn(in_dim, out_dim, degree + 1) * scale)
        
        #self.norm = nn.LayerNorm(in_dim)
    
    def forward(self, x):
        #x = self.norm(x)
        x_expanded = x.unsqueeze(2).unsqueeze(3)
        linear_term = x_expanded * self.slopes
        
        term_p = linear_term + self.weights_p
        term_q = linear_term + self.weights_q
        
        if self.training:
            # Smooth tropical (differentiable)
            temp = 1.0
            p_val = temp * torch.logsumexp(term_p / temp, dim=-1)
            q_val = temp * torch.logsumexp(term_q / temp, dim=-1)
        else:
            # Hard max (true tropical, hardware-efficient)
            p_val, _ = torch.max(term_p, dim=-1)
            q_val, _ = torch.max(term_q, dim=-1)
        
        return torch.sum(p_val - q_val, dim=1)


class StandardMLP(nn.Module):
    """Baseline MLP for comparison"""
    def __init__(self, in_dim, out_dim, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )
    
    def forward(self, x):
        return self.net(x)


# ============================================================================
# CONTROL POLICY DATASET (Inverted Pendulum)
# ============================================================================

def optimal_control_policy(state):
    """
    Simple heuristic control policy for inverted pendulum
    state = [angle, angular_velocity]
    Returns control force to keep pendulum upright
    """
    angle, velocity = state[:, 0], state[:, 1]
    
    # Piecewise-linear control strategy (perfect for Tropical KAN!)
    control = torch.zeros_like(angle)
    
    # # If falling right, push left
    # control += torch.where(angle > 0.5, -3.0 * angle, torch.zeros_like(angle))
    # control += torch.where((angle > 0.1) & (angle <= 0.5), -1.5 * angle, torch.zeros_like(angle))
    
    # # If falling left, push right
    # control += torch.where(angle < -0.5, -3.0 * angle, torch.zeros_like(angle))
    # control += torch.where((angle < -0.1) & (angle >= -0.5), -1.5 * angle, torch.zeros_like(angle))
    
    # # Damping based on velocity
    # control -= 0.5 * velocity

    # Pure piecewise linear segments
    control += torch.clamp(angle * -3.0, min=-5, max=5)
    control += torch.clamp(velocity * -0.5, min=-2, max=2)
    
    return control.unsqueeze(1)


def generate_dataset(n_samples=5000):
    """Generate control policy dataset"""
    # Sample state space
    angles = torch.rand(n_samples) * 4 - 2  # [-2, 2] radians
    velocities = torch.rand(n_samples) * 6 - 3  # [-3, 3] rad/s
    
    states = torch.stack([angles, velocities], dim=1)
    actions = optimal_control_policy(states)
    
    return states, actions


# ============================================================================
# TRAINING & EVALUATION
# ============================================================================

def train_model(model, train_x, train_y, epochs=300, lr=0.01):
    """Train a model and return training history"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    losses = []
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        pred = model(train_x)
        loss = criterion(pred, train_y)
        
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if epoch % 50 == 0:
            print(f"  Epoch {epoch:3d}: Loss = {loss.item():.6f}")
    
    return losses


def evaluate_model(model, test_x, test_y):
    """Evaluate model accuracy"""
    model.eval()
    with torch.no_grad():
        pred = model(test_x)
        mse = nn.MSELoss()(pred, test_y).item()
        mae = torch.mean(torch.abs(pred - test_y)).item()
    return mse, mae


def benchmark_inference_speed(model, test_x, n_trials=100):
    """Measure inference time"""
    model.eval()
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(test_x)
    
    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(n_trials):
            start = time.perf_counter()
            _ = model(test_x)
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms
    
    return np.mean(times), np.std(times)


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_control_policy(model, title="Control Policy"):
    """Visualize learned control policy as heatmap"""
    model.eval()
    
    # Create grid
    angles = torch.linspace(-2, 2, 100)
    velocities = torch.linspace(-3, 3, 100)
    A, V = torch.meshgrid(angles, velocities, indexing='ij')
    
    states = torch.stack([A.flatten(), V.flatten()], dim=1)
    
    with torch.no_grad():
        controls = model(states).reshape(100, 100)
    
    plt.figure(figsize=(8, 6))
    plt.contourf(A.numpy(), V.numpy(), controls.numpy(), levels=20, cmap='RdBu_r')
    plt.colorbar(label='Control Force')
    plt.xlabel('Angle (rad)')
    plt.ylabel('Angular Velocity (rad/s)')
    plt.title(title)
    plt.grid(True, alpha=0.3)


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("TROPICAL KAN: CONTROL SYSTEM EXPERIMENT")
    print("=" * 70)
    
    # 1. Generate Dataset
    print("\n[1] Generating Dataset...")
    train_x, train_y = generate_dataset(n_samples=3000)
    test_x, test_y = generate_dataset(n_samples=1000)
    print(f"  Train: {train_x.shape[0]} samples")
    print(f"  Test:  {test_x.shape[0]} samples")
    
    # 2. Initialize Models
    print("\n[2] Initializing Models...")
    models = {
        'Tropical KAN (deg=5)': RigorousTropicalKAN(in_dim=2, out_dim=1, degree=5),
        'Tropical KAN (deg=10)': RigorousTropicalKAN(in_dim=2, out_dim=1, degree=10),
        'Standard MLP': StandardMLP(in_dim=2, out_dim=1, hidden_dim=32)
    }
    
    for name, model in models.items():
        params = count_parameters(model)
        print(f"  {name:25s}: {params:5d} parameters")
    
    # 3. Train Models
    print("\n[3] Training Models...")
    results = {}
    
    for name, model in models.items():
        print(f"\n{name}:")
        losses = train_model(model, train_x, train_y, epochs=300, lr=0.01)
        results[name] = {'losses': losses, 'model': model}
    
    # 4. Evaluate Accuracy
    print("\n[4] Evaluating Accuracy...")
    print(f"\n{'Model':<25s} {'Test MSE':>12s} {'Test MAE':>12s}")
    print("-" * 52)
    
    for name, data in results.items():
        model = data['model']
        mse, mae = evaluate_model(model, test_x, test_y)
        results[name]['mse'] = mse
        results[name]['mae'] = mae
        print(f"{name:<25s} {mse:>12.6f} {mae:>12.6f}")
    
    # 5. Benchmark Inference Speed
    print("\n[5] Benchmarking Inference Speed...")
    print(f"\n{'Model':<25s} {'Mean (ms)':>12s} {'Std (ms)':>12s} {'Speedup':>10s}")
    print("-" * 62)
    
    baseline_time = None
    for name, data in results.items():
        model = data['model']
        mean_time, std_time = benchmark_inference_speed(model, test_x, n_trials=100)
        results[name]['inference_time'] = mean_time
        
        if baseline_time is None:
            baseline_time = mean_time
            speedup = "1.00x"
        else:
            speedup = f"{baseline_time / mean_time:.2f}x"
        
        print(f"{name:<25s} {mean_time:>12.4f} {std_time:>12.4f} {speedup:>10s}")
    
    # 6. Visualizations
    print("\n[6] Generating Visualizations...")
    
    # Training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    for name, data in results.items():
        plt.semilogy(data['losses'], label=name, alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Training Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Accuracy comparison
    plt.subplot(1, 3, 2)
    names = list(results.keys())
    mses = [results[n]['mse'] for n in names]
    plt.bar(range(len(names)), mses, color=['#ff6b6b', '#ff8787', '#4ecdc4'])
    plt.xticks(range(len(names)), names, rotation=45, ha='right')
    plt.ylabel('Test MSE')
    plt.title('Accuracy Comparison')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Speed comparison
    plt.subplot(1, 3, 3)
    times = [results[n]['inference_time'] for n in names]
    plt.bar(range(len(names)), times, color=['#ff6b6b', '#ff8787', '#4ecdc4'])
    plt.xticks(range(len(names)), names, rotation=45, ha='right')
    plt.ylabel('Inference Time (ms)')
    plt.title('Speed Comparison')
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('tropical_kan_comparison.png', dpi=150, bbox_inches='tight')
    print("  Saved: tropical_kan_comparison.png")
    
    # Control policy visualizations
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # True policy
    plt.sca(axes[0])
    visualize_control_policy(lambda x: optimal_control_policy(x), "True Policy")
    
    # Learned policies
    for idx, (name, data) in enumerate(results.items()):
        plt.sca(axes[idx + 1])
        visualize_control_policy(data['model'], name)
    
    plt.tight_layout()
    plt.savefig('tropical_kan_policies.png', dpi=150, bbox_inches='tight')
    print("  Saved: tropical_kan_policies.png")
    
    # 7. Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("\nKey Findings:")
    
    tropical_5 = results['Tropical KAN (deg=5)']
    mlp = results['Standard MLP']
    
    accuracy_ratio = (mlp['mse'] - tropical_5['mse']) / mlp['mse'] * 100
    speed_ratio = mlp['inference_time'] / tropical_5['inference_time']
    
    print(f"  • Tropical KAN (deg=5) accuracy: {accuracy_ratio:+.1f}% vs MLP")
    print(f"  • Tropical KAN (deg=5) speed:    {speed_ratio:.2f}x faster than MLP")
    print(f"  • Parameter efficiency:          {count_parameters(tropical_5['model'])} vs {count_parameters(mlp['model'])} params")
    print("\nConclusion:")
    if accuracy_ratio > -10 and speed_ratio > 1.5:
        print("  ✓ Tropical KAN achieves competitive accuracy with significant speedup!")
        print("    Suitable for edge deployment in control systems.")
    else:
        print("  ⚠ Trade-offs exist. Consider use-case requirements.")
    
    plt.show()