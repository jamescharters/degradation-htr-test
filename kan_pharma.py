import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error


def generate_pk_data(n_samples=1000, noise_level=0.05):
    """
    Generate realistic pharmacokinetics data
    
    Two-compartment model with first-order absorption:
    C(t) = (F*Dose*ka / V*(ka-ke)) * (e^(-ke*t) - e^(-ka*t))
    
    Where:
    - F: bioavailability (0.7-1.0)
    - Dose: drug dose (50-500 mg)
    - ka: absorption rate (0.5-3.0 hr^-1)
    - ke: elimination rate, function of clearance and volume
    - V: volume of distribution (10-100 L)
    - CL: clearance (1-20 L/hr)
    - Weight affects V and CL
    - Age affects CL (reduced in elderly)
    """
    
    # Patient characteristics
    dose = np.random.uniform(50, 500, n_samples)  # mg
    weight = np.random.uniform(50, 120, n_samples)  # kg
    age = np.random.uniform(18, 85, n_samples)  # years
    creatinine = np.random.uniform(0.5, 2.5, n_samples)  # renal function marker
    time = np.random.uniform(0.5, 24, n_samples)  # hours post-dose
    
    # Physiological parameters (with realistic correlations)
    F = np.random.uniform(0.75, 0.95, n_samples)  # bioavailability
    ka = np.random.uniform(0.8, 2.5, n_samples)  # absorption rate
    
    # Volume increases with weight
    V = 0.6 * weight + np.random.normal(0, 5, n_samples)
    V = np.clip(V, 20, 100)
    
    # Clearance depends on weight, age, and renal function
    CL = 0.15 * weight * (1 - 0.005 * (age - 40)) / creatinine
    CL = np.clip(CL, 2, 25)
    
    # Elimination rate
    ke = CL / V
    
    # Two-compartment model
    concentration = (F * dose * ka / (V * (ka - ke))) * \
                   (np.exp(-ke * time) - np.exp(-ka * time))
    
    # Add realistic measurement noise (proportional + additive)
    noise = noise_level * concentration * np.random.randn(n_samples) + \
            0.1 * np.random.randn(n_samples)
    concentration = np.maximum(0, concentration + noise)
    
    # Create input features
    X = np.column_stack([dose, weight, age, creatinine, time])
    y = concentration
    
    return torch.FloatTensor(X), torch.FloatTensor(y)


class KANLayer(nn.Module):
    """Single KAN layer with learnable basis functions"""
    def __init__(self, in_features, out_features, n_basis=8):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_basis = n_basis
        
        # Learnable spline coefficients for each input-output pair
        self.coeffs = nn.Parameter(
            torch.randn(out_features, in_features, n_basis) * 0.1
        )
        
        # Fixed basis centers (B-spline knots)
        self.register_buffer(
            'centers', 
            torch.linspace(-2, 2, n_basis)
        )
    
    def basis_functions(self, x):
        """Gaussian radial basis functions (simplified B-splines)"""
        # x: (batch, in_features)
        # output: (batch, in_features, n_basis)
        x_expanded = x.unsqueeze(-1)  # (batch, in_features, 1)
        centers = self.centers.unsqueeze(0).unsqueeze(0)  # (1, 1, n_basis)
        return torch.exp(-((x_expanded - centers) ** 2) / 0.5)
    
    def forward(self, x):
        # Normalize inputs
        x_norm = (x - x.mean(0)) / (x.std(0) + 1e-6)
        x_norm = torch.clamp(x_norm, -3, 3)
        
        # Compute basis functions
        basis = self.basis_functions(x_norm)  # (batch, in_features, n_basis)
        
        # Apply learned coefficients
        # basis: (batch, in_features, n_basis)
        # coeffs: (out_features, in_features, n_basis)
        output = torch.einsum('bin,oin->bo', basis, self.coeffs)
        
        return output


class KANNetwork(nn.Module):
    """Multi-layer KAN for regression"""
    def __init__(self, input_dim=5, hidden_dim=16, n_basis=8):
        super().__init__()
        self.layer1 = KANLayer(input_dim, hidden_dim, n_basis)
        self.layer2 = KANLayer(hidden_dim, hidden_dim, n_basis)
        self.layer3 = KANLayer(hidden_dim, 1, n_basis)
        
        # Add skip connection for better gradient flow
        self.skip = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        h = self.layer1(x)
        h = torch.tanh(h)
        h = self.layer2(h)
        h = torch.tanh(h)
        output = self.layer3(h).squeeze(-1)
        
        # Skip connection helps capture linear relationships
        output = output + self.skip(x).squeeze(-1)
        return output


class MLPBaseline(nn.Module):
    """Standard MLP for comparison"""
    def __init__(self, input_dim=5, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_model(model, X_train, y_train, X_val, y_val, epochs=500, lr=0.001):
    """Train with early stopping"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=50
    )
    
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        pred = model(X_train)
        loss = F.mse_loss(pred, y_train)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = F.mse_loss(val_pred, y_val)
        
        train_losses.append(loss.item())
        val_losses.append(val_loss.item())
        
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter > 100:
                break
    
    return train_losses, val_losses


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def evaluate_model(model, X, y, model_name="Model"):
    """Comprehensive evaluation"""
    model.eval()
    with torch.no_grad():
        pred = model(X)
        if pred.dim() > 1:
            pred = pred.squeeze(-1)
        pred = pred.numpy()
    
    y_np = y.numpy()
    
    r2 = r2_score(y_np, pred)
    mae = mean_absolute_error(y_np, pred)
    rmse = np.sqrt(np.mean((y_np - pred) ** 2))
    # Fix MAPE - only calculate for concentrations > 0.1
    mask = y_np > 0.1
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_np[mask] - pred[mask]) / y_np[mask])) * 100
    else:
        mape = float('nan')
    
    print(f"\n{model_name} Performance:")
    print(f"  R² Score:  {r2:.4f}")
    print(f"  MAE:       {mae:.3f} µg/mL")
    print(f"  RMSE:      {rmse:.3f} µg/mL")
    if not np.isnan(mape):
        print(f"  MAPE:      {mape:.2f}% (for C > 0.1 µg/mL)")
    
    return {'r2': r2, 'mae': mae, 'rmse': rmse, 'mape': mape, 'pred': pred}


def visualize_results(X_test, y_test, kan_results, mlp_results):
    """Create comprehensive visualization"""
    fig = plt.figure(figsize=(16, 10))
    
    # 1. Prediction scatter plots
    ax1 = plt.subplot(2, 3, 1)
    ax1.scatter(y_test, kan_results['pred'], alpha=0.5, s=10)
    ax1.plot([0, y_test.max()], [0, y_test.max()], 'r--', lw=2)
    ax1.set_xlabel('True Concentration (µg/mL)')
    ax1.set_ylabel('Predicted Concentration (µg/mL)')
    ax1.set_title(f'KAN: R²={kan_results["r2"]:.3f}')
    ax1.grid(True, alpha=0.3)
    
    ax2 = plt.subplot(2, 3, 2)
    ax2.scatter(y_test, mlp_results['pred'], alpha=0.5, s=10, color='orange')
    ax2.plot([0, y_test.max()], [0, y_test.max()], 'r--', lw=2)
    ax2.set_xlabel('True Concentration (µg/mL)')
    ax2.set_ylabel('Predicted Concentration (µg/mL)')
    ax2.set_title(f'MLP: R²={mlp_results["r2"]:.3f}')
    ax2.grid(True, alpha=0.3)
    
    # 2. Residual plots
    ax3 = plt.subplot(2, 3, 3)
    kan_residuals = y_test.numpy() - kan_results['pred']
    mlp_residuals = y_test.numpy() - mlp_results['pred']
    ax3.scatter(kan_results['pred'], kan_residuals, alpha=0.5, s=10, label='KAN')
    ax3.scatter(mlp_results['pred'], mlp_residuals, alpha=0.5, s=10, label='MLP')
    ax3.axhline(0, color='r', linestyle='--', lw=2)
    ax3.set_xlabel('Predicted Concentration (µg/mL)')
    ax3.set_ylabel('Residual (µg/mL)')
    ax3.set_title('Residual Analysis')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 3. Time-concentration curves for different doses
    ax4 = plt.subplot(2, 3, 4)
    times = torch.linspace(0.5, 24, 100)
    for dose in [100, 200, 400]:
        X_curve = torch.zeros(100, 5)
        X_curve[:, 0] = dose  # dose
        X_curve[:, 1] = 70    # weight
        X_curve[:, 2] = 40    # age
        X_curve[:, 3] = 1.0   # creatinine
        X_curve[:, 4] = times # time
        
        with torch.no_grad():
            kan_curve = kan_model(X_curve).numpy()
            mlp_curve = mlp_model(X_curve).numpy()
        
        ax4.plot(times, kan_curve, '-', label=f'{dose}mg (KAN)', linewidth=2)
        ax4.plot(times, mlp_curve, '--', label=f'{dose}mg (MLP)', alpha=0.7)
    
    ax4.set_xlabel('Time (hours)')
    ax4.set_ylabel('Concentration (µg/mL)')
    ax4.set_title('Dose Response Curves')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)
    
    # 4. Weight effect (at fixed time)
    ax5 = plt.subplot(2, 3, 5)
    weights = torch.linspace(50, 120, 100)
    X_weight = torch.zeros(100, 5)
    X_weight[:, 0] = 200  # dose
    X_weight[:, 1] = weights
    X_weight[:, 2] = 40   # age
    X_weight[:, 3] = 1.0  # creatinine
    X_weight[:, 4] = 4    # time
    
    with torch.no_grad():
        kan_weight = kan_model(X_weight).numpy()
        mlp_weight = mlp_model(X_weight).numpy()
    
    ax5.plot(weights, kan_weight, '-', label='KAN', linewidth=2)
    ax5.plot(weights, mlp_weight, '--', label='MLP', linewidth=2)
    ax5.set_xlabel('Patient Weight (kg)')
    ax5.set_ylabel('Concentration at 4h (µg/mL)')
    ax5.set_title('Weight Effect on Drug Levels')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 5. Age effect
    ax6 = plt.subplot(2, 3, 6)
    ages = torch.linspace(18, 85, 100)
    X_age = torch.zeros(100, 5)
    X_age[:, 0] = 200  # dose
    X_age[:, 1] = 70   # weight
    X_age[:, 2] = ages
    X_age[:, 3] = 1.0  # creatinine
    X_age[:, 4] = 4    # time
    
    with torch.no_grad():
        kan_age = kan_model(X_age).numpy()
        mlp_age = mlp_model(X_age).numpy()
    
    ax6.plot(ages, kan_age, '-', label='KAN', linewidth=2)
    ax6.plot(ages, mlp_age, '--', label='MLP', linewidth=2)
    ax6.set_xlabel('Patient Age (years)')
    ax6.set_ylabel('Concentration at 4h (µg/mL)')
    ax6.set_title('Age Effect on Drug Levels')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pk_comparison.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved visualization to 'pk_comparison.png'")
    plt.show()


if __name__ == '__main__':
    print("=" * 70)
    print("PHARMACOKINETICS: KAN vs MLP BASELINE")
    print("=" * 70)
    
    # Set seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Generate data
    print("\nGenerating realistic PK data...")
    X_train, y_train = generate_pk_data(n_samples=2000, noise_level=0.05)
    X_val, y_val = generate_pk_data(n_samples=500, noise_level=0.05)
    X_test, y_test = generate_pk_data(n_samples=500, noise_level=0.05)
    
    print(f"Training samples:   {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples:       {len(X_test)}")
    print(f"\nInput features: Dose, Weight, Age, Creatinine, Time")
    print(f"Target: Drug concentration (µg/mL)")
    
    # Train KAN
    print("\n" + "-" * 70)
    print("Training KAN Network...")
    print("-" * 70)
    kan_model = KANNetwork(input_dim=5, hidden_dim=20, n_basis=12)  # Increased capacity
    kan_params = count_parameters(kan_model)
    print(f"KAN Parameters: {kan_params:,}")
    kan_train_loss, kan_val_loss = train_model(
        kan_model, X_train, y_train, X_val, y_val, epochs=1500, lr=0.005  # More epochs, higher LR
    )
    print(f"✓ KAN training completed ({len(kan_train_loss)} epochs)")
    
    # Train MLP
    print("\n" + "-" * 70)
    print("Training MLP Baseline...")
    print("-" * 70)
    mlp_model = MLPBaseline(input_dim=5, hidden_dim=64)
    mlp_params = count_parameters(mlp_model)
    print(f"MLP Parameters: {mlp_params:,}")
    mlp_train_loss, mlp_val_loss = train_model(
        mlp_model, X_train, y_train, X_val, y_val, epochs=1000, lr=0.001
    )
    print(f"✓ MLP training completed ({len(mlp_train_loss)} epochs)")
    
    # Evaluate both models
    print("\n" + "=" * 70)
    print("TEST SET EVALUATION")
    print("=" * 70)
    
    kan_results = evaluate_model(kan_model, X_test, y_test, "KAN")
    mlp_results = evaluate_model(mlp_model, X_test, y_test, "MLP")
    
    # Compare
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    print(f"Model sizes: KAN={kan_params:,} params, MLP={mlp_params:,} params")
    print(f"Parameter ratio: KAN/MLP = {kan_params/mlp_params:.2f}x")
    print()
    r2_improvement = (kan_results['r2'] - mlp_results['r2']) / mlp_results['r2'] * 100
    mae_improvement = (mlp_results['mae'] - kan_results['mae']) / mlp_results['mae'] * 100
    
    print(f"R² improvement:  {r2_improvement:+.1f}%")
    print(f"MAE improvement: {mae_improvement:+.1f}%")
    
    if kan_results['r2'] > mlp_results['r2']:
        print("\n✓ KAN outperforms MLP on this smooth regression task!")
    else:
        print("\n→ MLP competitive on this task")
    
    # Visualize
    print("\nGenerating visualizations...")
    visualize_results(X_test, y_test, kan_results, mlp_results)
    
    print("\n" + "=" * 70)
    print("Key Insight: KANs should excel here because:")
    print("  • Low dimensional inputs (5 features)")
    print("  • Smooth underlying functions (exponential decay)")
    print("  • Clean, structured data with realistic noise")
    print("  • Interpretable relationships between variables")
    print("=" * 70)