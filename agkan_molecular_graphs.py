"""
AGKAN on Molecular Graphs (QM9 Dataset) - FIXED
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_add_pool
import numpy as np

# ============================================================================
# MinimalKAN (same as before)
# ============================================================================

class MinimalKAN_Simple(nn.Module):
    """Simple KAN with [x, x^2] basis functions"""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.coeffs = nn.Parameter(torch.randn(in_features, out_features, 2) * 0.01)
        self.linear = nn.Linear(in_features, out_features)
        self.alpha = nn.Parameter(torch.tensor(0.1))
        
    def forward(self, x):
        linear_out = self.linear(x)
        
        x_poly = x.unsqueeze(-1)
        basis = torch.stack([x_poly, x_poly**2], dim=-1).squeeze(2)
        basis = basis.unsqueeze(2)
        coeffs = self.coeffs.unsqueeze(0)
        kan_out = (basis * coeffs).sum(dim=-1).sum(dim=1)
        
        alpha = torch.sigmoid(self.alpha)
        return (1 - alpha) * linear_out + alpha * kan_out


# ============================================================================
# Molecular AGKAN
# ============================================================================

class MolecularAGKAN(nn.Module):
    """AGKAN for molecular property prediction"""
    def __init__(self, num_node_features, hidden_dim, num_layers=3, use_kan=True):
        super().__init__()
        self.num_layers = num_layers
        self.use_kan = use_kan
        
        # Initial embedding
        if use_kan:
            self.node_embedding = MinimalKAN_Simple(num_node_features, hidden_dim)
        else:
            self.node_embedding = nn.Linear(num_node_features, hidden_dim)
        
        # Message passing layers
        self.message_layers = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
        
        for _ in range(num_layers):
            if use_kan:
                self.message_layers.append(MinimalKAN_Simple(hidden_dim, hidden_dim))
            else:
                self.message_layers.append(nn.Linear(hidden_dim, hidden_dim))
            
            self.attention_layers.append(nn.Linear(hidden_dim * 2, 1))
        
        # Readout layers
        self.readout1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.readout2 = nn.Linear(hidden_dim // 2, 1)
        
        self.dropout = 0.1
        
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Initial embedding
        x = self.node_embedding(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Message passing
        for layer_idx in range(self.num_layers):
            messages = self.message_layers[layer_idx](x)
            messages = F.elu(messages)
            
            x = x + self.aggregate_with_attention(
                messages, x, edge_index, self.attention_layers[layer_idx]
            )
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Global pooling
        x = global_add_pool(x, batch)
        
        # Readout
        x = self.readout1(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.readout2(x)
        
        return x.squeeze(-1)
    
    def aggregate_with_attention(self, messages, x, edge_index, att_layer):
        source, target = edge_index
        num_nodes = x.size(0)
        
        x_source = x[source]
        x_target = x[target]
        x_cat = torch.cat([x_source, x_target], dim=-1)
        
        alpha = att_layer(x_cat).squeeze(-1)
        alpha = F.leaky_relu(alpha, 0.2)
        alpha = self.edge_softmax(alpha, target, num_nodes)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        
        weighted_messages = messages[source] * alpha.unsqueeze(-1)
        
        out = torch.zeros_like(messages)
        out.index_add_(0, target, weighted_messages)
        
        return out
    
    def edge_softmax(self, alpha, target, num_nodes):
        max_alpha = torch.full((num_nodes,), float('-inf'), device=alpha.device)
        max_alpha.scatter_reduce_(0, target, alpha, reduce='amax', include_self=False)
        alpha = torch.exp(alpha - max_alpha[target])
        sum_alpha = torch.zeros(num_nodes, device=alpha.device)
        sum_alpha.index_add_(0, target, alpha)
        alpha = alpha / (sum_alpha[target] + 1e-16)
        return alpha


# ============================================================================
# Training and Evaluation
# ============================================================================

def train_epoch(model, loader, optimizer, device, target_idx):
    model.train()
    total_loss = 0
    
    for data in loader:
        data = data.to(device)

        # CLIP INPUT FEATURES (molecular features can have extreme values)
        data.x = torch.clamp(data.x, -10, 10)

        optimizer.zero_grad()
        
        out = model(data)
        # Extract only the target property we care about
        target = data.y[:, target_idx]
        loss = F.mse_loss(out, target)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item() * data.num_graphs
    
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device, target_idx):
    model.eval()
    total_loss = 0
    
    for data in loader:
        data = data.to(device)

        # CLIP INPUT FEATURES
        data.x = torch.clamp(data.x, -10, 10)

        out = model(data)
        target = data.y[:, target_idx]
        loss = F.mse_loss(out, target)
        total_loss += loss.item() * data.num_graphs
    
    return total_loss / len(loader.dataset)


def train_model(model, train_loader, val_loader, device, target_idx, epochs=100, lr=0.001):
    """Train model and return best validation MAE"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.7, patience=10, min_lr=1e-6
    )
    
    best_val_loss = float('inf')
    patience = 20
    patience_counter = 0
    
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device, target_idx)
        val_loss = evaluate(model, val_loader, device, target_idx)
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"    Early stopping at epoch {epoch}")
            break
        
        if epoch % 10 == 0:
            print(f"    Epoch {epoch:03d}: Train MAE={train_loss:.4f}, Val MAE={val_loss:.4f}")
    
    return best_val_loss


# ============================================================================
# Multi-seed evaluation
# ============================================================================

def run_molecular_experiment(target_idx=7, n_seeds=5, subset_size=10000):
    """
    Run AGKAN vs baseline on QM9
    
    Args:
        target_idx: Which molecular property to predict (7 = HOMO energy)
        n_seeds: Number of random seeds
        subset_size: Use subset of data (None = full dataset)
    """
    print("="*70)
    print(f"QM9 Molecular Property Prediction")
    print(f"Target: Property {target_idx}")
    print(f"Seeds: {n_seeds}")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
    print(f"Using device: {device}\n")
    
    # Load QM9 dataset
    print("Loading QM9 dataset...")
    dataset = QM9(root='./data/QM9')
    
    # Use subset for faster experimentation
    if subset_size and subset_size < len(dataset):
        indices = torch.randperm(len(dataset))[:subset_size]
        dataset = dataset[indices]
        print(f"Using subset: {len(dataset)} molecules")
    else:
        print(f"Full dataset: {len(dataset)} molecules")
    
    # # Get target statistics for normalization
    # all_targets = torch.stack([data.y[target_idx] for data in dataset])
    # mean = all_targets.mean()
    # std = all_targets.std()
    
    # print(f"Target statistics: mean={mean:.4f}, std={std:.4f}")
    
    # # Normalize targets in-place
    # for data in dataset:
    #     data.y[target_idx] = (data.y[target_idx] - mean) / std
    
    # Get target statistics for normalization
    # Note: After subsetting, need to access original y shape
    all_targets = []
    for data in dataset:
        if data.y.dim() == 1:  # If y is 1D (single molecule)
            all_targets.append(data.y[target_idx].item())
        else:  # If y is 2D (batch)
            all_targets.append(data.y[0, target_idx].item())

    all_targets = torch.tensor(all_targets)
    mean = all_targets.mean()
    std = all_targets.std()

    print(f"Target statistics: mean={mean:.4f}, std={std:.4f}")

    # Normalize targets in-place
    for data in dataset:
        if data.y.dim() == 1:
            data.y[target_idx] = (data.y[target_idx] - mean) / std
        else:
            data.y[0, target_idx] = (data.y[0, target_idx] - mean) / std

    # Split dataset
    n = len(dataset)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    
    train_dataset = dataset[:n_train]
    val_dataset = dataset[n_train:n_train + n_val]
    test_dataset = dataset[n_train + n_val:]
    
    print(f"\nSplits: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    
    # Results storage
    agkan_results = []
    baseline_results = []
    agkan_alphas = {'embed': [], 'layers': [[] for _ in range(3)]}
    
    # Run multiple seeds
    for seed in range(n_seeds):
        print(f"\n{'='*70}")
        print(f"Seed {seed + 1}/{n_seeds}")
        print(f"{'='*70}")
        
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Data loaders
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        # Train baseline (no KAN)
        print("\nTraining Baseline (Attention without KAN)...")
        baseline_model = MolecularAGKAN(
            num_node_features=dataset.num_node_features,
            hidden_dim=64,
            num_layers=3,
            use_kan=False
        ).to(device)
        
        baseline_val = train_model(baseline_model, train_loader, val_loader, device, target_idx, epochs=50)
        baseline_test = evaluate(baseline_model, test_loader, device, target_idx)
        baseline_results.append(baseline_test)
        
        print(f"  Baseline - Val MAE: {baseline_val:.4f}, Test MAE: {baseline_test:.4f}")
        
        # Train AGKAN
        print("\nTraining AGKAN (Attention with KAN)...")
        agkan_model = MolecularAGKAN(
            num_node_features=dataset.num_node_features,
            hidden_dim=64,
            num_layers=3,
            use_kan=True
        ).to(device)
        
        agkan_val = train_model(agkan_model, train_loader, val_loader, device, target_idx, epochs=50)
        agkan_test = evaluate(agkan_model, test_loader, device, target_idx)
        agkan_results.append(agkan_test)
        
        print(f"  AGKAN - Val MAE: {agkan_val:.4f}, Test MAE: {agkan_test:.4f}")
        
        # Extract learned alphas
        if hasattr(agkan_model.node_embedding, 'alpha'):
            alpha_embed = torch.sigmoid(agkan_model.node_embedding.alpha).item()
            agkan_alphas['embed'].append(alpha_embed)
            print(f"  Learned alpha (embedding): {alpha_embed:.4f}")
            
            for i, layer in enumerate(agkan_model.message_layers):
                if hasattr(layer, 'alpha'):
                    alpha = torch.sigmoid(layer.alpha).item()
                    agkan_alphas['layers'][i].append(alpha)
                    print(f"  Learned alpha (layer {i+1}): {alpha:.4f}")
    
    # Compute statistics
    baseline_results = np.array(baseline_results) * std.item()  # Denormalize
    agkan_results = np.array(agkan_results) * std.item()
    
    print(f"\n{'='*70}")
    print("FINAL RESULTS (Test MAE in original units)")
    print(f"{'='*70}")
    print(f"Baseline:    {baseline_results.mean():.4f} ± {baseline_results.std():.4f}")
    print(f"AGKAN:       {agkan_results.mean():.4f} ± {agkan_results.std():.4f}")
    improvement = baseline_results.mean() - agkan_results.mean()
    pct_improvement = (improvement / baseline_results.mean()) * 100
    print(f"Improvement: {improvement:.4f} ({pct_improvement:.2f}%)")
    print(f"{'='*70}")
    
    # Alpha statistics
    print(f"\nLearned Alpha Statistics (across {n_seeds} seeds):")
    print(f"  Embedding:  {np.mean(agkan_alphas['embed']):.4f} ± {np.std(agkan_alphas['embed']):.4f}")
    for i in range(3):
        print(f"  Layer {i+1}:    {np.mean(agkan_alphas['layers'][i]):.4f} ± {np.std(agkan_alphas['layers'][i]):.4f}")
    
    # Statistical test
    try:
        from scipy import stats
        t_stat, p_value = stats.ttest_rel(baseline_results, agkan_results)
        print(f"\nPaired t-test:")
        print(f"  t-statistic: {t_stat:.4f}")
        print(f"  p-value: {p_value:.4f}")
        
        if p_value < 0.05:
            winner = "AGKAN" if agkan_results.mean() < baseline_results.mean() else "Baseline"
            print(f"  ✓ {winner} is significantly better (p < 0.05)")
        else:
            print(f"  ✗ No significant difference (p >= 0.05)")
    except ImportError:
        print("\nInstall scipy for statistical testing: pip install scipy")
    
    return {
        'baseline': baseline_results,
        'agkan': agkan_results,
        'alphas': agkan_alphas,
        'target_idx': target_idx
    }


# ============================================================================
# Main
# ============================================================================

def main():
    print("\nStarting molecular graph experiment...")
    print("This may take 15-20 minutes.\n")
    
    # Run on HOMO energy (target 7)
    results = run_molecular_experiment(
        target_idx=7,
        n_seeds=5,
        subset_size=10000
    )
    
    # Save results
    torch.save(results, 'qm9_results.pt')
    print("\nResults saved to 'qm9_results.pt'")


if __name__ == "__main__":
    main()