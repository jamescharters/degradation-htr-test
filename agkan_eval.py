import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
import numpy as np

# ============================================================================
# Model Definitions
# ============================================================================

class MinimalKAN_Simple(nn.Module):
    """Simple KAN with just [x, x^2] basis functions"""
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


class AGKAN(nn.Module):
    """AGKAN with simple KAN"""
    def __init__(self, in_features, hidden_features, out_features, dropout=0.6):
        super().__init__()
        self.kan1 = MinimalKAN_Simple(in_features, hidden_features)
        self.kan2 = MinimalKAN_Simple(hidden_features, out_features)
        
        self.att1 = nn.Linear(hidden_features * 2, 1)
        self.att2 = nn.Linear(out_features * 2, 1)
        
        self.dropout = dropout
        
    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.kan1(x)
        x = F.elu(x)
        x = self.aggregate_with_attention(x, edge_index, self.att1)
        
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.kan2(x)
        x = self.aggregate_with_attention(x, edge_index, self.att2)
        
        return F.log_softmax(x, dim=-1)
    
    def aggregate_with_attention(self, x, edge_index, att_layer):
        source, target = edge_index
        num_nodes = x.size(0)
        
        x_source = x[source]
        x_target = x[target]
        x_cat = torch.cat([x_source, x_target], dim=-1)
        
        alpha = att_layer(x_cat).squeeze(-1)
        alpha = F.leaky_relu(alpha, 0.2)
        alpha = self.edge_softmax(alpha, target, num_nodes)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        
        messages = x_source * alpha.unsqueeze(-1)
        
        out = torch.zeros_like(x)
        out.index_add_(0, target, messages)
        
        return out
    
    def edge_softmax(self, alpha, target, num_nodes):
        max_alpha = torch.full((num_nodes,), float('-inf'), device=alpha.device)
        max_alpha.scatter_reduce_(0, target, alpha, reduce='amax', include_self=False)
        alpha = torch.exp(alpha - max_alpha[target])
        sum_alpha = torch.zeros(num_nodes, device=alpha.device)
        sum_alpha.index_add_(0, target, alpha)
        alpha = alpha / (sum_alpha[target] + 1e-16)
        return alpha


class AttentionGNN(nn.Module):
    """Baseline attention GNN (no KAN)"""
    def __init__(self, in_features, hidden_features, out_features, dropout=0.6):
        super().__init__()
        self.lin1 = nn.Linear(in_features, hidden_features)
        self.lin2 = nn.Linear(hidden_features, out_features)
        
        self.att1 = nn.Linear(hidden_features * 2, 1)
        self.att2 = nn.Linear(out_features * 2, 1)
        
        self.dropout = dropout
        
    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin1(x)
        x = F.elu(x)
        x = self.aggregate_with_attention(x, edge_index, self.att1)
        
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        x = self.aggregate_with_attention(x, edge_index, self.att2)
        
        return F.log_softmax(x, dim=-1)
    
    def aggregate_with_attention(self, x, edge_index, att_layer):
        source, target = edge_index
        num_nodes = x.size(0)
        
        x_source = x[source]
        x_target = x[target]
        x_cat = torch.cat([x_source, x_target], dim=-1)
        
        alpha = att_layer(x_cat).squeeze(-1)
        alpha = F.leaky_relu(alpha, 0.2)
        alpha = self.edge_softmax(alpha, target, num_nodes)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        
        messages = x_source * alpha.unsqueeze(-1)
        
        out = torch.zeros_like(x)
        out.index_add_(0, target, messages)
        
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
# Training Functions
# ============================================================================

def train(model, data, optimizer):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def test(model, data):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=-1)
    
    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        correct = (pred[mask] == data.y[mask]).sum()
        accs.append(int(correct) / int(mask.sum()))
    return accs


def train_single_run(model, data, seed, epochs=200, lr=0.01, weight_decay=5e-4, verbose=False):
    """Train a single model and return best test accuracy"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    best_val_acc = 0
    best_test_acc = 0
    patience = 100
    patience_counter = 0
    
    for epoch in range(1, epochs + 1):
        loss = train(model, data, optimizer)
        train_acc, val_acc, test_acc = test(model, data)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            if verbose:
                print(f"  Early stopping at epoch {epoch}")
            break
        
        if verbose and epoch % 50 == 0:
            print(f"  Epoch {epoch:03d}: Loss={loss:.4f}, Val={val_acc:.4f}, Test={test_acc:.4f}")
    
    return best_test_acc, best_val_acc


# ============================================================================
# Multi-Seed Evaluation
# ============================================================================

def run_multi_seed_evaluation(model_class, model_name, data, dataset, n_seeds=10, epochs=200):
    """Run evaluation across multiple random seeds"""
    print(f"\n{'='*70}")
    print(f"Running {model_name} with {n_seeds} different seeds")
    print(f"{'='*70}")
    
    test_accs = []
    val_accs = []
    
    for seed in range(n_seeds):
        print(f"\nSeed {seed + 1}/{n_seeds}...", end=" ")
        
        # Initialize model with current seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        model = model_class(
            in_features=dataset.num_features,
            hidden_features=64,
            out_features=dataset.num_classes,
            dropout=0.6
        )
        
        # Train
        test_acc, val_acc = train_single_run(model, data, seed, epochs=epochs, verbose=False)
        test_accs.append(test_acc)
        val_accs.append(val_acc)
        
        print(f"Val: {val_acc:.4f}, Test: {test_acc:.4f}")
        
        # Extract learned alphas if available
        if hasattr(model, 'kan1') and hasattr(model.kan1, 'alpha'):
            alpha1 = torch.sigmoid(model.kan1.alpha).item()
            alpha2 = torch.sigmoid(model.kan2.alpha).item()
            print(f"         Alphas: Layer1={alpha1:.4f}, Layer2={alpha2:.4f}")
    
    # Compute statistics
    test_accs = np.array(test_accs)
    val_accs = np.array(val_accs)
    
    print(f"\n{'='*70}")
    print(f"Results for {model_name}:")
    print(f"{'='*70}")
    print(f"Test Accuracy:")
    print(f"  Mean:   {test_accs.mean():.4f} ± {test_accs.std():.4f}")
    print(f"  Median: {np.median(test_accs):.4f}")
    print(f"  Min:    {test_accs.min():.4f}")
    print(f"  Max:    {test_accs.max():.4f}")
    print(f"\nValidation Accuracy:")
    print(f"  Mean:   {val_accs.mean():.4f} ± {val_accs.std():.4f}")
    print(f"{'='*70}")
    
    return {
        'test_accs': test_accs,
        'val_accs': val_accs,
        'test_mean': test_accs.mean(),
        'test_std': test_accs.std(),
        'test_median': np.median(test_accs),
        'test_min': test_accs.min(),
        'test_max': test_accs.max()
    }


# ============================================================================
# Statistical Significance Test
# ============================================================================

def paired_t_test(results1, results2, name1, name2):
    """Perform paired t-test to check if difference is statistically significant"""
    from scipy import stats
    
    accs1 = results1['test_accs']
    accs2 = results2['test_accs']
    
    # Paired t-test
    t_stat, p_value = stats.ttest_rel(accs1, accs2)
    
    print(f"\n{'='*70}")
    print(f"Statistical Significance Test: {name1} vs {name2}")
    print(f"{'='*70}")
    print(f"Mean difference: {accs1.mean() - accs2.mean():.4f}")
    print(f"t-statistic: {t_stat:.4f}")
    print(f"p-value: {p_value:.4f}")
    
    if p_value < 0.05:
        winner = name1 if accs1.mean() > accs2.mean() else name2
        print(f"\n✓ The difference is statistically significant (p < 0.05)")
        print(f"  {winner} is significantly better!")
    else:
        print(f"\n✗ The difference is NOT statistically significant (p >= 0.05)")
        print(f"  The models perform similarly.")
    print(f"{'='*70}")


# ============================================================================
# Main Execution
# ============================================================================

def main():
    # Configuration
    N_SEEDS = 10
    EPOCHS = 200
    
    # Load dataset
    print("Loading Cora dataset...")
    dataset = Planetoid(root='./data', name='Cora', transform=NormalizeFeatures())
    data = dataset[0]
    
    print(f"\nDataset: {dataset.name}")
    print(f"Nodes: {data.num_nodes}, Edges: {data.num_edges}")
    print(f"Features: {dataset.num_features}, Classes: {dataset.num_classes}")
    
    # Run AttentionGNN (baseline)
    att_results = run_multi_seed_evaluation(
        AttentionGNN, 
        "AttentionGNN (Baseline)", 
        data, 
        dataset, 
        n_seeds=N_SEEDS,
        epochs=EPOCHS
    )
    
    # Run AGKAN-Simple
    agkan_results = run_multi_seed_evaluation(
        AGKAN,
        "AGKAN-Simple",
        data,
        dataset,
        n_seeds=N_SEEDS,
        epochs=EPOCHS
    )
    
    # Statistical comparison
    try:
        paired_t_test(agkan_results, att_results, "AGKAN-Simple", "AttentionGNN")
    except ImportError:
        print("\nNote: Install scipy for statistical significance testing: pip install scipy")
    
    # Final summary
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"AttentionGNN:   {att_results['test_mean']:.4f} ± {att_results['test_std']:.4f}")
    print(f"AGKAN-Simple:   {agkan_results['test_mean']:.4f} ± {agkan_results['test_std']:.4f}")
    print(f"Improvement:    {(agkan_results['test_mean'] - att_results['test_mean']) * 100:.2f} percentage points")
    print(f"{'='*70}")
    
    # Save results
    results_dict = {
        'AttentionGNN': att_results,
        'AGKAN-Simple': agkan_results
    }
    
    torch.save(results_dict, 'multi_seed_results.pt')
    print("\nResults saved to 'multi_seed_results.pt'")


if __name__ == "__main__":
    main()