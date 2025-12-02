import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

# ============================================================================
# Tweak 1: Adaptive scaling
# ============================================================================

class MinimalKAN_Adaptive(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.coeffs = nn.Parameter(torch.randn(in_features, out_features, 3) * 0.01)
        self.linear = nn.Linear(in_features, out_features)
        self.alpha = nn.Parameter(torch.tensor(0.1))  # Learnable
        
    def forward(self, x):
        linear_out = self.linear(x)
        
        x_poly = x.unsqueeze(-1)
        basis = torch.stack([x_poly, x_poly**2, x_poly**3], dim=-1).squeeze(2)
        basis = basis.unsqueeze(2)
        coeffs = self.coeffs.unsqueeze(0)
        kan_out = (basis * coeffs).sum(dim=-1).sum(dim=1)
        
        alpha = torch.sigmoid(self.alpha)
        return (1 - alpha) * linear_out + alpha * kan_out


# ============================================================================
# Tweak 2: Fewer basis functions
# ============================================================================

class MinimalKAN_Simple(nn.Module):
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
# Tweak 3: Normalize KAN output
# ============================================================================

class MinimalKAN_Normalized(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.coeffs = nn.Parameter(torch.randn(in_features, out_features, 3) * 0.01)
        self.linear = nn.Linear(in_features, out_features)
        self.alpha = nn.Parameter(torch.tensor(0.1))
        
    def forward(self, x):
        linear_out = self.linear(x)
        
        x_poly = x.unsqueeze(-1)
        basis = torch.stack([x_poly, x_poly**2, x_poly**3], dim=-1).squeeze(2)
        basis = basis.unsqueeze(2)
        coeffs = self.coeffs.unsqueeze(0)
        kan_out = (basis * coeffs).sum(dim=-1).sum(dim=1)
        
        # Normalize by sqrt(in_features) to prevent explosion
        kan_out = kan_out / (self.in_features ** 0.5)
        
        alpha = torch.sigmoid(self.alpha)
        return (1 - alpha) * linear_out + alpha * kan_out


# ============================================================================
# Generic AGKAN class that uses any KAN
# ============================================================================

class AGKAN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, 
                 kan_class, dropout=0.6):
        super().__init__()
        self.kan1 = kan_class(in_features, hidden_features)
        self.kan2 = kan_class(hidden_features, out_features)
        
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


# ============================================================================
# Training
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


def run_experiment(model_name, model, data, epochs=200):
    print(f"\n{'='*60}")
    print(f"Training {model_name}...")
    print(f"Parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"{'='*60}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    
    best_val_acc = 0
    best_test_acc = 0
    
    for epoch in range(1, epochs + 1):
        loss = train(model, data, optimizer)
        train_acc, val_acc, test_acc = test(model, data)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
        
        if epoch % 40 == 0:
            print(f'Epoch {epoch:03d}: Loss={loss:.4f}, Train={train_acc:.4f}, '
                  f'Val={val_acc:.4f}, Test={test_acc:.4f}')
    
    print(f"\nBest Val: {best_val_acc:.4f}, Test at Best Val: {best_test_acc:.4f}")
    
    # Check learned alpha values
    if hasattr(model.kan1, 'alpha'):
        alpha1 = torch.sigmoid(model.kan1.alpha).item()
        alpha2 = torch.sigmoid(model.kan2.alpha).item()
        print(f"Learned alphas: Layer1={alpha1:.4f}, Layer2={alpha2:.4f}")
    
    return best_test_acc


# ============================================================================
# Main
# ============================================================================

def main():
    dataset = Planetoid(root='./data', name='Cora', transform=NormalizeFeatures())
    data = dataset[0]
    
    print(f"Dataset: {dataset.name}")
    print(f"Nodes: {data.num_nodes}, Edges: {data.num_edges}")
    
    results = {}
    
    # Baseline (from previous runs)
    print("\nBaseline Results (from previous experiments):")
    print("AttentionGNN: 82.7%")
    print("AGKAN (original): 81.1%")
    
    # Tweak 1: Adaptive scaling
    model1 = AGKAN(
        in_features=dataset.num_features,
        hidden_features=64,
        out_features=dataset.num_classes,
        kan_class=MinimalKAN_Adaptive,
        dropout=0.6
    )
    results['Adaptive'] = run_experiment("AGKAN-Adaptive", model1, data)
    
    # Tweak 2: Fewer basis functions
    model2 = AGKAN(
        in_features=dataset.num_features,
        hidden_features=64,
        out_features=dataset.num_classes,
        kan_class=MinimalKAN_Simple,
        dropout=0.6
    )
    results['Simple'] = run_experiment("AGKAN-Simple", model2, data)
    
    # Tweak 3: Normalized
    model3 = AGKAN(
        in_features=dataset.num_features,
        hidden_features=64,
        out_features=dataset.num_classes,
        kan_class=MinimalKAN_Normalized,
        dropout=0.6
    )
    results['Normalized'] = run_experiment("AGKAN-Normalized", model3, data)
    
    # Summary
    print(f"\n{'='*60}")
    print("FINAL COMPARISON:")
    print(f"{'='*60}")
    print(f"AttentionGNN:          82.7%") # See attention_gnn.py
    print(f"AGKAN (original):      81.1%") # See attention_gnn.py
    print(f"AGKAN-Adaptive:        {results['Adaptive']:.1%}")
    print(f"AGKAN-Simple:          {results['Simple']:.1%}")
    print(f"AGKAN-Normalized:      {results['Normalized']:.1%}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()