import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

# ============================================================================
# Simplest possible GNN - just linear layers + message passing
# ============================================================================

class SimpleGNN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.lin1 = nn.Linear(in_features, hidden_features)
        self.lin2 = nn.Linear(hidden_features, out_features)
        
    def forward(self, x, edge_index):
        # Layer 1: transform then aggregate
        x = self.lin1(x)
        x = F.relu(x)
        x = self.aggregate(x, edge_index)
        
        # Layer 2: transform then aggregate
        x = self.lin2(x)
        x = self.aggregate(x, edge_index)
        
        return F.log_softmax(x, dim=-1)
    
    def aggregate(self, x, edge_index):
        """Simple mean aggregation of neighbors"""
        source, target = edge_index
        
        # Count neighbors for each node
        num_nodes = x.size(0)
        degree = torch.zeros(num_nodes, device=x.device)
        degree.index_add_(0, target, torch.ones(source.size(0), device=x.device))
        degree = torch.clamp(degree, min=1.0)  # Avoid division by zero
        
        # Aggregate messages
        out = torch.zeros_like(x)
        out.index_add_(0, target, x[source])
        
        # Normalize by degree
        out = out / degree.unsqueeze(-1)
        
        return out


# KAN

class MinimalKAN(nn.Module):
    """
    Absolute simplest KAN: just learns a few basis functions per input dimension
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Just 3 basis functions: x, x^2, x^3
        # Coefficients: (in_features, out_features, 3)
        self.coeffs = nn.Parameter(torch.randn(in_features, out_features, 3) * 0.01)
        
        # Base linear (most of the work happens here initially)
        self.linear = nn.Linear(in_features, out_features)
        
    def forward(self, x):
        # Linear part
        out = self.linear(x)  # (batch, out_features)
        
        # Polynomial basis: [x, x^2, x^3]
        x = x.unsqueeze(-1)  # (batch, in_features, 1)
        basis = torch.stack([x, x**2, x**3], dim=-1).squeeze(2)  # (batch, in_features, 3)
        
        # Apply coefficients
        basis = basis.unsqueeze(2)  # (batch, in_features, 1, 3)
        coeffs = self.coeffs.unsqueeze(0)  # (1, in_features, out_features, 3)
        
        kan_out = (basis * coeffs).sum(dim=-1).sum(dim=1)  # (batch, out_features)
        
        # Combine: start with 90% linear, 10% KAN
        return out + 0.1 * kan_out


class KAN_GNN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.kan1 = MinimalKAN(in_features, hidden_features)
        self.kan2 = MinimalKAN(hidden_features, out_features)
        
    def forward(self, x, edge_index):
        # Layer 1
        x = self.kan1(x)
        x = F.relu(x)
        x = self.aggregate(x, edge_index)
        
        # Layer 2
        x = self.kan2(x)
        x = self.aggregate(x, edge_index)
        
        return F.log_softmax(x, dim=-1)
    
    def aggregate(self, x, edge_index):
        """Same as before"""
        source, target = edge_index
        num_nodes = x.size(0)
        degree = torch.zeros(num_nodes, device=x.device)
        degree.index_add_(0, target, torch.ones(source.size(0), device=x.device))
        degree = torch.clamp(degree, min=1.0)
        
        out = torch.zeros_like(x)
        out.index_add_(0, target, x[source])
        out = out / degree.unsqueeze(-1)
        return out

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

# ============================================================================
# Main
# ============================================================================

def main():
    # Load data
    dataset = Planetoid(root='./data', name='Cora', transform=NormalizeFeatures())
    data = dataset[0]
    
    print(f"Dataset: {dataset.name}")
    print(f"Nodes: {data.num_nodes}, Edges: {data.num_edges}")
    print(f"Features: {dataset.num_features}, Classes: {dataset.num_classes}\n")
    
    # Model
    model = KAN_GNN(
        in_features=dataset.num_features,
        hidden_features=64,
        out_features=dataset.num_classes
    )
    
    print(f"Parameters: {sum(p.numel() for p in model.parameters())}\n")
    
    # Train
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    
    print("Training SimpleGNN...")
    for epoch in range(1, 201):
        loss = train(model, data, optimizer)
        
        if epoch % 20 == 0:
            train_acc, val_acc, test_acc = test(model, data)
            print(f'Epoch {epoch:03d}: Loss={loss:.4f}, Train={train_acc:.4f}, Val={val_acc:.4f}, Test={test_acc:.4f}')
    
    # Final test
    train_acc, val_acc, test_acc = test(model, data)
    print(f'\nFinal: Train={train_acc:.4f}, Val={val_acc:.4f}, Test={test_acc:.4f}')

if __name__ == "__main__":
    main()