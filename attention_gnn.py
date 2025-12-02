import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

class AttentionGNN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, dropout=0.6):
        super().__init__()
        self.lin1 = nn.Linear(in_features, hidden_features)
        self.lin2 = nn.Linear(hidden_features, out_features)
        
        self.att1 = nn.Linear(hidden_features * 2, 1)
        self.att2 = nn.Linear(out_features * 2, 1)
        
        self.dropout = dropout
        
    def forward(self, x, edge_index):
        # Layer 1
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin1(x)
        x = F.elu(x)  # ELU often works better than ReLU for attention
        x = self.aggregate_with_attention(x, edge_index, self.att1)
        
        # Layer 2
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
        
        # Dropout on attention weights
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

def main():
    dataset = Planetoid(root='./data', name='Cora', transform=NormalizeFeatures())
    data = dataset[0]
    
    print(f"Dataset: {dataset.name}")
    print(f"Nodes: {data.num_nodes}, Edges: {data.num_edges}")
    print(f"Features: {dataset.num_features}, Classes: {dataset.num_classes}\n")
    
    model = AttentionGNN(
        in_features=dataset.num_features,
        hidden_features=64,
        out_features=dataset.num_classes
    )
    
    print(f"Parameters: {sum(p.numel() for p in model.parameters())}\n")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    
    print("Training AttentionGNN...")
    for epoch in range(1, 201):
        loss = train(model, data, optimizer)
        
        if epoch % 20 == 0:
            train_acc, val_acc, test_acc = test(model, data)
            print(f'Epoch {epoch:03d}: Loss={loss:.4f}, Train={train_acc:.4f}, Val={val_acc:.4f}, Test={test_acc:.4f}')
    
    train_acc, val_acc, test_acc = test(model, data)
    print(f'\nFinal: Train={train_acc:.4f}, Val={val_acc:.4f}, Test={test_acc:.4f}')

if __name__ == "__main__":
    main()