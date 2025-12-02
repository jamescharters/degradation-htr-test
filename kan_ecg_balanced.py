import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import ast
import os
import wfdb
from sklearn.metrics import roc_auc_score, classification_report
from collections import Counter


# ============ IMPROVED PTB-XL LOADER ============
def load_ptbxl(data_path, sampling_rate=100, min_samples_per_class=100):
    """Load PTB-XL with better preprocessing"""
    
    Y = pd.read_csv(os.path.join(data_path, 'ptbxl_database.csv'), index_col='ecg_id')
    Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))
    
    scp = pd.read_csv(os.path.join(data_path, 'scp_statements.csv'), index_col=0)
    scp = scp[scp.diagnostic == 1]
    
    def get_superclass(scp_codes):
        for code, conf in scp_codes.items():
            if conf >= 50 and code in scp.index:
                dc = scp.loc[code].diagnostic_class
                if pd.notna(dc):
                    return dc
        return None
    
    Y['superclass'] = Y.scp_codes.apply(get_superclass)
    Y = Y[Y.superclass.notna()]
    
    classes = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
    class_to_idx = {c: i for i, c in enumerate(classes)}
    Y['label'] = Y.superclass.map(class_to_idx)
    Y = Y[Y.label.notna()]
    
    # Check class distribution
    print(f"Original class distribution: {Y['superclass'].value_counts().to_dict()}")
    
    folder = 'records100' if sampling_rate == 100 else 'records500'
    
    X, labels, valid_idx = [], [], []
    for idx, row in Y.iterrows():
        try:
            fpath = os.path.join(data_path, row.filename_lr if sampling_rate == 100 else row.filename_hr)
            record = wfdb.rdrecord(fpath.replace('.dat', '').replace('.hea', ''))
            sig = record.p_signal
            
            # Use lead II (index 1) - standard clinical lead
            lead_ii = sig[:, 1]
            
            # Handle NaN/Inf
            if np.isnan(lead_ii).any() or np.isinf(lead_ii).any():
                continue
            
            X.append(lead_ii)
            labels.append(int(row.label))
            valid_idx.append(idx)
        except:
            continue
    
    X = np.array(X)
    y = np.array(labels)
    
    print(f"Loaded {len(X)} records")
    print(f"Class counts: {Counter(y)}")
    
    return X, y, classes


def balance_dataset(X, y, max_per_class=1000, min_per_class=100):
    """Balance by undersampling majority and keeping minority"""
    classes = np.unique(y)
    X_bal, y_bal = [], []
    
    for c in classes:
        mask = y == c
        X_c, y_c = X[mask], y[mask]
        n = len(X_c)
        
        if n < min_per_class:
            # Oversample minority class
            indices = np.random.choice(n, min_per_class, replace=True)
        elif n > max_per_class:
            # Undersample majority class
            indices = np.random.choice(n, max_per_class, replace=False)
        else:
            indices = np.arange(n)
        
        X_bal.append(X_c[indices])
        y_bal.append(y_c[indices])
    
    X_bal = np.concatenate(X_bal)
    y_bal = np.concatenate(y_bal)
    
    # Shuffle
    perm = np.random.permutation(len(X_bal))
    return X_bal[perm], y_bal[perm]


# ============ DEEPER BAYESIAN KAN ============
class BayesianKAN_Deep(nn.Module):
    """Deeper model for real data"""
    
    def __init__(self, seq_len=1000, hidden1=64, hidden2=32, out_dim=5, n_basis=16, min_std=0.1):
        super().__init__()
        self.min_std = min_std
        self.pool_size = seq_len // 100
        in_dim = 100
        
        # Layer 1
        self.c1_mu = nn.Parameter(torch.randn(hidden1, in_dim, n_basis) * 0.02)
        self.c1_logstd = nn.Parameter(torch.full((hidden1, in_dim, n_basis), -3.0))
        
        # Layer 2
        self.c2_mu = nn.Parameter(torch.randn(hidden2, hidden1, n_basis) * 0.02)
        self.c2_logstd = nn.Parameter(torch.full((hidden2, hidden1, n_basis), -3.0))
        
        # Layer 3 (output)
        self.c3_mu = nn.Parameter(torch.randn(out_dim, hidden2, n_basis) * 0.02)
        self.c3_logstd = nn.Parameter(torch.full((out_dim, hidden2, n_basis), -3.0))
        
        self.register_buffer('centers', torch.linspace(-3, 3, n_basis))
        
    def basis(self, x):
        return torch.exp(-0.5 * (x.unsqueeze(-1) - self.centers)**2 / 0.36)
    
    def kan_layer(self, x, c_mu, c_logstd, sample=False):
        B = self.basis(x)
        if sample:
            std = torch.clamp(torch.exp(c_logstd), min=self.min_std)
            c = c_mu + torch.randn_like(c_mu) * std
        else:
            c = c_mu
        return torch.einsum('bin,oin->bo', B, c)
    
    def forward(self, x, sample=False):
        # Downsample
        x = x.unfold(1, self.pool_size, self.pool_size).mean(-1)
        
        # Robust normalization
        x = (x - x.mean(1, keepdim=True)) / (x.std(1, keepdim=True) + 1e-6)
        x = torch.clamp(x, -5, 5)  # Clip outliers
        
        # Layer 1
        h = self.kan_layer(x, self.c1_mu, self.c1_logstd, sample)
        h = F.leaky_relu(h, 0.1)
        h = (h - h.mean(1, keepdim=True)) / (h.std(1, keepdim=True) + 1e-6)
        
        # Layer 2
        h = self.kan_layer(h, self.c2_mu, self.c2_logstd, sample)
        h = F.leaky_relu(h, 0.1)
        h = (h - h.mean(1, keepdim=True)) / (h.std(1, keepdim=True) + 1e-6)
        
        # Output
        return self.kan_layer(h, self.c3_mu, self.c3_logstd, sample)


def get_uncertainty(model, X, n_samples=50):
    model.eval()
    with torch.no_grad():
        logits = torch.stack([model(X, sample=True) for _ in range(n_samples)])
        probs = torch.softmax(logits, -1)
        mean_probs = probs.mean(0)
        
        preds = logits.argmax(-1)
        mode_pred = preds.mode(0).values
        
    return {
        'predictions': mean_probs.argmax(-1),
        'confidence': mean_probs.max(-1).values,
        'entropy': -(mean_probs * torch.log(mean_probs + 1e-8)).sum(-1),
        'disagreement': (preds != mode_pred).float().mean(0),
        'logit_std': logits.std(0).mean(-1),
    }


def create_ood_data(n=200, seq_len=1000):
    return torch.cat([
        torch.randn(n//4, seq_len) * 3,
        torch.zeros(n//4, seq_len) + torch.randn(n//4, seq_len) * 0.01,
        torch.sin(torch.linspace(0, 200, seq_len)).unsqueeze(0).repeat(n//4, 1) * 2,
        torch.randn(n//4, seq_len).cumsum(1) * 0.1,  # Random walk
    ])


if __name__ == '__main__':
    print("=" * 60)
    print("BAYESIAN KAN - REAL PTB-XL")
    print("=" * 60)
    
    # Load data
    DATA_PATH = './data/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/'
    X, y, classes = load_ptbxl(DATA_PATH)
    
    # Balance dataset
    print("\nBalancing dataset...")
    X, y = balance_dataset(X, y, max_per_class=800, min_per_class=200)
    print(f"Balanced: {Counter(y)}")
    
    # Convert to tensors
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)
    
    # Split
    n = len(X)
    perm = torch.randperm(n)
    n_train, n_val = int(0.7 * n), int(0.15 * n)
    
    X_train, y_train = X[perm[:n_train]], y[perm[:n_train]]
    X_val, y_val = X[perm[n_train:n_train+n_val]], y[perm[n_train:n_train+n_val]]
    X_test, y_test = X[perm[n_train+n_val:]], y[perm[n_train+n_val:]]
    
    print(f"\nTrain: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Model
    model = BayesianKAN_Deep(seq_len=X.shape[1], hidden1=64, hidden2=32, n_basis=16, min_std=0.1)
    
    # Class weights
    counts = torch.bincount(y_train)
    weights = 1.0 / counts.float()
    weights = weights / weights.sum() * len(classes)
    print(f"Class weights: {weights.tolist()}")
    
    # Optimizer with weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    
    # Train
    print("\nTraining...")
    best_val_acc = 0
    patience = 0
    
    for epoch in range(200):
        model.train()
        perm = torch.randperm(len(X_train))
        total_loss = 0
        
        for i in range(0, len(X_train), 64):
            idx = perm[i:i+64]
            
            # Sample multiple times for training
            logits = torch.stack([model(X_train[idx], sample=True) for _ in range(3)]).mean(0)
            loss = F.cross_entropy(logits, y_train[idx], weight=weights)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        
        scheduler.step()
        
        # Validate
        if epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                val_logits = model(X_val, sample=False)
                val_pred = val_logits.argmax(1)
                val_acc = (val_pred == y_val).float().mean().item()
                pred_dist = torch.bincount(val_pred, minlength=5).tolist()
            
            print(f"Epoch {epoch}: loss={total_loss:.2f}, val_acc={val_acc:.1%}, dist={pred_dist}")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = model.state_dict().copy()
                patience = 0
            else:
                patience += 1
            
            if patience > 10:
                print("Early stopping")
                break
    
    model.load_state_dict(best_state)
    
    # Test
    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    
    scores = get_uncertainty(model, X_test)
    test_acc = (scores['predictions'] == y_test).float().mean().item()
    
    print(f"\nTest Accuracy: {test_acc:.1%}")
    print(f"Predictions: {torch.bincount(scores['predictions'], minlength=5).tolist()}")
    print(f"True labels: {torch.bincount(y_test, minlength=5).tolist()}")
    
    print("\nPer-class:")
    for c, name in enumerate(classes):
        mask = y_test == c
        if mask.sum() > 0:
            acc = (scores['predictions'][mask] == c).float().mean().item()
            print(f"  {name}: {acc:.1%} ({mask.sum().item()} samples)")
    
    # OOD
    print("\n" + "=" * 60)
    print("OOD DETECTION")
    print("=" * 60)
    
    X_ood = create_ood_data(200, X.shape[1])
    scores_ood = get_uncertainty(model, X_ood)
    
    print(f"\n{'Metric':<15} {'ID':>10} {'OOD':>10} {'AUROC':>10}")
    print("-" * 48)
    
    for metric in ['entropy', 'logit_std', 'disagreement']:
        id_s = scores[metric].numpy()
        ood_s = scores_ood[metric].numpy()
        labels = np.concatenate([np.zeros(len(id_s)), np.ones(len(ood_s))])
        vals = np.concatenate([id_s, ood_s])
        try:
            auroc = roc_auc_score(labels, vals)
        except:
            auroc = 0.5
        print(f"{metric:<15} {id_s.mean():>10.3f} {ood_s.mean():>10.3f} {auroc:>10.3f}")
    
    print("\nDONE")