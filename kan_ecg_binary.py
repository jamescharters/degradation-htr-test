import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import ast
import os
import wfdb
from sklearn.metrics import roc_auc_score, classification_report, roc_curve
from collections import Counter
import matplotlib.pyplot as plt


def load_ptbxl_binary(data_path, sampling_rate=100):
    """Load PTB-XL as binary: Normal vs Abnormal"""
    
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
    
    # Binary: 0 = Normal, 1 = Abnormal
    Y['label'] = (Y['superclass'] != 'NORM').astype(int)
    
    print(f"Class distribution: Normal={len(Y[Y.label==0])}, Abnormal={len(Y[Y.label==1])}")
    
    folder = 'records100' if sampling_rate == 100 else 'records500'
    
    X, labels = [], []
    for idx, row in Y.iterrows():
        try:
            fpath = os.path.join(data_path, row.filename_lr if sampling_rate == 100 else row.filename_hr)
            record = wfdb.rdrecord(fpath.replace('.dat', '').replace('.hea', ''))
            sig = record.p_signal[:, 1]  # Lead II
            
            if np.isnan(sig).any() or np.isinf(sig).any():
                continue
                
            X.append(sig)
            labels.append(int(row.label))
        except:
            continue
    
    return np.array(X), np.array(labels)


def balance_binary(X, y, max_per_class=3000):
    """Balance binary dataset"""
    X0, X1 = X[y == 0], X[y == 1]
    
    n = min(len(X0), len(X1), max_per_class)
    
    idx0 = np.random.choice(len(X0), n, replace=False)
    idx1 = np.random.choice(len(X1), n, replace=False)
    
    X_bal = np.concatenate([X0[idx0], X1[idx1]])
    y_bal = np.array([0] * n + [1] * n)
    
    perm = np.random.permutation(len(X_bal))
    return X_bal[perm], y_bal[perm]


class BayesianKAN_Binary(nn.Module):
    def __init__(self, seq_len=1000, hidden=48, n_basis=12, min_std=0.12):
        super().__init__()
        self.min_std = min_std
        self.pool_size = seq_len // 100
        in_dim = 100
        
        self.c1_mu = nn.Parameter(torch.randn(hidden, in_dim, n_basis) * 0.03)
        self.c1_logstd = nn.Parameter(torch.full((hidden, in_dim, n_basis), -2.5))
        
        self.c2_mu = nn.Parameter(torch.randn(2, hidden, n_basis) * 0.03)
        self.c2_logstd = nn.Parameter(torch.full((2, hidden, n_basis), -2.5))
        
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
        x = x.unfold(1, self.pool_size, self.pool_size).mean(-1)
        x = (x - x.mean(1, keepdim=True)) / (x.std(1, keepdim=True) + 1e-6)
        x = torch.clamp(x, -5, 5)
        
        h = self.kan_layer(x, self.c1_mu, self.c1_logstd, sample)
        h = torch.tanh(h)
        h = (h - h.mean(1, keepdim=True)) / (h.std(1, keepdim=True) + 1e-6)
        
        return self.kan_layer(h, self.c2_mu, self.c2_logstd, sample)


class MLP_Baseline(nn.Module):
    """MLP baseline for comparison"""
    def __init__(self, seq_len=1000, hidden=48):
        super().__init__()
        self.pool_size = seq_len // 100
        self.fc1 = nn.Linear(100, hidden)
        self.fc2 = nn.Linear(hidden, 2)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x, sample=False):
        x = x.unfold(1, self.pool_size, self.pool_size).mean(-1)
        x = (x - x.mean(1, keepdim=True)) / (x.std(1, keepdim=True) + 1e-6)
        
        h = torch.tanh(self.fc1(x))
        h = self.dropout(h) if sample else h
        return self.fc2(h)


class MCDropoutMLP(nn.Module):
    """MC Dropout MLP for uncertainty baseline"""
    def __init__(self, seq_len=1000, hidden=48, dropout=0.3):
        super().__init__()
        self.pool_size = seq_len // 100
        self.fc1 = nn.Linear(100, hidden)
        self.fc2 = nn.Linear(hidden, 2)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, sample=False):
        x = x.unfold(1, self.pool_size, self.pool_size).mean(-1)
        x = (x - x.mean(1, keepdim=True)) / (x.std(1, keepdim=True) + 1e-6)
        
        h = torch.tanh(self.fc1(x))
        if sample:
            h = self.dropout(h)
        return self.fc2(h)


def get_uncertainty(model, X, n_samples=50):
    model.train()  # Keep dropout active for MC sampling
    with torch.no_grad():
        logits = torch.stack([model(X, sample=True) for _ in range(n_samples)])
        probs = torch.softmax(logits, -1)
        mean_probs = probs.mean(0)
        
        preds = logits.argmax(-1)
        mode_pred = preds.mode(0).values
        
    model.eval()
    return {
        'predictions': mean_probs.argmax(-1),
        'prob_abnormal': mean_probs[:, 1],
        'confidence': mean_probs.max(-1).values,
        'entropy': -(mean_probs * torch.log(mean_probs + 1e-8)).sum(-1),
        'disagreement': (preds != mode_pred).float().mean(0),
        'logit_std': logits.std(0).mean(-1),
    }


def train_model(model, X_train, y_train, X_val, y_val, epochs=100, lr=0.01):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    best_val_acc = 0
    best_state = None
    
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(X_train))
        
        for i in range(0, len(X_train), 64):
            idx = perm[i:i+64]
            logits = model(X_train[idx], sample=True)
            loss = F.cross_entropy(logits, y_train[idx])
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        scheduler.step()
        
        # Validate
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val).argmax(1)
            val_acc = (val_pred == y_val).float().mean().item()
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict().copy()
        
        if epoch % 20 == 0:
            print(f"  Epoch {epoch}: val_acc={val_acc:.1%}")
    
    model.load_state_dict(best_state)
    return model, best_val_acc


def evaluate(model, X_test, y_test, name="Model"):
    scores = get_uncertainty(model, X_test)
    
    acc = (scores['predictions'] == y_test).float().mean().item()
    
    # ROC-AUC for classification
    try:
        class_auroc = roc_auc_score(y_test.numpy(), scores['prob_abnormal'].numpy())
    except:
        class_auroc = 0.5
    
    return {
        'name': name,
        'accuracy': acc,
        'auroc': class_auroc,
        'scores': scores
    }


def evaluate_ood(model, X_id, X_ood, name="Model"):
    scores_id = get_uncertainty(model, X_id)
    scores_ood = get_uncertainty(model, X_ood)
    
    results = {}
    for metric in ['entropy', 'logit_std', 'disagreement']:
        id_s = scores_id[metric].numpy()
        ood_s = scores_ood[metric].numpy()
        labels = np.concatenate([np.zeros(len(id_s)), np.ones(len(ood_s))])
        vals = np.concatenate([id_s, ood_s])
        try:
            results[metric] = roc_auc_score(labels, vals)
        except:
            results[metric] = 0.5
    
    return results


def create_ood_data(n=300, seq_len=1000):
    """Various OOD signals"""
    return torch.cat([
        torch.randn(n//3, seq_len) * 3,  # High noise
        torch.zeros(n//3, seq_len) + torch.randn(n//3, seq_len) * 0.02,  # Flat
        torch.sin(torch.linspace(0, 100, seq_len)).unsqueeze(0).repeat(n//3, 1) * 2,  # Sine
    ])


if __name__ == '__main__':
    print("=" * 70)
    print("BAYESIAN KAN vs BASELINES - BINARY CLASSIFICATION")
    print("=" * 70)
    
    # Load data
    DATA_PATH = './data/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/'
    
    print("\nLoading PTB-XL (binary)...")
    X, y = load_ptbxl_binary(DATA_PATH)
    
    print("\nBalancing...")
    X, y = balance_binary(X, y, max_per_class=2500)
    print(f"Balanced: {Counter(y)}")
    
    # Convert
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)
    
    # Split
    n = len(X)
    perm = torch.randperm(n)
    n_train, n_val = int(0.7 * n), int(0.15 * n)
    
    X_train, y_train = X[perm[:n_train]], y[perm[:n_train]]
    X_val, y_val = X[perm[n_train:n_train+n_val]], y[perm[n_train:n_train+n_val]]
    X_test, y_test = X[perm[n_train+n_val:]], y[perm[n_train+n_val:]]
    
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # OOD data
    X_ood = create_ood_data(300, X.shape[1])
    
    # Results storage
    all_results = []
    ood_results = []
    
    # ============ Model 1: Bayesian KAN ============
    print("\n" + "=" * 50)
    print("Training: Bayesian KAN")
    print("=" * 50)
    
    bkan = BayesianKAN_Binary(seq_len=X.shape[1], hidden=48, n_basis=12, min_std=0.12)
    bkan, _ = train_model(bkan, X_train, y_train, X_val, y_val, epochs=100, lr=0.01)
    
    res = evaluate(bkan, X_test, y_test, "Bayesian KAN")
    all_results.append(res)
    ood = evaluate_ood(bkan, X_test, X_ood, "Bayesian KAN")
    ood_results.append(("Bayesian KAN", ood))
    
    # ============ Model 2: MC Dropout MLP ============
    print("\n" + "=" * 50)
    print("Training: MC Dropout MLP")
    print("=" * 50)
    
    mcdrop = MCDropoutMLP(seq_len=X.shape[1], hidden=48, dropout=0.3)
    mcdrop, _ = train_model(mcdrop, X_train, y_train, X_val, y_val, epochs=100, lr=0.01)
    
    res = evaluate(mcdrop, X_test, y_test, "MC Dropout")
    all_results.append(res)
    ood = evaluate_ood(mcdrop, X_test, X_ood, "MC Dropout")
    ood_results.append(("MC Dropout", ood))
    
    # ============ Model 3: Standard MLP ============
    print("\n" + "=" * 50)
    print("Training: Standard MLP")
    print("=" * 50)
    
    mlp = MLP_Baseline(seq_len=X.shape[1], hidden=48)
    mlp, _ = train_model(mlp, X_train, y_train, X_val, y_val, epochs=100, lr=0.01)
    
    res = evaluate(mlp, X_test, y_test, "Standard MLP")
    all_results.append(res)
    
    # ============ Results ============
    print("\n" + "=" * 70)
    print("CLASSIFICATION RESULTS")
    print("=" * 70)
    print(f"\n{'Model':<20} {'Accuracy':>12} {'AUROC':>12}")
    print("-" * 46)
    for r in all_results:
        print(f"{r['name']:<20} {r['accuracy']:>12.1%} {r['auroc']:>12.3f}")
    
    print("\n" + "=" * 70)
    print("OOD DETECTION RESULTS (AUROC)")
    print("=" * 70)
    print(f"\n{'Model':<20} {'Entropy':>12} {'Logit Std':>12} {'Disagree':>12}")
    print("-" * 58)
    for name, ood in ood_results:
        print(f"{name:<20} {ood['entropy']:>12.3f} {ood['logit_std']:>12.3f} {ood['disagreement']:>12.3f}")
    
    # ============ Summary ============
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    bkan_acc = all_results[0]['accuracy']
    mcdrop_acc = all_results[1]['accuracy']
    mlp_acc = all_results[2]['accuracy']
    
    bkan_ood = max(ood_results[0][1].values())
    mcdrop_ood = max(ood_results[1][1].values())
    
    print(f"\nClassification:")
    print(f"  Bayesian KAN: {bkan_acc:.1%}")
    print(f"  MC Dropout:   {mcdrop_acc:.1%}")
    print(f"  Standard MLP: {mlp_acc:.1%}")
    
    print(f"\nBest OOD AUROC:")
    print(f"  Bayesian KAN: {bkan_ood:.3f}")
    print(f"  MC Dropout:   {mcdrop_ood:.3f}")
    
    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)