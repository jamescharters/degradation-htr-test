import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import ast
import os
import wfdb
from sklearn.metrics import roc_auc_score
from collections import Counter


def load_ptbxl_12lead(data_path, sampling_rate=100):
    """Load all 12 leads"""
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
    Y['label'] = (Y['superclass'] != 'NORM').astype(int)
    
    folder = 'records100' if sampling_rate == 100 else 'records500'
    
    X, labels = [], []
    for idx, row in Y.iterrows():
        try:
            fpath = os.path.join(data_path, row.filename_lr if sampling_rate == 100 else row.filename_hr)
            record = wfdb.rdrecord(fpath.replace('.dat', '').replace('.hea', ''))
            sig = record.p_signal  # All 12 leads: (1000, 12)
            
            if np.isnan(sig).any() or np.isinf(sig).any():
                continue
            
            X.append(sig)
            labels.append(int(row.label))
        except:
            continue
    
    return np.array(X), np.array(labels)


def balance_binary(X, y, max_per_class=2500):
    X0, X1 = X[y == 0], X[y == 1]
    n = min(len(X0), len(X1), max_per_class)
    idx0 = np.random.choice(len(X0), n, replace=False)
    idx1 = np.random.choice(len(X1), n, replace=False)
    X_bal = np.concatenate([X0[idx0], X1[idx1]])
    y_bal = np.array([0] * n + [1] * n)
    perm = np.random.permutation(len(X_bal))
    return X_bal[perm], y_bal[perm]


class ConvBayesianKAN(nn.Module):
    """Conv frontend + Bayesian KAN backend"""
    
    def __init__(self, n_leads=12, hidden=32, n_basis=10, min_std=0.15):
        super().__init__()
        self.min_std = min_std
        self.n_basis = n_basis
        
        # Conv frontend to extract features from 12 leads
        self.conv1 = nn.Conv1d(n_leads, 32, kernel_size=7, stride=2, padding=3)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=3, stride=2, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(16)
        
        # Bayesian KAN layers
        in_dim = 64 * 16  # After conv + pool
        
        self.c1_mu = nn.Parameter(torch.randn(hidden, in_dim, n_basis) * 0.01)
        self.c1_logstd = nn.Parameter(torch.full((hidden, in_dim, n_basis), -3.0))
        
        self.c2_mu = nn.Parameter(torch.randn(2, hidden, n_basis) * 0.01)
        self.c2_logstd = nn.Parameter(torch.full((2, hidden, n_basis), -3.0))
        
        self.register_buffer('centers', torch.linspace(-2, 2, n_basis))
        
    def basis(self, x):
        return torch.exp(-0.5 * (x.unsqueeze(-1) - self.centers)**2 / 0.25)
    
    def kan_layer(self, x, c_mu, c_logstd, sample=False):
        B = self.basis(x)
        if sample:
            std = torch.clamp(torch.exp(c_logstd), min=self.min_std)
            c = c_mu + torch.randn_like(c_mu) * std
        else:
            c = c_mu
        return torch.einsum('bin,oin->bo', B, c)
    
    def forward(self, x, sample=False):
        # x: [batch, seq_len, 12] -> [batch, 12, seq_len]
        x = x.transpose(1, 2)
        
        # Conv frontend
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        
        # Flatten
        x = x.flatten(1)
        
        # Normalize
        x = (x - x.mean(1, keepdim=True)) / (x.std(1, keepdim=True) + 1e-6)
        
        # Bayesian KAN
        h = self.kan_layer(x, self.c1_mu, self.c1_logstd, sample)
        h = torch.tanh(h)
        h = (h - h.mean(1, keepdim=True)) / (h.std(1, keepdim=True) + 1e-6)
        
        return self.kan_layer(h, self.c2_mu, self.c2_logstd, sample)


class ConvMLP(nn.Module):
    """Conv frontend + MLP backend (baseline)"""
    
    def __init__(self, n_leads=12, hidden=32):
        super().__init__()
        
        self.conv1 = nn.Conv1d(n_leads, 32, kernel_size=7, stride=2, padding=3)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=3, stride=2, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(16)
        
        self.fc1 = nn.Linear(64 * 16, hidden)
        self.fc2 = nn.Linear(hidden, 2)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x, sample=False):
        x = x.transpose(1, 2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = x.flatten(1)
        
        x = F.relu(self.fc1(x))
        if sample:
            x = self.dropout(x)
        return self.fc2(x)


def get_uncertainty(model, X, n_samples=50):
    model.train()
    with torch.no_grad():
        logits = torch.stack([model(X, sample=True) for _ in range(n_samples)])
        probs = torch.softmax(logits, -1)
        mean_probs = probs.mean(0)
        preds = logits.argmax(-1)
        mode_pred = preds.mode(0).values
    model.eval()
    
    return {
        'predictions': mean_probs.argmax(-1),
        'prob_positive': mean_probs[:, 1],
        'confidence': mean_probs.max(-1).values,
        'entropy': -(mean_probs * torch.log(mean_probs + 1e-8)).sum(-1),
        'disagreement': (preds != mode_pred).float().mean(0),
        'logit_std': logits.std(0).mean(-1),
    }


def train_model(model, X_train, y_train, X_val, y_val, epochs=80, lr=0.001, batch_size=32):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    best_val_auc = 0
    best_state = None
    patience = 0
    
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(X_train))
        
        for i in range(0, len(X_train), batch_size):
            idx = perm[i:i+batch_size]
            logits = model(X_train[idx], sample=True)
            loss = F.cross_entropy(logits, y_train[idx])
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        scheduler.step()
        
        # Validate with AUROC
        model.eval()
        with torch.no_grad():
            val_probs = torch.softmax(model(X_val), -1)[:, 1]
            val_pred = (val_probs > 0.5).long()
            val_acc = (val_pred == y_val).float().mean().item()
            try:
                val_auc = roc_auc_score(y_val.numpy(), val_probs.numpy())
            except:
                val_auc = 0.5
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = model.state_dict().copy()
            patience = 0
        else:
            patience += 1
        
        if epoch % 10 == 0:
            print(f"  Epoch {epoch}: acc={val_acc:.1%}, AUC={val_auc:.3f}")
        
        if patience > 15:
            print("  Early stopping")
            break
    
    model.load_state_dict(best_state)
    return model, best_val_auc


def create_ood_data(n=200, seq_len=1000, n_leads=12):
    """OOD data for 12-lead"""
    ood = []
    
    # High noise
    ood.append(torch.randn(n//4, seq_len, n_leads) * 3)
    
    # Flat
    ood.append(torch.randn(n//4, seq_len, n_leads) * 0.01)
    
    # Sine waves
    t = torch.linspace(0, 50, seq_len).unsqueeze(0).unsqueeze(2)
    sine = torch.sin(t).repeat(n//4, 1, n_leads) + torch.randn(n//4, seq_len, n_leads) * 0.1
    ood.append(sine)
    
    # Random spikes
    spikes = torch.randn(n//4, seq_len, n_leads) * 0.3
    for i in range(n//4):
        for j in range(0, seq_len, 50):
            spikes[i, j:j+3, :] = 5.0
    ood.append(spikes)
    
    return torch.cat(ood)


if __name__ == '__main__':
    print("=" * 70)
    print("12-LEAD ECG: Conv-Bayesian-KAN vs Conv-MLP")
    print("=" * 70)
    
    DATA_PATH = './data/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/'
    
    print("\nLoading 12-lead PTB-XL...")
    X, y = load_ptbxl_12lead(DATA_PATH)
    print(f"Loaded: {X.shape}, Classes: {Counter(y)}")
    
    print("\nBalancing...")
    X, y = balance_binary(X, y, max_per_class=2000)
    print(f"Balanced: {Counter(y)}")
    
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
    
    X_ood = create_ood_data(200, X.shape[1], X.shape[2])
    
    results = []
    
    # ========== Conv-Bayesian-KAN ==========
    print("\n" + "=" * 50)
    print("Training: Conv-Bayesian-KAN")
    print("=" * 50)
    
    model1 = ConvBayesianKAN(n_leads=12, hidden=32, n_basis=10, min_std=0.15)
    model1, val_auc1 = train_model(model1, X_train, y_train, X_val, y_val)
    
    scores1 = get_uncertainty(model1, X_test)
    test_acc1 = (scores1['predictions'] == y_test).float().mean().item()
    test_auc1 = roc_auc_score(y_test.numpy(), scores1['prob_positive'].numpy())
    
    scores1_ood = get_uncertainty(model1, X_ood)
    
    results.append({
        'name': 'Conv-Bayesian-KAN',
        'acc': test_acc1,
        'auc': test_auc1,
        'ood_entropy': roc_auc_score(
            np.concatenate([np.zeros(len(X_test)), np.ones(len(X_ood))]),
            np.concatenate([scores1['entropy'].numpy(), scores1_ood['entropy'].numpy()])
        ),
        'ood_logit_std': roc_auc_score(
            np.concatenate([np.zeros(len(X_test)), np.ones(len(X_ood))]),
            np.concatenate([scores1['logit_std'].numpy(), scores1_ood['logit_std'].numpy()])
        ),
    })
    
    # ========== Conv-MLP (MC Dropout) ==========
    print("\n" + "=" * 50)
    print("Training: Conv-MLP (MC Dropout)")
    print("=" * 50)
    
    model2 = ConvMLP(n_leads=12, hidden=32)
    model2, val_auc2 = train_model(model2, X_train, y_train, X_val, y_val)
    
    scores2 = get_uncertainty(model2, X_test)
    test_acc2 = (scores2['predictions'] == y_test).float().mean().item()
    test_auc2 = roc_auc_score(y_test.numpy(), scores2['prob_positive'].numpy())
    
    scores2_ood = get_uncertainty(model2, X_ood)
    
    results.append({
        'name': 'Conv-MLP (MC Dropout)',
        'acc': test_acc2,
        'auc': test_auc2,
        'ood_entropy': roc_auc_score(
            np.concatenate([np.zeros(len(X_test)), np.ones(len(X_ood))]),
            np.concatenate([scores2['entropy'].numpy(), scores2_ood['entropy'].numpy()])
        ),
        'ood_logit_std': roc_auc_score(
            np.concatenate([np.zeros(len(X_test)), np.ones(len(X_ood))]),
            np.concatenate([scores2['logit_std'].numpy(), scores2_ood['logit_std'].numpy()])
        ),
    })
    
    # ========== Results ==========
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    
    print(f"\n{'Model':<25} {'Accuracy':>10} {'AUROC':>10} {'OOD-Ent':>10} {'OOD-Std':>10}")
    print("-" * 67)
    for r in results:
        print(f"{r['name']:<25} {r['acc']:>10.1%} {r['auc']:>10.3f} {r['ood_entropy']:>10.3f} {r['ood_logit_std']:>10.3f}")
    
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    
    if results[0]['auc'] >= results[1]['auc'] - 0.02:
        print("✓ Conv-Bayesian-KAN achieves competitive classification performance")
    else:
        print("✗ Conv-Bayesian-KAN underperforms on classification")
    
    if results[0]['ood_entropy'] > 0.6 or results[0]['ood_logit_std'] > 0.6:
        print("✓ Conv-Bayesian-KAN provides useful OOD detection")
    else:
        print("✗ Conv-Bayesian-KAN OOD detection needs improvement")
    
    print("\nDONE")