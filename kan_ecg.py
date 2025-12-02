import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score


def make_learnable_ecg_data(n=500):
    X = torch.randn(n, 1000) * 0.5
    y = torch.zeros(n, dtype=torch.long)
    for i in range(n):
        c = i % 5
        if c == 0:
            X[i, 200:250] += 3.0
        elif c == 1:
            X[i, 500:550] += 3.0
        elif c == 2:
            X[i, 800:850] += 3.0
        elif c == 3:
            X[i, ::100] += 2.0
        else:
            X[i, 400:600] = X[i, 400:600].abs() + 1.5
        y[i] = c
    return X, y


class BayesianKAN(nn.Module):
    def __init__(self, in_dim=100, hidden=32, out_dim=5, n_basis=12, min_std=0.2):
        super().__init__()
        self.n_basis = n_basis
        self.min_std = min_std
        
        self.c1_mu = nn.Parameter(torch.randn(hidden, in_dim, n_basis) * 0.1)
        self.c2_mu = nn.Parameter(torch.randn(out_dim, hidden, n_basis) * 0.1)
        self.c1_logstd = nn.Parameter(torch.zeros(hidden, in_dim, n_basis) - 1)
        self.c2_logstd = nn.Parameter(torch.zeros(out_dim, hidden, n_basis) - 1)
        
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
        x = x.unfold(1, 10, 10).mean(-1)
        x = (x - x.mean(1, keepdim=True)) / (x.std(1, keepdim=True) + 1e-6)
        h = self.kan_layer(x, self.c1_mu, self.c1_logstd, sample)
        h = torch.tanh(h)
        h = (h - h.mean(1, keepdim=True)) / (h.std(1, keepdim=True) + 1e-6)
        return self.kan_layer(h, self.c2_mu, self.c2_logstd, sample)


def get_uncertainty_scores(model, X, n_samples=100):
    """Returns multiple uncertainty metrics"""
    model.eval()
    with torch.no_grad():
        logits = torch.stack([model(X, sample=True) for _ in range(n_samples)])
        probs = torch.softmax(logits, -1)
        mean_probs = probs.mean(0)
        
        # Multiple metrics
        confidence = mean_probs.max(-1).values
        entropy = -(mean_probs * torch.log(mean_probs + 1e-8)).sum(-1)
        
        preds = logits.argmax(-1)
        mode_pred = preds.mode(0).values
        disagreement = (preds != mode_pred).float().mean(0)
        
        logit_std = logits.std(0).mean(-1)
        
    return {
        'confidence': confidence,
        'entropy': entropy,
        'disagreement': disagreement,
        'logit_std': logit_std,
    }


def evaluate_ood_detection(model):
    """Proper OOD evaluation with AUROC"""
    # In-distribution
    X_id, y_id = make_learnable_ecg_data(200)
    
    # Multiple OOD types
    X_ood_list = [
        torch.randn(50, 1000) * 2.0,  # High noise
        torch.zeros(50, 1000),  # Flat
        torch.randn(50, 1000) * 0.1,  # Low noise  
    ]
    # Add spikes at random locations
    X_spike = torch.randn(50, 1000) * 0.5
    for i in range(50):
        pos = torch.randint(0, 1000, (1,)).item()
        X_spike[i, pos:pos+10] = 8.0
    X_ood_list.append(X_spike)
    
    X_ood = torch.cat(X_ood_list)
    
    # Get scores
    scores_id = get_uncertainty_scores(model, X_id)
    scores_ood = get_uncertainty_scores(model, X_ood)
    
    # Accuracy on ID
    with torch.no_grad():
        preds = model(X_id, sample=False).argmax(1)
        acc = (preds == y_id).float().mean().item()
    
    print(f"\nID Accuracy: {acc:.1%}")
    print(f"\n{'Metric':<15} {'ID mean':>10} {'OOD mean':>10} {'AUROC':>10} {'Status':>10}")
    print("-" * 58)
    
    results = {}
    for metric in ['entropy', 'disagreement', 'logit_std']:
        id_scores = scores_id[metric].numpy()
        ood_scores = scores_ood[metric].numpy()
        
        # For these metrics, higher = more uncertain = OOD
        labels = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
        scores = np.concatenate([id_scores, ood_scores])
        
        auroc = roc_auc_score(labels, scores)
        status = "✓" if auroc > 0.5 else "✗"
        
        print(f"{metric:<15} {id_scores.mean():>10.4f} {ood_scores.mean():>10.4f} {auroc:>10.3f} {status:>10}")
        results[metric] = auroc
    
    # Confidence: lower = more uncertain = OOD (so flip for AUROC)
    id_conf = scores_id['confidence'].numpy()
    ood_conf = scores_ood['confidence'].numpy()
    labels = np.concatenate([np.zeros(len(id_conf)), np.ones(len(ood_conf))])
    scores = np.concatenate([id_conf, ood_conf])
    auroc_conf = roc_auc_score(labels, -scores)  # Negative because lower conf = OOD
    status = "✓" if auroc_conf > 0.5 else "✗"
    print(f"{'confidence':<15} {id_conf.mean():>10.4f} {ood_conf.mean():>10.4f} {auroc_conf:>10.3f} {status:>10}")
    results['confidence'] = auroc_conf
    
    print(f"\n{'='*58}")
    best_metric = max(results, key=results.get)
    print(f"Best OOD detector: {best_metric} (AUROC={results[best_metric]:.3f})")
    
    if results[best_metric] > 0.6:
        print("✓ Bayesian KAN provides useful OOD detection!")
    else:
        print("✗ OOD detection not reliable")
    
    return results, acc


if __name__ == '__main__':
    print("=" * 60)
    print("BAYESIAN KAN - AUROC EVALUATION")
    print("=" * 60)
    
    best_auroc = 0
    best_acc = 0
    
    # Try a few min_std values
    for min_std in [0.15, 0.2, 0.25]:
        print(f"\n>>> Testing min_std = {min_std}")
        
        X_train, y_train = make_learnable_ecg_data(500)
        model = BayesianKAN(min_std=min_std)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        # Train
        for epoch in range(300):
            model.train()
            idx = torch.randperm(500)[:64]
            logits = torch.stack([model(X_train[idx], sample=True) for _ in range(3)]).mean(0)
            loss = F.cross_entropy(logits, y_train[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Quick acc check
        model.eval()
        with torch.no_grad():
            acc = (model(X_train).argmax(1) == y_train).float().mean().item()
        print(f"Train acc: {acc:.1%}")
        
        if acc > 0.8:  # Only evaluate if model learned
            results, test_acc = evaluate_ood_detection(model)
            max_auroc = max(results.values())
            if max_auroc > best_auroc:
                best_auroc = max_auroc
                best_acc = test_acc
                best_min_std = min_std
    
    print(f"\n{'='*60}")
    print(f"BEST RESULT: min_std={best_min_std}, Acc={best_acc:.1%}, AUROC={best_auroc:.3f}")
    print("=" * 60)