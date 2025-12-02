import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt


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


def visualize_synthetic_data(X, y, n_samples=2):
    """Visualize examples from each class"""
    fig, axes = plt.subplots(5, 1, figsize=(14, 10))
    class_names = [
        "Class 0: Early spike (200-250)",
        "Class 1: Mid spike (500-550)", 
        "Class 2: Late spike (800-850)",
        "Class 3: Regular peaks (every 100)",
        "Class 4: Rectified hump (400-600)"
    ]
    
    for class_idx in range(5):
        # Get samples from this class
        mask = y == class_idx
        samples = X[mask][:n_samples]
        
        ax = axes[class_idx]
        for i, sample in enumerate(samples):
            ax.plot(sample.numpy(), alpha=0.7, label=f'Sample {i+1}')
        
        ax.set_title(class_names[class_idx], fontsize=11, fontweight='bold')
        ax.set_ylabel('Amplitude')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
        
        if class_idx == 4:
            ax.set_xlabel('Time steps')
    
    plt.tight_layout()
    plt.savefig('synthetic_ecg_patterns.png', dpi=150, bbox_inches='tight')
    print("✓ Saved visualization to 'synthetic_ecg_patterns.png'")
    plt.show()


def visualize_ood_comparison(model, save_path='ood_comparison.png'):
    """Visualize ID vs OOD samples"""
    # Generate data
    X_id, y_id = make_learnable_ecg_data(10)
    
    X_ood_noise = torch.randn(3, 1000) * 2.0
    X_ood_flat = torch.zeros(3, 1000)
    X_ood_spike = torch.randn(3, 1000) * 0.5
    for i in range(3):
        pos = 300 + i * 200
        X_ood_spike[i, pos:pos+10] = 8.0
    
    # Get uncertainty scores
    scores_id = get_uncertainty_scores(model, X_id)
    
    fig, axes = plt.subplots(4, 3, figsize=(15, 10))
    
    # Row 1: In-distribution samples
    for i in range(3):
        ax = axes[0, i]
        ax.plot(X_id[i].numpy(), 'b-', linewidth=1)
        ent = scores_id['entropy'][i].item()
        conf = scores_id['confidence'][i].item()
        pred = model(X_id[i:i+1]).argmax().item()
        ax.set_title(f'ID (Class {y_id[i].item()})\nPred: {pred}, Ent: {ent:.2f}, Conf: {conf:.2f}', 
                     fontsize=9)
        ax.set_ylim(-4, 8)
        if i == 0:
            ax.set_ylabel('In-Distribution', fontweight='bold')
    
    # Row 2: High noise OOD
    scores_noise = get_uncertainty_scores(model, X_ood_noise)
    for i in range(3):
        ax = axes[1, i]
        ax.plot(X_ood_noise[i].numpy(), 'r-', linewidth=1, alpha=0.7)
        ent = scores_noise['entropy'][i].item()
        conf = scores_noise['confidence'][i].item()
        ax.set_title(f'High Noise\nEnt: {ent:.2f}, Conf: {conf:.2f}', fontsize=9)
        ax.set_ylim(-4, 8)
        if i == 0:
            ax.set_ylabel('OOD: Noise', fontweight='bold')
    
    # Row 3: Flat OOD
    scores_flat = get_uncertainty_scores(model, X_ood_flat)
    for i in range(3):
        ax = axes[2, i]
        ax.plot(X_ood_flat[i].numpy(), 'orange', linewidth=1)
        ent = scores_flat['entropy'][i].item()
        conf = scores_flat['confidence'][i].item()
        ax.set_title(f'Flat Signal\nEnt: {ent:.2f}, Conf: {conf:.2f}', fontsize=9)
        ax.set_ylim(-4, 8)
        if i == 0:
            ax.set_ylabel('OOD: Flat', fontweight='bold')
    
    # Row 4: Spike OOD
    scores_spike = get_uncertainty_scores(model, X_ood_spike)
    for i in range(3):
        ax = axes[3, i]
        ax.plot(X_ood_spike[i].numpy(), 'purple', linewidth=1)
        ent = scores_spike['entropy'][i].item()
        conf = scores_spike['confidence'][i].item()
        ax.set_title(f'Random Spike\nEnt: {ent:.2f}, Conf: {conf:.2f}', fontsize=9)
        ax.set_ylim(-4, 8)
        ax.set_xlabel('Time steps')
        if i == 0:
            ax.set_ylabel('OOD: Spike', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved OOD comparison to '{save_path}'")
    plt.show()


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
    print("BAYESIAN KAN - AUROC EVALUATION WITH VISUALIZATION")
    print("=" * 60)
    
    # First, visualize the synthetic data
    print("\nGenerating synthetic data visualization...")
    X_viz, y_viz = make_learnable_ecg_data(500)
    visualize_synthetic_data(X_viz, y_viz, n_samples=2)
    
    best_auroc = 0
    best_acc = 0
    best_model = None
    
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
                best_model = model
    
    print(f"\n{'='*60}")
    print(f"BEST RESULT: min_std={best_min_std}, Acc={best_acc:.1%}, AUROC={best_auroc:.3f}")
    print("=" * 60)
    
    # Visualize OOD detection with best model
    if best_model is not None:
        print("\nGenerating OOD comparison visualization...")
        visualize_ood_comparison(best_model)