import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import ast
import os
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import train_test_split
import wfdb
import matplotlib.pyplot as plt


# ============ PTB-XL DATA LOADING ============
def load_ptbxl(data_path='ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/', 
               sampling_rate=100):
    """
    Load PTB-XL dataset
    
    Download from: https://physionet.org/content/ptb-xl/1.0.3/
    
    Extract to get folder structure:
    ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/
        ├── ptbxl_database.csv
        ├── scp_statements.csv
        ├── records100/
        └── records500/
    """
    
    # Load metadata
    Y = pd.read_csv(os.path.join(data_path, 'ptbxl_database.csv'), index_col='ecg_id')
    Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))
    
    # Load SCP statements for mapping
    scp = pd.read_csv(os.path.join(data_path, 'scp_statements.csv'), index_col=0)
    scp = scp[scp.diagnostic == 1]
    
    # Map to 5 superclasses: NORM, MI, STTC, CD, HYP
    def get_superclass(scp_codes):
        for code, conf in scp_codes.items():
            if conf >= 50:  # At least 50% confidence
                if code in scp.index:
                    dc = scp.loc[code].diagnostic_class
                    if pd.notna(dc):
                        return dc
        return None
    
    Y['superclass'] = Y.scp_codes.apply(get_superclass)
    Y = Y[Y.superclass.notna()]  # Keep only samples with valid superclass
    
    # Create label mapping
    classes = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
    class_to_idx = {c: i for i, c in enumerate(classes)}
    Y['label'] = Y.superclass.map(class_to_idx)
    Y = Y[Y.label.notna()]
    
    print(f"Loading {len(Y)} ECG records...")
    
    # Load signals
    if sampling_rate == 100:
        folder = 'records100'
    else:
        folder = 'records500'
    
    X = []
    valid_indices = []
    
    for idx, row in Y.iterrows():
        file_path = os.path.join(data_path, row.filename_lr if sampling_rate == 100 else row.filename_hr)
        try:
            record = wfdb.rdrecord(file_path.replace('.dat', '').replace('.hea', ''))
            signal = record.p_signal  # Shape: (1000, 12) for 100Hz
            X.append(signal)
            valid_indices.append(idx)
        except Exception as e:
            continue
    
    X = np.array(X)  # Shape: (N, 1000, 12)
    Y = Y.loc[valid_indices]
    y = Y['label'].values.astype(int)
    
    print(f"Loaded {len(X)} records, shape: {X.shape}")
    print(f"Class distribution: {np.bincount(y)}")
    
    return X, y, classes


def load_ptbxl_simple(data_path='ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/',
                      max_samples=2000, lead=0):
    """
    Simplified loader - single lead, limited samples for quick experiments
    """
    X_full, y_full, classes = load_ptbxl(data_path)
    
    # Use single lead and limit samples
    X = X_full[:max_samples, :, lead]  # Shape: (N, 1000)
    y = y_full[:max_samples]
    
    # Normalize per-sample
    X = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-6)
    
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long), classes


# ============ ALTERNATIVE: DOWNLOAD-FREE SYNTHETIC VERSION ============
def make_realistic_ecg_data(n=1000, seq_len=1000):
    """
    More realistic synthetic ECG with 5 classes mimicking real patterns
    Use this if you can't download PTB-XL
    """
    X = torch.randn(n, seq_len) * 0.3
    y = torch.zeros(n, dtype=torch.long)
    
    for i in range(n):
        c = i % 5
        t = torch.linspace(0, 10, seq_len)
        
        if c == 0:  # NORM - regular sinus rhythm
            for beat in range(10):
                pos = int(100 * beat + 50)
                if pos + 50 < seq_len:
                    # P wave
                    X[i, pos:pos+20] += 0.5 * torch.sin(torch.linspace(0, np.pi, 20))
                    # QRS complex
                    X[i, pos+25:pos+30] -= 0.3
                    X[i, pos+30:pos+35] += 2.5
                    X[i, pos+35:pos+40] -= 0.3
                    # T wave
                    X[i, pos+45:pos+65] += 0.7 * torch.sin(torch.linspace(0, np.pi, 20))
        
        elif c == 1:  # MI - ST elevation, Q waves
            for beat in range(10):
                pos = int(100 * beat + 50)
                if pos + 70 < seq_len:
                    X[i, pos:pos+20] += 0.3 * torch.sin(torch.linspace(0, np.pi, 20))
                    X[i, pos+20:pos+25] -= 1.0  # Deep Q wave
                    X[i, pos+25:pos+30] += 2.0
                    X[i, pos+30:pos+35] -= 0.2
                    X[i, pos+35:pos+60] += 1.2  # ST elevation
                    X[i, pos+55:pos+70] += 0.5 * torch.sin(torch.linspace(0, np.pi, 15))
        
        elif c == 2:  # STTC - ST/T changes
            for beat in range(10):
                pos = int(100 * beat + 50)
                if pos + 70 < seq_len:
                    X[i, pos:pos+20] += 0.4 * torch.sin(torch.linspace(0, np.pi, 20))
                    X[i, pos+25:pos+30] -= 0.2
                    X[i, pos+30:pos+35] += 2.2
                    X[i, pos+35:pos+40] -= 0.2
                    X[i, pos+40:pos+60] -= 0.8  # ST depression
                    X[i, pos+55:pos+75] -= 0.6 * torch.sin(torch.linspace(0, np.pi, 20))  # Inverted T
        
        elif c == 3:  # CD - conduction defect, wide QRS
            for beat in range(8):  # Slower rate
                pos = int(125 * beat + 50)
                if pos + 80 < seq_len:
                    X[i, pos:pos+20] += 0.4 * torch.sin(torch.linspace(0, np.pi, 20))
                    # Wide QRS
                    X[i, pos+25:pos+35] -= 0.4
                    X[i, pos+35:pos+50] += 1.8
                    X[i, pos+50:pos+60] -= 0.4
                    X[i, pos+65:pos+85] += 0.5 * torch.sin(torch.linspace(0, np.pi, 20))
        
        else:  # HYP - high voltage, LVH pattern
            for beat in range(10):
                pos = int(100 * beat + 50)
                if pos + 70 < seq_len:
                    X[i, pos:pos+20] += 0.6 * torch.sin(torch.linspace(0, np.pi, 20))
                    X[i, pos+25:pos+30] -= 0.3
                    X[i, pos+30:pos+35] += 4.0  # High voltage
                    X[i, pos+35:pos+40] -= 0.3
                    X[i, pos+40:pos+55] -= 0.5  # Strain pattern
                    X[i, pos+50:pos+70] -= 0.4 * torch.sin(torch.linspace(0, np.pi, 20))
        
        y[i] = c
    
    # Add noise
    X += torch.randn_like(X) * 0.2
    
    return X, y


# ============ BAYESIAN KAN MODEL ============
class BayesianKAN_ECG(nn.Module):
    def __init__(self, seq_len=1000, hidden=64, out_dim=5, n_basis=16, min_std=0.15):
        super().__init__()
        self.min_std = min_std
        self.n_basis = n_basis
        
        # Downsample 1000 -> 100
        self.pool_size = seq_len // 100
        in_dim = 100
        
        # Layer 1
        self.c1_mu = nn.Parameter(torch.randn(hidden, in_dim, n_basis) * 0.05)
        self.c1_logstd = nn.Parameter(torch.zeros(hidden, in_dim, n_basis) - 2)
        
        # Layer 2
        self.c2_mu = nn.Parameter(torch.randn(out_dim, hidden, n_basis) * 0.05)
        self.c2_logstd = nn.Parameter(torch.zeros(out_dim, hidden, n_basis) - 2)
        
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
        # x: [batch, seq_len]
        batch = x.shape[0]
        
        # Downsample
        x = x.unfold(1, self.pool_size, self.pool_size).mean(-1)  # [batch, 100]
        
        # Normalize
        x = (x - x.mean(1, keepdim=True)) / (x.std(1, keepdim=True) + 1e-6)
        
        # KAN layers
        h = self.kan_layer(x, self.c1_mu, self.c1_logstd, sample)
        h = torch.tanh(h)
        h = (h - h.mean(1, keepdim=True)) / (h.std(1, keepdim=True) + 1e-6)
        
        out = self.kan_layer(h, self.c2_mu, self.c2_logstd, sample)
        return out


# ============ TRAINING ============
def train_model(model, X_train, y_train, X_val, y_val, epochs=100, lr=0.01, batch_size=64):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    best_val_acc = 0
    best_state = None
    
    for epoch in range(epochs):
        model.train()
        
        # Shuffle
        perm = torch.randperm(len(X_train))
        total_loss = 0
        
        for i in range(0, len(X_train), batch_size):
            idx = perm[i:i+batch_size]
            
            # Forward with sampling (average 3 samples for stability)
            logits = torch.stack([model(X_train[idx], sample=True) for _ in range(3)]).mean(0)
            loss = F.cross_entropy(logits, y_train[idx])
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val, sample=False)
            val_loss = F.cross_entropy(val_logits, y_val)
            val_acc = (val_logits.argmax(1) == y_val).float().mean().item()
        
        scheduler.step(val_loss)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict().copy()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: train_loss={total_loss/len(X_train)*batch_size:.4f}, "
                  f"val_acc={val_acc:.1%}, lr={optimizer.param_groups[0]['lr']:.6f}")
    
    # Load best model
    model.load_state_dict(best_state)
    return model, best_val_acc


# ============ UNCERTAINTY EVALUATION ============
def get_uncertainty(model, X, n_samples=100):
    model.eval()
    with torch.no_grad():
        logits = torch.stack([model(X, sample=True) for _ in range(n_samples)])
        probs = torch.softmax(logits, -1)
        mean_probs = probs.mean(0)
        
        confidence = mean_probs.max(-1).values
        entropy = -(mean_probs * torch.log(mean_probs + 1e-8)).sum(-1)
        
        preds = logits.argmax(-1)
        mode_pred = preds.mode(0).values
        disagreement = (preds != mode_pred).float().mean(0)
        
        logit_std = logits.std(0).mean(-1)
        
    return {
        'predictions': mean_probs.argmax(-1),
        'probabilities': mean_probs,
        'confidence': confidence,
        'entropy': entropy,
        'disagreement': disagreement,
        'logit_std': logit_std,
    }


def evaluate_ood_detection(model, X_id, y_id, X_ood):
    """Evaluate OOD detection with AUROC"""
    scores_id = get_uncertainty(model, X_id)
    scores_ood = get_uncertainty(model, X_ood)
    
    # Classification accuracy
    acc = (scores_id['predictions'] == y_id).float().mean().item()
    
    print(f"\nClassification Accuracy: {acc:.1%}")
    print(f"\n{'Metric':<15} {'ID mean':>10} {'OOD mean':>10} {'AUROC':>10}")
    print("-" * 50)
    
    results = {}
    for metric in ['entropy', 'disagreement', 'logit_std']:
        id_scores = scores_id[metric].numpy()
        ood_scores = scores_ood[metric].numpy()
        
        labels = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
        scores = np.concatenate([id_scores, ood_scores])
        
        auroc = roc_auc_score(labels, scores)
        print(f"{metric:<15} {id_scores.mean():>10.4f} {ood_scores.mean():>10.4f} {auroc:>10.3f}")
        results[metric] = auroc
    
    # Confidence (flip for AUROC)
    id_conf = scores_id['confidence'].numpy()
    ood_conf = scores_ood['confidence'].numpy()
    labels = np.concatenate([np.zeros(len(id_conf)), np.ones(len(ood_conf))])
    scores = np.concatenate([id_conf, ood_conf])
    auroc_conf = roc_auc_score(labels, -scores)
    print(f"{'confidence':<15} {id_conf.mean():>10.4f} {ood_conf.mean():>10.4f} {auroc_conf:>10.3f}")
    results['confidence'] = auroc_conf
    
    return results, acc, scores_id


def create_ood_data(n=200, seq_len=1000):
    """Create various OOD ECG-like signals"""
    ood_data = []
    
    # 1. Pure noise
    ood_data.append(torch.randn(n//4, seq_len) * 2)
    
    # 2. Flat line (asystole-like)
    ood_data.append(torch.randn(n//4, seq_len) * 0.05)
    
    # 3. High frequency noise (muscle artifact)
    t = torch.linspace(0, 100, seq_len)
    hf_noise = torch.sin(t * 50).unsqueeze(0).repeat(n//4, 1) + torch.randn(n//4, seq_len) * 0.5
    ood_data.append(hf_noise)
    
    # 4. Abnormal spikes (pacemaker-like)
    spikes = torch.randn(n//4, seq_len) * 0.3
    for i in range(n//4):
        for j in range(0, seq_len, 80):
            if j + 5 < seq_len:
                spikes[i, j:j+5] = 10.0
    ood_data.append(spikes)
    
    return torch.cat(ood_data)


# ============ VISUALIZATION ============
def plot_results(X_test, y_test, scores, classes, save_path='ptbxl_results.png'):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Sample ECGs per class
    ax = axes[0, 0]
    for c in range(min(5, len(classes))):
        idx = (y_test == c).nonzero()[0]
        if len(idx) > 0:
            ax.plot(X_test[idx[0]].numpy() + c * 5, label=classes[c], alpha=0.8)
    ax.set_xlabel('Time')
    ax.set_ylabel('Amplitude (offset)')
    ax.set_title('Sample ECGs per Class')
    ax.legend(loc='upper right', fontsize=8)
    
    # 2. Confidence histogram by correctness
    ax = axes[0, 1]
    correct = scores['predictions'] == y_test
    ax.hist(scores['confidence'][correct].numpy(), bins=20, alpha=0.6, label='Correct', color='green')
    ax.hist(scores['confidence'][~correct].numpy(), bins=20, alpha=0.6, label='Wrong', color='red')
    ax.set_xlabel('Confidence')
    ax.set_ylabel('Count')
    ax.set_title('Confidence Distribution')
    ax.legend()
    
    # 3. Entropy vs Confidence
    ax = axes[0, 2]
    colors = ['green' if c else 'red' for c in correct.numpy()]
    ax.scatter(scores['confidence'].numpy(), scores['entropy'].numpy(), c=colors, alpha=0.5, s=10)
    ax.set_xlabel('Confidence')
    ax.set_ylabel('Entropy')
    ax.set_title('Confidence vs Entropy (green=correct)')
    
    # 4. Confusion-style: uncertainty per class
    ax = axes[1, 0]
    class_entropy = [scores['entropy'][y_test == c].mean().item() for c in range(len(classes))]
    ax.bar(classes, class_entropy)
    ax.set_ylabel('Mean Entropy')
    ax.set_title('Uncertainty by Class')
    ax.tick_params(axis='x', rotation=45)
    
    # 5. Calibration
    ax = axes[1, 1]
    conf = scores['confidence'].numpy()
    bins = np.linspace(0, 1, 11)
    bin_accs, bin_confs = [], []
    for i in range(len(bins) - 1):
        mask = (conf >= bins[i]) & (conf < bins[i+1])
        if mask.sum() > 0:
            bin_accs.append(correct[mask].float().mean().item())
            bin_confs.append((bins[i] + bins[i+1]) / 2)
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect')
    ax.plot(bin_confs, bin_accs, 'bo-', label='Model')
    ax.set_xlabel('Confidence')
    ax.set_ylabel('Accuracy')
    ax.set_title('Calibration Plot')
    ax.legend()
    
    # 6. Per-class accuracy
    ax = axes[1, 2]
    class_acc = [(scores['predictions'][y_test == c] == c).float().mean().item() 
                  for c in range(len(classes))]
    ax.bar(classes, class_acc)
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy by Class')
    ax.tick_params(axis='x', rotation=45)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Saved: {save_path}")


# ============ MAIN ============
if __name__ == '__main__':
    print("=" * 70)
    print("BAYESIAN KAN FOR ECG CLASSIFICATION")
    print("=" * 70)
    
    # Try to load real PTB-XL, fall back to synthetic
    USE_REAL_DATA = True  # Set to True if you have PTB-XL downloaded
    
    if USE_REAL_DATA:
        try:
            print("\nLoading PTB-XL dataset...")
            X, y, classes = load_ptbxl_simple(
                data_path='./data/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/',
                max_samples=3000,
                lead=0
            )
        except Exception as e:
            print(f"Failed to load PTB-XL: {e}")
            print("Falling back to synthetic data...")
            USE_REAL_DATA = False
    
    if not USE_REAL_DATA:
        print("\nUsing realistic synthetic ECG data...")
        X, y = make_realistic_ecg_data(n=2000, seq_len=1000)
        classes = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
    
    print(f"Data shape: {X.shape}")
    print(f"Classes: {classes}")
    print(f"Class distribution: {torch.bincount(y).tolist()}")
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)
    
    # Convert to tensors if needed
    if not isinstance(X_train, torch.Tensor):
        X_train, X_val, X_test = map(lambda x: torch.tensor(x, dtype=torch.float32), [X_train, X_val, X_test])
        y_train, y_val, y_test = map(lambda x: torch.tensor(x, dtype=torch.long), [y_train, y_val, y_test])
    
    print(f"\nTrain: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Create model
    print("\nCreating Bayesian KAN model...")
    model = BayesianKAN_ECG(
        seq_len=X_train.shape[1],
        hidden=64,
        out_dim=len(classes),
        n_basis=16,
        min_std=0.15
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
    
    # Train
    print("\nTraining...")
    model, best_val_acc = train_model(
        model, X_train, y_train, X_val, y_val,
        epochs=100, lr=0.01, batch_size=64
    )
    print(f"\nBest validation accuracy: {best_val_acc:.1%}")
    
    # Test set evaluation
    print("\n" + "=" * 70)
    print("TEST SET EVALUATION")
    print("=" * 70)
    
    scores = get_uncertainty(model, X_test)
    test_acc = (scores['predictions'] == y_test).float().mean().item()
    print(f"\nTest Accuracy: {test_acc:.1%}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test.numpy(), scores['predictions'].numpy(), 
                                target_names=classes, digits=3))
    
    # OOD Detection
    print("\n" + "=" * 70)
    print("OOD DETECTION EVALUATION")
    print("=" * 70)
    
    X_ood = create_ood_data(n=200, seq_len=X_test.shape[1])
    ood_results, _, _ = evaluate_ood_detection(model, X_test, y_test, X_ood)
    
    print(f"\nBest OOD metric: {max(ood_results, key=ood_results.get)} "
          f"(AUROC={max(ood_results.values()):.3f})")
    
    # Visualize
    print("\nGenerating plots...")
    plot_results(X_test, y_test, scores, classes)
    
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)