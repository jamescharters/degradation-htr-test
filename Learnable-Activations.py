import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np

# Simple CNN with adaptive activations
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10, activation_type='relu'):
        super().__init__()
        self.activation_type = activation_type
        
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(32 * 7 * 7, num_classes)
        
        if activation_type == 'relu':
            self.act1 = nn.ReLU()
            self.act2 = nn.ReLU()
        elif activation_type == 'prelu_shared':
            self.act1 = nn.PReLU(num_parameters=1)
            self.act2 = nn.PReLU(num_parameters=1)
        elif activation_type == 'prelu_perchannel':
            self.act1 = nn.PReLU(num_parameters=16)
            self.act2 = nn.PReLU(num_parameters=32)
    
    def forward(self, x):
        x = self.pool(self.act1(self.conv1(x)))
        x = self.pool(self.act2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
    def get_alpha_values(self):
        alphas = {}
        if self.activation_type != 'relu':
            alphas['conv1'] = self.act1.weight.detach().cpu().numpy()
            alphas['conv2'] = self.act2.weight.detach().cpu().numpy()
        return alphas

def add_intensity_shift(images, shift_type=None):
    """Apply random intensity transformations"""
    if shift_type is None:
        shift_type = np.random.choice(['none', 'gamma', 'contrast', 'bias'])
    
    if shift_type == 'none':
        return images
    elif shift_type == 'gamma':
        gamma = np.random.uniform(0.5, 1.5)
        return torch.pow(torch.clamp(images, 0.01, 1.0), gamma)
    elif shift_type == 'contrast':
        factor = np.random.uniform(0.6, 1.4)
        mean = images.mean()
        return torch.clamp((images - mean) * factor + mean, 0, 1)
    elif shift_type == 'bias':
        bias = np.random.uniform(-0.3, 0.3)
        return torch.clamp(images + bias, 0, 1)
    return images

class AugmentedDataset(Dataset):
    """Dataset with random intensity augmentations"""
    def __init__(self, base_dataset, augment=True):
        self.base_dataset = base_dataset
        self.augment = augment
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        img, label = self.base_dataset[idx]
        if self.augment:
            img = add_intensity_shift(img, shift_type=None)  # Random
        return img, label

class FixedShiftDataset(Dataset):
    """Dataset with fixed shift type for testing"""
    def __init__(self, base_dataset, shift_type='gamma'):
        self.base_dataset = base_dataset
        self.shift_type = shift_type
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        img, label = self.base_dataset[idx]
        img = add_intensity_shift(img, shift_type=self.shift_type)
        return img, label

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
    
    return total_loss / len(loader), correct / total

def test(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    
    return total_loss / len(loader), correct / total

def run_multidomain_experiment(activation_type, use_augmentation=False, epochs=15):
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    
    mode = "Multi-Domain" if use_augmentation else "Single-Domain"
    print(f"\n{'='*60}")
    print(f"{mode} Training with {activation_type}")
    print(f"Device: {device}")
    print(f"{'='*60}")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Training data - with or without augmentation
    train_dataset_base = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_dataset = AugmentedDataset(train_dataset_base, augment=use_augmentation)
    
    # Test data - always clean and with fixed shifts
    test_dataset_base = datasets.MNIST('./data', train=False, transform=transform)
    test_dataset_clean = AugmentedDataset(test_dataset_base, augment=False)
    test_dataset_gamma = FixedShiftDataset(test_dataset_base, shift_type='gamma')
    test_dataset_contrast = FixedShiftDataset(test_dataset_base, shift_type='contrast')
    test_dataset_bias = FixedShiftDataset(test_dataset_base, shift_type='bias')
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader_clean = DataLoader(test_dataset_clean, batch_size=128, shuffle=False)
    test_loader_gamma = DataLoader(test_dataset_gamma, batch_size=128, shuffle=False)
    test_loader_contrast = DataLoader(test_dataset_contrast, batch_size=128, shuffle=False)
    test_loader_bias = DataLoader(test_dataset_bias, batch_size=128, shuffle=False)
    
    # Model
    model = SimpleCNN(num_classes=10, activation_type=activation_type).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Training
    results = {
        'train_accs': [],
        'test_clean': [],
        'test_gamma': [],
        'test_contrast': [],
        'test_bias': []
    }
    
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        
        _, test_acc_clean = test(model, test_loader_clean, criterion, device)
        _, test_acc_gamma = test(model, test_loader_gamma, criterion, device)
        _, test_acc_contrast = test(model, test_loader_contrast, criterion, device)
        _, test_acc_bias = test(model, test_loader_bias, criterion, device)
        
        results['train_accs'].append(train_acc)
        results['test_clean'].append(test_acc_clean)
        results['test_gamma'].append(test_acc_gamma)
        results['test_contrast'].append(test_acc_contrast)
        results['test_bias'].append(test_acc_bias)
        
        if (epoch + 1) % 3 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Train: {train_acc:.4f} | "
                  f"Clean: {test_acc_clean:.4f} | Gamma: {test_acc_gamma:.4f} | "
                  f"Contrast: {test_acc_contrast:.4f} | Bias: {test_acc_bias:.4f}")
    
    # Get alpha values
    alphas = model.get_alpha_values()
    
    # Calculate metrics
    clean_acc = results['test_clean'][-1]
    shifted_accs = [results['test_gamma'][-1], results['test_contrast'][-1], results['test_bias'][-1]]
    avg_shifted = np.mean(shifted_accs)
    
    results['alphas'] = alphas
    results['final_clean'] = clean_acc
    results['final_shifted_avg'] = avg_shifted
    results['robustness'] = avg_shifted
    
    return results

def visualize_comparison(single_domain_results, multi_domain_results):
    """Compare single-domain vs multi-domain training"""
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # Plot 1: Clean accuracy - Single Domain
    ax1 = fig.add_subplot(gs[0, :2])
    for name, res in single_domain_results.items():
        ax1.plot(res['test_clean'], label=f"{name}", marker='o', markersize=4)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Single-Domain Training: Clean Test Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Clean accuracy - Multi Domain
    ax2 = fig.add_subplot(gs[0, 2:])
    for name, res in multi_domain_results.items():
        ax2.plot(res['test_clean'], label=f"{name}", marker='o', markersize=4)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Multi-Domain Training: Clean Test Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Shifted accuracy - Single Domain
    ax3 = fig.add_subplot(gs[1, :2])
    for name, res in single_domain_results.items():
        epochs = range(len(res['test_gamma']))
        ax3.plot(epochs, res['test_gamma'], label=f"{name} (gamma)", alpha=0.7)
        ax3.plot(epochs, res['test_contrast'], label=f"{name} (contrast)", alpha=0.7, linestyle='--')
        ax3.plot(epochs, res['test_bias'], label=f"{name} (bias)", alpha=0.7, linestyle=':')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Accuracy')
    ax3.set_title('Single-Domain: Performance Under Shift')
    ax3.legend(fontsize=7, ncol=3)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Shifted accuracy - Multi Domain
    ax4 = fig.add_subplot(gs[1, 2:])
    for name, res in multi_domain_results.items():
        epochs = range(len(res['test_gamma']))
        ax4.plot(epochs, res['test_gamma'], label=f"{name} (gamma)", alpha=0.7)
        ax4.plot(epochs, res['test_contrast'], label=f"{name} (contrast)", alpha=0.7, linestyle='--')
        ax4.plot(epochs, res['test_bias'], label=f"{name} (bias)", alpha=0.7, linestyle=':')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Accuracy')
    ax4.set_title('Multi-Domain: Performance Under Shift')
    ax4.legend(fontsize=7, ncol=3)
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Bar comparison - Clean accuracy
    ax5 = fig.add_subplot(gs[2, 0])
    names = list(single_domain_results.keys())
    single_clean = [single_domain_results[n]['final_clean'] for n in names]
    multi_clean = [multi_domain_results[n]['final_clean'] for n in names]
    
    x = np.arange(len(names))
    width = 0.35
    ax5.bar(x - width/2, single_clean, width, label='Single-Domain', alpha=0.8)
    ax5.bar(x + width/2, multi_clean, width, label='Multi-Domain', alpha=0.8)
    ax5.set_ylabel('Accuracy')
    ax5.set_title('Clean Test Accuracy')
    ax5.set_xticks(x)
    ax5.set_xticklabels(names, rotation=15, ha='right')
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='y')
    
    # Plot 6: Bar comparison - Shifted accuracy
    ax6 = fig.add_subplot(gs[2, 1])
    single_shifted = [single_domain_results[n]['final_shifted_avg'] for n in names]
    multi_shifted = [multi_domain_results[n]['final_shifted_avg'] for n in names]
    
    ax6.bar(x - width/2, single_shifted, width, label='Single-Domain', alpha=0.8)
    ax6.bar(x + width/2, multi_shifted, width, label='Multi-Domain', alpha=0.8)
    ax6.set_ylabel('Accuracy')
    ax6.set_title('Avg Shifted Test Accuracy')
    ax6.set_xticks(x)
    ax6.set_xticklabels(names, rotation=15, ha='right')
    ax6.legend()
    ax6.grid(True, alpha=0.3, axis='y')
    
    # Plot 7: Robustness improvement
    ax7 = fig.add_subplot(gs[2, 2])
    improvements = [multi_shifted[i] - single_shifted[i] for i in range(len(names))]
    colors = ['green' if imp > 0 else 'red' for imp in improvements]
    ax7.bar(x, improvements, color=colors, alpha=0.7)
    ax7.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax7.set_ylabel('Improvement')
    ax7.set_title('Robustness Gain (Multi - Single)')
    ax7.set_xticks(x)
    ax7.set_xticklabels(names, rotation=15, ha='right')
    ax7.grid(True, alpha=0.3, axis='y')
    
    # Plot 8: Alpha comparison for per-channel
    ax8 = fig.add_subplot(gs[2, 3])
    if 'prelu_perchannel' in single_domain_results:
        single_alphas = single_domain_results['prelu_perchannel']['alphas']
        multi_alphas = multi_domain_results['prelu_perchannel']['alphas']
        
        if 'conv1' in single_alphas and 'conv1' in multi_alphas:
            s_c1 = single_alphas['conv1'].flatten()
            m_c1 = multi_alphas['conv1'].flatten()
            s_c2 = single_alphas['conv2'].flatten()
            m_c2 = multi_alphas['conv2'].flatten()
            
            ax8.scatter(s_c1, m_c1, label='Conv1', alpha=0.6, s=40)
            ax8.scatter(s_c2, m_c2, label='Conv2', alpha=0.6, s=40)
            ax8.plot([0, 0.6], [0, 0.6], 'k--', alpha=0.3, label='y=x')
            ax8.set_xlabel('Single-Domain Alpha')
            ax8.set_ylabel('Multi-Domain Alpha')
            ax8.set_title('Alpha Values Comparison')
            ax8.legend()
            ax8.grid(True, alpha=0.3)
    
    plt.savefig('multidomain_comparison.png', dpi=150, bbox_inches='tight')
    print("\nResults saved to 'multidomain_comparison.png'")
    plt.show()

if __name__ == '__main__':
    print("="*60)
    print("EXPERIMENT: Single-Domain vs Multi-Domain Training")
    print("="*60)
    
    # Single-domain training (original)
    print("\n### PHASE 1: Single-Domain Training ###")
    single_domain_results = {}
    for activation_type in ['relu', 'prelu_shared', 'prelu_perchannel']:
        single_domain_results[activation_type] = run_multidomain_experiment(
            activation_type, use_augmentation=False, epochs=15
        )
    
    # Multi-domain training (with augmentation)
    print("\n### PHASE 2: Multi-Domain Training ###")
    multi_domain_results = {}
    for activation_type in ['relu', 'prelu_shared', 'prelu_perchannel']:
        multi_domain_results[activation_type] = run_multidomain_experiment(
            activation_type, use_augmentation=True, epochs=15
        )
    
    # Print comprehensive comparison
    print(f"\n{'='*80}")
    print("FINAL COMPARISON:")
    print(f"{'='*80}")
    print(f"{'Method':<20} {'Training':<15} {'Clean':>8} {'Shifted':>8} {'Drop':>8}")
    print(f"{'-'*80}")
    
    for name in single_domain_results.keys():
        s_res = single_domain_results[name]
        m_res = multi_domain_results[name]
        
        print(f"{name:<20} {'Single-Domain':<15} {s_res['final_clean']:>8.4f} "
              f"{s_res['final_shifted_avg']:>8.4f} {s_res['final_clean']-s_res['final_shifted_avg']:>8.4f}")
        print(f"{'':<20} {'Multi-Domain':<15} {m_res['final_clean']:>8.4f} "
              f"{m_res['final_shifted_avg']:>8.4f} {m_res['final_clean']-m_res['final_shifted_avg']:>8.4f}")
        
        improvement = m_res['final_shifted_avg'] - s_res['final_shifted_avg']
        print(f"{'':<20} {'Improvement':<15} {'':<8} {improvement:>8.4f}")
        print(f"{'-'*80}")
    
    # Visualize
    visualize_comparison(single_domain_results, multi_domain_results)