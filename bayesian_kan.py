import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class BayesianKANLayer(nn.Module):
    def __init__(self, n_basis=10, x_min=-5, x_max=5):
        super().__init__()
        # B-spline coefficients: mean and log-variance
        self.c_mu = nn.Parameter(torch.zeros(n_basis))
        self.c_logvar = nn.Parameter(torch.zeros(n_basis) - 2)
        
        # Fixed B-spline knots (uniform)
        self.knots = torch.linspace(x_min, x_max, n_basis + 4)
        self.n_basis = n_basis
    
    def basis(self, x):
        # Simplified: use RBF instead of true B-splines for toy
        centers = torch.linspace(-5, 5, self.n_basis)
        return torch.exp(-0.5 * (x.unsqueeze(-1) - centers)**2 / 0.5**2)
    
    def forward(self, x, n_samples=1):
        B = self.basis(x)  # [batch, n_basis]
        
        # Reparameterization trick
        std = torch.exp(0.5 * self.c_logvar)
        eps = torch.randn(n_samples, self.n_basis)
        c_samples = self.c_mu + eps * std  # [n_samples, n_basis]
        
        # Output: [n_samples, batch]
        return (B @ c_samples.T).T
    
    def kl_divergence(self):
        # KL(q(c) || p(c)) where p(c) = N(0, I)
        return -0.5 * torch.sum(1 + self.c_logvar - self.c_mu**2 - self.c_logvar.exp())

# Training
model = BayesianKANLayer(n_basis=15)
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

# Sparse training data (gap in middle)
x_train = torch.cat([torch.linspace(-5, -1, 20), torch.linspace(2, 5, 20)])
y_train = torch.sin(x_train) + 0.1 * torch.randn_like(x_train)

for epoch in range(1000):
    y_pred = model(x_train, n_samples=10).mean(0)
    nll = ((y_pred - y_train)**2).mean()
    kl = model.kl_divergence() / len(x_train)
    loss = nll + 0.01 * kl  # ELBO
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Prediction with uncertainty
x_test = torch.linspace(-5, 5, 200)
with torch.no_grad():
    samples = model(x_test, n_samples=100).numpy()
    mean = samples.mean(0)
    std = samples.std(0)

# Plot
plt.figure(figsize=(10, 4))
plt.fill_between(x_test, mean - 2*std, mean + 2*std, alpha=0.3, label='±2σ')
plt.plot(x_test, mean, 'b-', label='Mean')
plt.plot(x_test, np.sin(x_test), 'k--', label='True')
plt.scatter(x_train, y_train, c='r', s=20, label='Train')
plt.legend()
plt.title('Bayesian KAN: Uncertainty should be HIGH in [-1, 2] gap')
plt.show()