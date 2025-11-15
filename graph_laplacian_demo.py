"""
Graph Laplacian: Minimal Educational Demo
No PyTorch - just NumPy and visualization
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import networkx as nx

print("=" * 60)
print("GRAPH LAPLACIAN DEMONSTRATION")
print("=" * 60)

# ============================================
# PART 1: Build a Simple Graph
# ============================================

print("\n[1] Creating a simple graph...")

# Define graph as adjacency matrix
# Example: 6 nodes in a cycle with one shortcut
#
#     0 --- 1
#     |     |
#     5     2
#      \   /
#       \ /
#        3 --- 4

# Adjacency matrix (undirected graph)
A = np.array([
    [0, 1, 0, 0, 0, 1],  # Node 0 connects to 1, 5
    [1, 0, 1, 0, 0, 0],  # Node 1 connects to 0, 2
    [0, 1, 0, 1, 0, 0],  # Node 2 connects to 1, 3
    [0, 0, 1, 0, 1, 1],  # Node 3 connects to 2, 4, 5 (shortcut!)
    [0, 0, 0, 1, 0, 0],  # Node 4 connects to 3
    [1, 0, 0, 1, 0, 0],  # Node 5 connects to 0, 3
], dtype=float)

n_nodes = A.shape[0]
print(f"✓ Graph with {n_nodes} nodes")
print(f"✓ {np.sum(A)//2:.0f} edges")

# ============================================
# PART 2: Compute Graph Laplacian
# ============================================

print("\n[2] Computing Laplacian matrix...")

# Degree matrix: D[i,i] = sum of row i in A (number of connections)
D = np.diag(A.sum(axis=1))

# Graph Laplacian: L = D - A
L = D - A

print("\nAdjacency Matrix A:")
print(A.astype(int))

print("\nDegree Matrix D:")
print(D.astype(int))

print("\nLaplacian Matrix L = D - A:")
print(L.astype(int))

# Key property: L is symmetric, positive semi-definite
print(f"\n✓ L is symmetric: {np.allclose(L, L.T)}")
eigenvalues = np.linalg.eigvalsh(L)
print(f"✓ L is PSD (all eigenvalues ≥ 0): {np.all(eigenvalues >= -1e-10)}")

# ============================================
# PART 3: Eigendecomposition
# ============================================

print("\n[3] Eigendecomposition of Laplacian...")

eigvals, eigvecs = np.linalg.eigh(L)  # Use eigh for symmetric matrices

print(f"\nEigenvalues: {eigvals}")
print("\n(Eigenvalues tell us about graph connectivity)")
print(f"  λ₀ = {eigvals[0]:.6f} (always 0 for connected graphs)")
print(f"  λ₁ = {eigvals[1]:.6f} (Fiedler value - connectivity strength)")

# Smallest eigenvalue eigenvector (constant for connected graphs)
print(f"\nFirst eigenvector (λ₀): {eigvecs[:, 0]}")
print("  (Constant vector - all nodes have same value)")

# Second smallest (Fiedler vector) - useful for graph partitioning
fiedler = eigvecs[:, 1]
print(f"\nSecond eigenvector (λ₁, Fiedler): {fiedler}")
print("  (Signs indicate natural graph partition)")

# ============================================
# PART 4: Graph Smoothness
# ============================================

print("\n[4] Demonstrating graph smoothness...")

# Create two signals on the graph
signal_smooth = np.array([1.0, 1.2, 1.1, 0.9, 1.0, 1.1])  # Similar values
signal_rough = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0])  # Alternating

def graph_smoothness(signal, L):
    """Compute signal smoothness: s^T L s"""
    return signal.T @ L @ signal

smoothness_smooth = graph_smoothness(signal_smooth, L)
smoothness_rough = graph_smoothness(signal_rough, L)

print(f"\nSmooth signal: {signal_smooth}")
print(f"  Smoothness (s^T L s) = {smoothness_smooth:.3f}")

print(f"\nRough signal: {signal_rough}")
print(f"  Smoothness (s^T L s) = {smoothness_rough:.3f}")

print("\n✓ Lower smoothness value = smoother signal on graph")
print("  (Neighboring nodes have similar values)")

# ============================================
# PART 5: Spectral Graph Partitioning
# ============================================

print("\n[5] Using Fiedler vector for graph partitioning...")

# Partition based on sign of Fiedler vector
partition = (fiedler > 0).astype(int)

print(f"\nFiedler vector: {fiedler}")
print(f"Partition:      {partition}")
print("\nNodes grouped by partition:")
print(f"  Group 0: {np.where(partition == 0)[0].tolist()}")
print(f"  Group 1: {np.where(partition == 1)[0].tolist()}")

# ============================================
# PART 6: Visualization
# ============================================

print("\n[6] Generating visualizations...")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Define node positions for visualization
pos = {
    0: (0, 2),
    1: (2, 2),
    2: (3, 1),
    3: (2, 0),
    4: (3, -1),
    5: (1, 0),
}

def draw_graph(ax, node_colors, title, edge_alpha=0.3):
    """Helper to draw graph"""
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(-1.5, 2.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Draw edges
    for i in range(n_nodes):
        for j in range(i+1, n_nodes):
            if A[i, j] > 0:
                x = [pos[i][0], pos[j][0]]
                y = [pos[i][1], pos[j][1]]
                ax.plot(x, y, 'k-', alpha=edge_alpha, linewidth=2)
    
    # Draw nodes
    for i in range(n_nodes):
        circle = Circle(pos[i], 0.3, color=node_colors[i], 
                       ec='black', linewidth=2, zorder=10)
        ax.add_patch(circle)
        ax.text(pos[i][0], pos[i][1], str(i), 
               ha='center', va='center', fontsize=16, 
               fontweight='bold', zorder=11)

# (1) Basic graph structure
colors = ['lightblue'] * n_nodes
draw_graph(axes[0, 0], colors, "Graph Structure")

# (2) Node degrees
degree_colors = plt.cm.Reds(D.diagonal() / D.diagonal().max())
draw_graph(axes[0, 1], degree_colors, "Node Degrees (darker = more connections)")

# (3) Smooth signal
signal_colors = plt.cm.viridis((signal_smooth - signal_smooth.min()) / 
                               (signal_smooth.max() - signal_smooth.min()))
draw_graph(axes[0, 2], signal_colors, f"Smooth Signal (smoothness={smoothness_smooth:.2f})")

# (4) Rough signal
signal_colors = plt.cm.viridis((signal_rough - signal_rough.min()) / 
                               (signal_rough.max() - signal_rough.min()))
draw_graph(axes[1, 0], signal_colors, f"Rough Signal (smoothness={smoothness_rough:.2f})")

# (5) Fiedler vector
fiedler_norm = (fiedler - fiedler.min()) / (fiedler.max() - fiedler.min())
fiedler_colors = plt.cm.coolwarm(fiedler_norm)
draw_graph(axes[1, 1], fiedler_colors, "Fiedler Vector (2nd eigenvector)")

# (6) Graph partition
partition_colors = ['salmon' if p == 0 else 'lightgreen' for p in partition]
draw_graph(axes[1, 2], partition_colors, "Spectral Partitioning")

plt.tight_layout()
plt.savefig('graph_laplacian_demo.png', dpi=150, bbox_inches='tight')
print("✓ Saved visualization: graph_laplacian_demo.png")

# ============================================
# PART 7: Laplacian Eigenmaps (Dimensionality Reduction)
# ============================================

print("\n[7] Laplacian Eigenmaps: Embedding graph in 2D...")

# Use first 2 non-trivial eigenvectors for 2D embedding
embed_x = eigvecs[:, 1]  # 2nd eigenvector
embed_y = eigvecs[:, 2]  # 3rd eigenvector

fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ax.scatter(embed_x, embed_y, s=500, c=range(n_nodes), 
          cmap='tab10', edgecolors='black', linewidths=2)

# Draw edges in embedding space
for i in range(n_nodes):
    for j in range(i+1, n_nodes):
        if A[i, j] > 0:
            ax.plot([embed_x[i], embed_x[j]], 
                   [embed_y[i], embed_y[j]], 
                   'k-', alpha=0.3, linewidth=1)

# Label nodes
for i in range(n_nodes):
    ax.text(embed_x[i], embed_y[i], str(i), 
           ha='center', va='center', fontsize=16, 
           fontweight='bold', color='white')

ax.set_xlabel('2nd Eigenvector (λ₁)', fontsize=12)
ax.set_ylabel('3rd Eigenvector (λ₂)', fontsize=12)
ax.set_title('Laplacian Eigenmap: Graph Embedded in 2D', 
            fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='k', linewidth=0.5)
ax.axvline(x=0, color='k', linewidth=0.5)

plt.savefig('laplacian_eigenmap.png', dpi=150, bbox_inches='tight')
print("✓ Saved visualization: laplacian_eigenmap.png")

# ============================================
# PART 8: Random Walk Interpretation
# ============================================

print("\n[8] Random walk interpretation...")

# Normalized Laplacian: L_norm = I - D^(-1/2) A D^(-1/2)
D_inv_sqrt = np.diag(1.0 / np.sqrt(D.diagonal()))
L_norm = np.eye(n_nodes) - D_inv_sqrt @ A @ D_inv_sqrt

print("\nNormalized Laplacian L_norm:")
print(L_norm)

# Transition matrix for random walk: P = D^(-1) A
P = np.linalg.inv(D) @ A

print("\nRandom walk transition matrix P = D^(-1) A:")
print(P)
print("\n(Each row sums to 1 - probability distribution)")
print(f"Row sums: {P.sum(axis=1)}")

# Stationary distribution (eigenvector of P^T with eigenval 1)
eigenvals_P, eigenvecs_P = np.linalg.eig(P.T)
stationary_idx = np.argmax(np.abs(eigenvals_P - 1) < 1e-10)
stationary = np.abs(eigenvecs_P[:, stationary_idx])
stationary = stationary / stationary.sum()

print(f"\nStationary distribution: {stationary}")
print("(Long-term probability of being at each node)")

# ============================================
# SUMMARY
# ============================================

print("\n" + "=" * 60)
print("SUMMARY: What is the Graph Laplacian?")
print("=" * 60)

print("""
The Graph Laplacian L = D - A encodes graph structure:

1. DEFINITION:
   - D: Degree matrix (diagonal, D[i,i] = # connections to node i)
   - A: Adjacency matrix (A[i,j] = 1 if edge exists)
   - L = D - A

2. PROPERTIES:
   - Symmetric, positive semi-definite
   - Smallest eigenvalue λ₀ = 0 (eigenvector: constant)
   - λ₁ (Fiedler value) measures connectivity

3. USES:
   - Spectral clustering (partition graphs)
   - Graph signal smoothness (s^T L s)
   - Dimensionality reduction (Laplacian Eigenmaps)
   - Random walks on graphs
   - Graph neural networks (diffusion)

4. INTERPRETATION:
   - L measures "difference across edges"
   - s^T L s = Σ(edges) (s[i] - s[j])²
   - Small values → smooth signals on graph
   - Large values → rough/discontinuous signals

5. WHY IT MATTERS FOR YOUR RESEARCH:
   - Manuscript patches form a graph (spatial neighbors)
   - Degradation should be smooth (nearby regions similar)
   - Laplacian enforces spatial consistency
   - Can regularize: min Loss + λ·(d^T L d) where d=degradation
""")

print("=" * 60)
print("✓ Demo complete! Check generated PNG files.")
print("=" * 60)

plt.show()
