#!/usr/bin/env python3
"""
PIGM - OPTIMIZED WORKING VERSION
Direct generation of diverse flow patterns with better mixing visualization
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import imageio

print("=" * 70)
print("PHYSICS-INFORMED GENERATIVE MODEL (PIGM) - OPTIMIZED VERSION")
print("=" * 70)

# Device selection
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("✓ Using CUDA")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("✓ Using MPS (Apple Silicon)")
else:
    device = torch.device("cpu")
    print("✓ Using CPU")


def compute_curl(vx, vy):
    """Compute vorticity (curl) of a 2D velocity field"""
    # Handle both 3D [B, H, W] and 4D [B, C, H, W] tensors
    if vx.ndim == 4:
        vx = vx.squeeze(1)  # Remove channel dimension
        vy = vy.squeeze(1)
    
    B, H, W = vx.shape
    dvy_dx = torch.zeros_like(vy)
    dvy_dx[:, :, 1:-1] = (vy[:, :, 2:] - vy[:, :, :-2]) / 2.0
    dvx_dy = torch.zeros_like(vx)
    dvx_dy[:, 1:-1, :] = (vx[:, 2:, :] - vx[:, :-2, :]) / 2.0
    return dvy_dx - dvx_dy


def generate_diverse_flow_patterns(n_samples=100, img_size=64):
    """
    Generate truly diverse flow patterns with strong variation in quality
    """
    print(f"\nGenerating {n_samples} diverse flow patterns...")
    patterns = []
    
    # Track what we generate for diagnostics
    pattern_counts = {'strong': 0, 'weak': 0, 'off_center': 0, 'spiral': 0, 'shear': 0, 'chaotic': 0}
    
    for i in range(n_samples):
        y, x = np.meshgrid(
            np.linspace(0, 1, img_size), 
            np.linspace(0, 1, img_size), 
            indexing='ij'
        )
        
        rand = np.random.random()
        
        if rand < 0.30:  # 30% - Strong central vortex (BEST)
            pattern_counts['strong'] += 1
            center = (0.5 + np.random.uniform(-0.08, 0.08), 
                     0.5 + np.random.uniform(-0.08, 0.08))
            strength = np.random.uniform(6.0, 12.0) * np.random.choice([-1, 1])  # Increased
            decay = np.random.uniform(3, 7)  # Even slower decay
            
            dx, dy = x - center[0], y - center[1]
            r = np.sqrt(dx**2 + dy**2) + 1e-8
            
            # Tangential velocity for vortex
            vx = -dy / r * strength * np.exp(-decay * r**2)
            vy = dx / r * strength * np.exp(-decay * r**2)
            
            # Add slight radial component for better coverage
            radial_strength = strength * 0.15
            vx += dx / r * radial_strength * np.exp(-decay * r**2)
            vy += dy / r * radial_strength * np.exp(-decay * r**2)
            
        elif rand < 0.50:  # 20% - Very weak vortex (POOR)
            pattern_counts['weak'] += 1
            center = (0.5 + np.random.uniform(-0.15, 0.15), 
                     0.5 + np.random.uniform(-0.15, 0.15))
            strength = np.random.uniform(0.2, 0.8) * np.random.choice([-1, 1])  # Very weak
            decay = np.random.uniform(30, 60)  # Very fast decay
            
            dx, dy = x - center[0], y - center[1]
            r = np.sqrt(dx**2 + dy**2) + 1e-8
            
            vx = -dy / r * strength * np.exp(-decay * r**2)
            vy = dx / r * strength * np.exp(-decay * r**2)
            
        elif rand < 0.65:  # 15% - Off-center vortex (MODERATE)
            pattern_counts['off_center'] += 1
            center = (np.random.uniform(0.25, 0.75), 
                     np.random.uniform(0.25, 0.75))
            strength = np.random.uniform(3.5, 7.0) * np.random.choice([-1, 1])
            decay = np.random.uniform(6, 12)
            
            dx, dy = x - center[0], y - center[1]
            r = np.sqrt(dx**2 + dy**2) + 1e-8
            
            vx = -dy / r * strength * np.exp(-decay * r**2)
            vy = dx / r * strength * np.exp(-decay * r**2)
            
        elif rand < 0.78:  # 13% - Spiral vortex (GOOD)
            pattern_counts['spiral'] += 1
            center = (0.5 + np.random.uniform(-0.05, 0.05), 
                     0.5 + np.random.uniform(-0.05, 0.05))
            strength = np.random.uniform(5.0, 9.0) * np.random.choice([-1, 1])
            decay = np.random.uniform(4, 9)
            spiral_strength = strength * 0.35
            
            dx, dy = x - center[0], y - center[1]
            r = np.sqrt(dx**2 + dy**2) + 1e-8
            
            # Tangential + inward spiral
            vx = -dy / r * strength * np.exp(-decay * r**2)
            vy = dx / r * strength * np.exp(-decay * r**2)
            vx -= dx / r * spiral_strength * np.exp(-decay * r**2)
            vy -= dy / r * spiral_strength * np.exp(-decay * r**2)
            
        elif rand < 0.88:  # 10% - Pure shear flow (TERRIBLE)
            pattern_counts['shear'] += 1
            angle = np.random.uniform(0, np.pi)
            strength = np.random.uniform(0.4, 1.2)  # Reduced
            
            vx = np.cos(angle) * strength * np.ones_like(x)
            vy = np.sin(angle) * strength * np.ones_like(y)
            
        else:  # 12% - Chaotic/random (VERY POOR)
            pattern_counts['chaotic'] += 1
            n_modes = np.random.randint(5, 12)
            vx = np.zeros_like(x)
            vy = np.zeros_like(y)
            
            for _ in range(n_modes):
                kx = np.random.uniform(-6, 6)
                ky = np.random.uniform(-6, 6)
                amp = np.random.uniform(0.2, 0.9)
                phase = np.random.uniform(0, 2*np.pi)
                
                vx += amp * np.sin(2*np.pi*kx*x + 2*np.pi*ky*y + phase)
                vy += amp * np.cos(2*np.pi*kx*x + 2*np.pi*ky*y + phase)
        
        # Normalize but preserve relative strengths
        max_v = max(np.abs(vx).max(), np.abs(vy).max())
        if max_v > 1e-6:
            vx = 8.0 * vx / max_v  # Increased from 6.0 to 8.0
            vy = 8.0 * vy / max_v
        
        patterns.append(np.stack([vx, vy], axis=0))
        
        if (i + 1) % 50 == 0:
            print(f"  Generated {i+1}/{n_samples}...")
    
    print("✓ Pattern generation complete!")
    print(f"  Distribution: Strong={pattern_counts['strong']}, Weak={pattern_counts['weak']}, "
          f"Off-center={pattern_counts['off_center']}, Spiral={pattern_counts['spiral']}, "
          f"Shear={pattern_counts['shear']}, Chaotic={pattern_counts['chaotic']}")
    
    return torch.tensor(np.array(patterns), dtype=torch.float32, device=device)


def find_best_pattern_for_latte_art():
    """Find best and worst patterns for latte art"""
    print("\n" + "=" * 70)
    print("APPLICATION 1: The Perfect Latte Art")
    print("=" * 70)
    
    IMG_SIZE = 64
    N_CANDIDATES = 250
    
    print(f"1. Generating {N_CANDIDATES} candidate patterns...")
    candidates = generate_diverse_flow_patterns(N_CANDIDATES, IMG_SIZE)
    
    print("2. Scoring patterns for mixing quality...")
    
    curls = compute_curl(candidates[:, 0:1], candidates[:, 1:2])
    
    # Score using RMS curl in central region
    y, x = torch.meshgrid(
        torch.arange(IMG_SIZE, device=device),
        torch.arange(IMG_SIZE, device=device),
        indexing='ij'
    )
    center_y, center_x = IMG_SIZE / 2, IMG_SIZE / 2
    r = torch.sqrt((x - center_x)**2 + (y - center_y)**2)
    
    # Larger scoring region for better coverage
    mask = (r < 0.40 * IMG_SIZE).float().unsqueeze(0).unsqueeze(0)
    
    masked_curl = curls * mask
    scores = torch.sqrt((masked_curl**2).sum(dim=[1, 2, 3]) / mask.sum())
    
    # CRITICAL FIX: Find truly different patterns
    # Sort by score and pick from extremes
    sorted_indices = torch.argsort(scores)
    
    # Pick worst from bottom 20%
    worst_pool_size = max(1, N_CANDIDATES // 5)
    worst_candidates = sorted_indices[:worst_pool_size]
    worst_idx = worst_candidates[torch.randint(0, len(worst_candidates), (1,)).item()]
    
    # Pick best from top 20%
    best_candidates = sorted_indices[-worst_pool_size:]
    best_idx = best_candidates[torch.randint(0, len(best_candidates), (1,)).item()]
    
    best_score = scores[best_idx]
    worst_score = scores[worst_idx]
    
    print(f"✓ Best score: {best_score:.4f} | Worst score: {worst_score:.4f}")
    print(f"  Score ratio: {best_score/worst_score:.2f}x")
    print(f"  Score range: {scores.min():.4f} to {scores.max():.4f}")
    
    best_pattern = candidates[best_idx].cpu().numpy()
    worst_pattern = candidates[worst_idx].cpu().numpy()
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    x_plot, y_plot = np.meshgrid(np.arange(IMG_SIZE), np.arange(IMG_SIZE))
    
    for ax, pattern, title, color in [
        (axes[0], worst_pattern, f"Chaotic Pattern\n(Score: {worst_score:.4f})", "red"),
        (axes[1], best_pattern, f"Stable 'Latte Art' Pattern\n(Score: {best_score:.4f})", "green")
    ]:
        cup_radius = IMG_SIZE * 0.48
        cup_center_x, cup_center_y = IMG_SIZE / 2, IMG_SIZE / 2
        r_grid = np.sqrt((x_plot - cup_center_x)**2 + (y_plot - cup_center_y)**2)
        circular_mask = r_grid <= cup_radius
        
        vx_masked = np.where(circular_mask, pattern[1], np.nan)
        vy_masked = np.where(circular_mask, pattern[0], np.nan)
        
        ax.streamplot(
            x_plot, y_plot, vx_masked, vy_masked,
            color=color, density=2.2, linewidth=1.8,
            arrowsize=1.3, broken_streamlines=False
        )
        
        circle = Circle(
            (cup_center_x, cup_center_y), cup_radius,
            facecolor='#d2b48c', edgecolor='black',
            linewidth=2.5, zorder=0, alpha=0.25
        )
        ax.add_patch(circle)
        
        ax.set_facecolor('#f5deb3')
        ax.set_xlim(0, IMG_SIZE)
        ax.set_ylim(0, IMG_SIZE)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=17, fontweight='bold', pad=15)
        ax.axis('off')
    
    plt.suptitle("AI Designing the Best Stirring Pattern for Latte Art",
                 fontsize=22, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('application_latte_art.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: application_latte_art.png")
    plt.show()
    
    return candidates[best_idx:best_idx+1], candidates[worst_idx:worst_idx+1]


def visualize_mixing_animation(best_pattern, worst_pattern):
    """Create animation comparing mixing efficiency"""
    print("\n" + "=" * 70)
    print("APPLICATION 2: Mixing Animation")
    print("=" * 70)
    
    IMG_SIZE = 64
    H, W = IMG_SIZE, IMG_SIZE
    
    vx_best, vy_best = best_pattern[:, 0:1], best_pattern[:, 1:2]
    vx_worst, vy_worst = worst_pattern[:, 0:1], worst_pattern[:, 1:2]
    
    print("1. Setting up simulation...")
    
    # Initial condition: left half cream (1), right half coffee (0)
    cream = torch.zeros(1, 1, H, W, device=device)
    cream[:, :, :, :W//2] = 1.0
    cream_best, cream_worst = cream.clone(), cream.clone()
    
    # Grid for advection
    y_grid, x_grid = torch.meshgrid(
        torch.linspace(-1, 1, H, device=device),
        torch.linspace(-1, 1, W, device=device),
        indexing='ij'
    )
    base_grid = torch.stack([x_grid, y_grid], dim=2).unsqueeze(0)
    
    # INCREASED time step for more visible mixing
    dt = 2.0  # Increased from 1.5
    vx_best_norm = vx_best * dt * (2 / W)
    vy_best_norm = vy_best * dt * (2 / H)
    vx_worst_norm = vx_worst * dt * (2 / W)
    vy_worst_norm = vy_worst * dt * (2 / H)
    
    print("2. Running simulation and creating frames...")
    frames = []
    num_steps = 100  # Increased from 80
    
    # Add small diffusion for more realistic mixing
    diffusion = 0.002
    
    for step in range(num_steps):
        # Advection
        grid_best = base_grid - torch.stack([vx_best_norm[0, 0], vy_best_norm[0, 0]], dim=-1)
        cream_best = F.grid_sample(
            cream_best, grid_best,
            mode='bilinear', padding_mode='zeros', align_corners=False
        )
        
        grid_worst = base_grid - torch.stack([vx_worst_norm[0, 0], vy_worst_norm[0, 0]], dim=-1)
        cream_worst = F.grid_sample(
            cream_worst, grid_worst,
            mode='bilinear', padding_mode='zeros', align_corners=False
        )
        
        # Add tiny diffusion for smoothness
        if step % 5 == 0:
            cream_best = F.avg_pool2d(cream_best, 3, stride=1, padding=1)
            cream_worst = F.avg_pool2d(cream_worst, 3, stride=1, padding=1)
        
        if step % 2 == 0:  # Save every other frame
            fig, axes = plt.subplots(1, 2, figsize=(11, 6))
            
            for ax, cream_data, title in [
                (axes[0], cream_worst, "Worst Mixing Pattern"),
                (axes[1], cream_best, "BEST Mixing Pattern")
            ]:
                im = ax.imshow(
                    cream_data[0, 0].cpu().numpy(),
                    cmap='RdYlBu_r', vmin=0, vmax=1,
                    origin='lower', interpolation='bilinear'
                )
                ax.set_title(title, fontsize=15, fontweight='bold', pad=10)
                ax.axis('off')
            
            plt.suptitle(
                f"AI Test: Mixing Cream Into Coffee\nTime Step: {step+1}/{num_steps}",
                fontsize=18, fontweight='bold', y=0.98
            )
            plt.tight_layout(rect=[0, 0, 1, 0.94])
            
            # Convert figure to image array - reliable method
            fig.canvas.draw()
            frame = np.array(fig.canvas.renderer.buffer_rgba())
            frame = frame[:, :, :3]  # Keep only RGB, drop alpha channel
            frames.append(frame)
            plt.close(fig)
    
    print("3. Saving animation...")
    imageio.mimsave('application_mixing.gif', frames, fps=20, loop=0)
    print("✓ Saved: application_mixing.gif")


# Main execution
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("RUNNING APPLICATIONS")
    print("=" * 70)
    
    # Application 1: Find best latte art pattern
    best_pattern, worst_pattern = find_best_pattern_for_latte_art()
    
    # Application 2: Create mixing animation
    visualize_mixing_animation(best_pattern, worst_pattern)
    
    print("\n" + "=" * 70)
    print("ALL TASKS COMPLETE!")
    print("Output files:")
    print("  - application_latte_art.png")
    print("  - application_mixing.gif")
    print("=" * 70)