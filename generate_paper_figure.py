import torch
import matplotlib.pyplot as plt
from torch.fft import fft2, ifft2
import nibabel as nib
from skimage.transform import resize
import numpy as np

# ... (Paste the Model Classes RBFKANLayer and SimpleUNet from previous script here) ...
# ... (Or just run this in the same session where models are defined) ...

def visualize_paper_figure(cnn_model, kan_model, val_loader):
    print("Generating High-Res Figure...")
    cnn_model.eval()
    kan_model.eval()
    
    # Get a specific image that looks interesting (e.g., index 0 or 1)
    # We iterate to find one with high contrast
    iterator = iter(val_loader)
    inp, gt = next(iterator)
    
    with torch.no_grad():
        out_c = cnn_model(inp)
        out_k = kan_model(inp)
    
    # Calculate Residuals (The Error Maps)
    # We amplify them by 5x so they are visible to the human eye
    res_c = torch.abs(out_c - gt) * 5 
    res_k = torch.abs(out_k - gt) * 5
    
    # Prepare Plot
    fig, ax = plt.subplots(2, 3, figsize=(15, 10))
    
    # Row 1: Images
    ax[0,0].imshow(inp[0,0], cmap='gray'); ax[0,0].set_title("Aliased Input (4x)")
    ax[0,1].imshow(out_c[0,0], cmap='gray'); ax[0,1].set_title("Standard U-Net")
    ax[0,2].imshow(out_k[0,0], cmap='gray'); ax[0,2].set_title("U-KAN (Ours)")
    
    # Row 2: Error Maps (The "Scientific Proof")
    ax[1,0].imshow(gt[0,0], cmap='gray'); ax[1,0].set_title("Ground Truth")
    
    # Use 'inferno' for errors - Black means perfect, Yellow means big error
    im1 = ax[1,1].imshow(res_c[0,0], cmap='inferno', vmin=0, vmax=1)
    ax[1,1].set_title("U-Net Error (Brighter = Worse)")
    
    im2 = ax[1,2].imshow(res_k[0,0], cmap='inferno', vmin=0, vmax=1)
    ax[1,2].set_title("U-KAN Error (Darker = Better)")
    
    # Remove axes
    for a in ax.ravel(): a.axis('off')
    
    plt.tight_layout()
    plt.savefig("paper_figure.png", dpi=300) # High DPI for report
    plt.show()
    
    # Print numerical gap
    err_c_sum = torch.sum(res_c).item()
    err_k_sum = torch.sum(res_k).item()
    print(f"Total Error Mass - CNN: {err_c_sum:.2f}")
    print(f"Total Error Mass - KAN: {err_k_sum:.2f}")
    print(f"Error Reduction: {((err_c_sum - err_k_sum)/err_c_sum)*100:.1f}%")

# Assuming you have the models from the previous run in memory:
# visualize_paper_figure(cnn, kan, val_loader)