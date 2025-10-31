"""
Model 1 Interpretation Script

Analyzes the relationship between Model 1's outputs (z1, z2) and 
the true derivatives of the wave equation (u_tt, u_xx).

This script:
1. Loads trained Model 1 from checkpoint
2. Loads dataset triplets (x, t, u(x,t))
3. Computes ground truth derivatives u_tt and u_xx
4. Runs inference to get (z1, z2) predictions
5. Determines which z corresponds to which derivative
6. Generates comprehensive comparison plots
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from pathlib import Path
import argparse

import config
from models import Model1
from dataset import get_dataloaders, WaveDatasetGrid


def compute_true_derivatives_analytical(x, t, c=1.0):
    """
    Compute second derivatives analytically for the standing wave solution:
    u(x,t) = sin(kx) * sin(ωt)
    
    Derivatives:
    ∂²u/∂x² = -k² * sin(kx) * sin(ωt) = -k² * u
    ∂²u/∂t² = -ω² * sin(kx) * sin(ωt) = -ω² * u
    
    Args:
        x: spatial coordinate (numpy array or tensor)
        t: temporal coordinate (numpy array or tensor)
        c: wave speed (default: 1.0)
    
    Returns:
        u_xx: ∂²u/∂x² 
        u_tt: ∂²u/∂t²
    """
    # Convert to numpy if needed
    if torch.is_tensor(x):
        x = x.cpu().numpy()
    if torch.is_tensor(t):
        t = t.cpu().numpy()
    
    # Wave parameters
    k = 2 * np.pi / config.WAVELENGTH  # wave number
    omega = k * c  # angular frequency
    
    # u(x,t) = sin(kx) * sin(ωt)
    u = np.sin(k * x) * np.sin(omega * t)
    
    # Second derivatives
    u_xx = -k**2 * u  # ∂²u/∂x²
    u_tt = -omega**2 * u  # ∂²u/∂t²
    
    # Convert back to torch tensors
    u_xx = torch.FloatTensor(u_xx).to(config.DEVICE)
    u_tt = torch.FloatTensor(u_tt).to(config.DEVICE)
    
    return u_xx, u_tt


def analyze_model(checkpoint_path, dataset_size='test'):
    """
    Analyze Model 1 outputs vs true derivatives
    
    Args:
        checkpoint_path: path to model checkpoint
        dataset_size: 'train', 'test', or 'grid' for which dataset to use
    
    Returns:
        results: dictionary with all analysis results
    """
    print("=" * 70)
    print("MODEL 1 INTERPRETATION ANALYSIS")
    print("=" * 70)
    
    # Load model
    print(f"\nLoading model from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE, weights_only=False)
    model = Model1(config.MODEL1_CONFIG).to(config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("Model loaded successfully!")
    
    # Load dataset
    print("\nLoading dataset...")
    train_loader, test_loader, grid_dataset = get_dataloaders(
        load_from_disk=True,
        save_to_disk=False
    )
    
    if dataset_size == 'train':
        loader = train_loader
        print(f"Using training dataset: {len(train_loader.dataset)} samples")
    elif dataset_size == 'test':
        loader = test_loader
        print(f"Using test dataset: {len(test_loader.dataset)} samples")
    else:  # grid
        # Use grid dataset for spatial visualization
        print("Using grid dataset for visualization")
        x = grid_dataset.x
        t = grid_dataset.t
        u = grid_dataset.u
        
        # Compute derivatives
        print("Computing ground truth derivatives...")
        u_xx, u_tt = compute_true_derivatives_analytical(x, t, c=config.WAVE_SPEED)
        
        # Get model predictions
        print("Running model inference...")
        with torch.no_grad():
            Z = model(x, t, u)
            z1 = Z[:, 0:1]
            z2 = Z[:, 1:2]
        
        return {
            'x': x.detach().cpu().numpy(),
            't': t.detach().cpu().numpy(),
            'u': u.detach().cpu().numpy(),
            'u_xx': u_xx.detach().cpu().numpy(),
            'u_tt': u_tt.detach().cpu().numpy(),
            'z1': z1.cpu().numpy(),
            'z2': z2.cpu().numpy(),
            'nx': grid_dataset.nx,
            'nt': grid_dataset.nt,
            'X': grid_dataset.X,
            'T': grid_dataset.T
        }
    
    # Collect data from loader
    all_x, all_t, all_u = [], [], []
    all_z1, all_z2 = [], []
    all_u_xx, all_u_tt = [], []
    
    print("Processing batches...")
    for batch_idx, (x, t, u) in enumerate(loader):
        x = x.to(config.DEVICE)
        t = t.to(config.DEVICE)
        u = u.to(config.DEVICE)
        
        # Compute derivatives
        u_xx, u_tt = compute_true_derivatives_analytical(x, t, c=config.WAVE_SPEED)
        
        # Get model predictions
        with torch.no_grad():
            Z = model(x, t, u)
            z1 = Z[:, 0:1]
            z2 = Z[:, 1:2]
        
        # Store results
        all_x.append(x.detach().cpu().numpy())
        all_t.append(t.detach().cpu().numpy())
        all_u.append(u.detach().cpu().numpy())
        all_u_xx.append(u_xx.detach().cpu().numpy())
        all_u_tt.append(u_tt.detach().cpu().numpy())
        all_z1.append(z1.cpu().numpy())
        all_z2.append(z2.cpu().numpy())
        
        if (batch_idx + 1) % 10 == 0:
            print(f"  Processed {batch_idx + 1}/{len(loader)} batches")
    
    # Concatenate all batches
    results = {
        'x': np.concatenate(all_x, axis=0),
        't': np.concatenate(all_t, axis=0),
        'u': np.concatenate(all_u, axis=0),
        'u_xx': np.concatenate(all_u_xx, axis=0),
        'u_tt': np.concatenate(all_u_tt, axis=0),
        'z1': np.concatenate(all_z1, axis=0),
        'z2': np.concatenate(all_z2, axis=0),
    }
    
    print(f"Total samples processed: {len(results['x'])}")
    
    return results


def determine_correspondence(results):
    """
    Determine which z corresponds to which derivative
    
    Computes correlations and MSE between:
    - z1 vs u_xx and z1 vs u_tt
    - z2 vs u_xx and z2 vs u_tt
    
    Args:
        results: dictionary with z1, z2, u_xx, u_tt
    
    Returns:
        correspondence: dictionary with matching information
    """
    z1 = results['z1'].flatten()
    z2 = results['z2'].flatten()
    u_xx = results['u_xx'].flatten()
    u_tt = results['u_tt'].flatten()
    
    # Compute correlations
    corr_z1_uxx = np.corrcoef(z1, u_xx)[0, 1]
    corr_z1_utt = np.corrcoef(z1, u_tt)[0, 1]
    corr_z2_uxx = np.corrcoef(z2, u_xx)[0, 1]
    corr_z2_utt = np.corrcoef(z2, u_tt)[0, 1]
    
    # Compute MSE
    mse_z1_uxx = np.mean((z1 - u_xx) ** 2)
    mse_z1_utt = np.mean((z1 - u_tt) ** 2)
    mse_z2_uxx = np.mean((z2 - u_xx) ** 2)
    mse_z2_utt = np.mean((z2 - u_tt) ** 2)
    
    # Compute MAE
    mae_z1_uxx = np.mean(np.abs(z1 - u_xx))
    mae_z1_utt = np.mean(np.abs(z1 - u_tt))
    mae_z2_uxx = np.mean(np.abs(z2 - u_xx))
    mae_z2_utt = np.mean(np.abs(z2 - u_tt))
    
    correspondence = {
        'correlations': {
            'z1_vs_uxx': corr_z1_uxx,
            'z1_vs_utt': corr_z1_utt,
            'z2_vs_uxx': corr_z2_uxx,
            'z2_vs_utt': corr_z2_utt,
        },
        'mse': {
            'z1_vs_uxx': mse_z1_uxx,
            'z1_vs_utt': mse_z1_utt,
            'z2_vs_uxx': mse_z2_uxx,
            'z2_vs_utt': mse_z2_utt,
        },
        'mae': {
            'z1_vs_uxx': mae_z1_uxx,
            'z1_vs_utt': mae_z1_utt,
            'z2_vs_uxx': mae_z2_uxx,
            'z2_vs_utt': mae_z2_utt,
        }
    }
    
    # Determine best match based on correlation
    if abs(corr_z1_uxx) > abs(corr_z1_utt):
        correspondence['z1_matches'] = 'u_xx'
        correspondence['z2_matches'] = 'u_tt'
    else:
        correspondence['z1_matches'] = 'u_tt'
        correspondence['z2_matches'] = 'u_xx'
    
    return correspondence


def plot_analysis(results, correspondence, save_dir='./results/interpretation'):
    """
    Create comprehensive visualization plots
    
    Args:
        results: dictionary with all data
        correspondence: dictionary with matching information
        save_dir: directory to save plots
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nGenerating plots...")
    
    # Plot 1: Correlation scatter plots
    plot_scatter_comparison(results, correspondence, save_dir)
    
    # Plot 2: Statistical comparison
    plot_statistical_comparison(results, correspondence, save_dir)
    
    # Plot 3: Heatmaps (if grid data available)
    if 'X' in results and 'T' in results:
        plot_heatmaps(results, correspondence, save_dir)
    
    # Plot 4: Error distributions
    plot_error_distributions(results, correspondence, save_dir)
    
    print(f"\nAll plots saved to: {save_dir}")


def plot_scatter_comparison(results, correspondence, save_dir):
    """Plot scatter plots comparing z1, z2 with u_xx, u_tt"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Model Outputs vs True Derivatives: Scatter Comparison', 
                 fontsize=16, fontweight='bold')
    
    z1 = results['z1'].flatten()
    z2 = results['z2'].flatten()
    u_xx = results['u_xx'].flatten()
    u_tt = results['u_tt'].flatten()
    
    corr = correspondence['correlations']
    
    # z1 vs u_xx
    ax = axes[0, 0]
    ax.scatter(u_xx, z1, alpha=0.3, s=1)
    ax.plot([u_xx.min(), u_xx.max()], [u_xx.min(), u_xx.max()], 
            'r--', linewidth=2, label='Perfect match')
    ax.set_xlabel('u_xx (true)', fontsize=12)
    ax.set_ylabel('z1 (predicted)', fontsize=12)
    ax.set_title(f'z1 vs u_xx\nCorr: {corr["z1_vs_uxx"]:.4f}', 
                 fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # z1 vs u_tt
    ax = axes[0, 1]
    ax.scatter(u_tt, z1, alpha=0.3, s=1, color='orange')
    ax.plot([u_tt.min(), u_tt.max()], [u_tt.min(), u_tt.max()], 
            'r--', linewidth=2, label='Perfect match')
    ax.set_xlabel('u_tt (true)', fontsize=12)
    ax.set_ylabel('z1 (predicted)', fontsize=12)
    ax.set_title(f'z1 vs u_tt\nCorr: {corr["z1_vs_utt"]:.4f}', 
                 fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # z2 vs u_xx
    ax = axes[1, 0]
    ax.scatter(u_xx, z2, alpha=0.3, s=1, color='green')
    ax.plot([u_xx.min(), u_xx.max()], [u_xx.min(), u_xx.max()], 
            'r--', linewidth=2, label='Perfect match')
    ax.set_xlabel('u_xx (true)', fontsize=12)
    ax.set_ylabel('z2 (predicted)', fontsize=12)
    ax.set_title(f'z2 vs u_xx\nCorr: {corr["z2_vs_uxx"]:.4f}', 
                 fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # z2 vs u_tt
    ax = axes[1, 1]
    ax.scatter(u_tt, z2, alpha=0.3, s=1, color='purple')
    ax.plot([u_tt.min(), u_tt.max()], [u_tt.min(), u_tt.max()], 
            'r--', linewidth=2, label='Perfect match')
    ax.set_xlabel('u_tt (true)', fontsize=12)
    ax.set_ylabel('z2 (predicted)', fontsize=12)
    ax.set_title(f'z2 vs u_tt\nCorr: {corr["z2_vs_utt"]:.4f}', 
                 fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'scatter_comparison.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: scatter_comparison.png")
    plt.close()


def plot_statistical_comparison(results, correspondence, save_dir):
    """Plot statistical comparison metrics"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Statistical Comparison: Model Outputs vs True Derivatives', 
                 fontsize=16, fontweight='bold')
    
    corr = correspondence['correlations']
    mse = correspondence['mse']
    mae = correspondence['mae']
    
    # Correlation comparison
    ax = axes[0, 0]
    comparisons = ['z1 vs u_xx', 'z1 vs u_tt', 'z2 vs u_xx', 'z2 vs u_tt']
    corr_values = [corr['z1_vs_uxx'], corr['z1_vs_utt'], 
                   corr['z2_vs_uxx'], corr['z2_vs_utt']]
    colors = ['blue', 'orange', 'green', 'purple']
    bars = ax.bar(range(len(comparisons)), corr_values, color=colors, alpha=0.7)
    ax.set_xticks(range(len(comparisons)))
    ax.set_xticklabels(comparisons, rotation=45, ha='right')
    ax.set_ylabel('Correlation Coefficient', fontsize=12)
    ax.set_title('Correlation Comparison', fontsize=12, fontweight='bold')
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, corr_values)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}', ha='center', va='bottom' if val > 0 else 'top',
                fontsize=9)
    
    # MSE comparison
    ax = axes[0, 1]
    mse_values = [mse['z1_vs_uxx'], mse['z1_vs_utt'], 
                  mse['z2_vs_uxx'], mse['z2_vs_utt']]
    bars = ax.bar(range(len(comparisons)), mse_values, color=colors, alpha=0.7)
    ax.set_xticks(range(len(comparisons)))
    ax.set_xticklabels(comparisons, rotation=45, ha='right')
    ax.set_ylabel('Mean Squared Error', fontsize=12)
    ax.set_title('MSE Comparison', fontsize=12, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')
    
    # MAE comparison
    ax = axes[1, 0]
    mae_values = [mae['z1_vs_uxx'], mae['z1_vs_utt'], 
                  mae['z2_vs_uxx'], mae['z2_vs_utt']]
    bars = ax.bar(range(len(comparisons)), mae_values, color=colors, alpha=0.7)
    ax.set_xticks(range(len(comparisons)))
    ax.set_xticklabels(comparisons, rotation=45, ha='right')
    ax.set_ylabel('Mean Absolute Error', fontsize=12)
    ax.set_title('MAE Comparison', fontsize=12, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Best match summary
    ax = axes[1, 1]
    ax.axis('off')
    
    # Create summary text
    summary_text = "BEST CORRESPONDENCE:\n\n"
    summary_text += f"z1 matches: {correspondence['z1_matches']}\n"
    summary_text += f"z2 matches: {correspondence['z2_matches']}\n\n"
    summary_text += "Correlations:\n"
    summary_text += f"  z1 ↔ {correspondence['z1_matches']}: "
    if correspondence['z1_matches'] == 'u_xx':
        summary_text += f"{corr['z1_vs_uxx']:.4f}\n"
    else:
        summary_text += f"{corr['z1_vs_utt']:.4f}\n"
    summary_text += f"  z2 ↔ {correspondence['z2_matches']}: "
    if correspondence['z2_matches'] == 'u_xx':
        summary_text += f"{corr['z2_vs_uxx']:.4f}\n"
    else:
        summary_text += f"{corr['z2_vs_utt']:.4f}\n"
    
    ax.text(0.1, 0.5, summary_text, transform=ax.transAxes,
            fontsize=14, verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
            family='monospace')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'statistical_comparison.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: statistical_comparison.png")
    plt.close()


def plot_heatmaps(results, correspondence, save_dir):
    """Plot heatmaps for grid data"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Spatial Distribution: Model Outputs vs True Derivatives', 
                 fontsize=16, fontweight='bold')
    
    nx = results['nx']
    nt = results['nt']
    X = results['X']
    T = results['T']
    
    # Reshape data to grid
    z1_grid = results['z1'].reshape(nt, nx)
    z2_grid = results['z2'].reshape(nt, nx)
    u_xx_grid = results['u_xx'].reshape(nt, nx)
    u_tt_grid = results['u_tt'].reshape(nt, nx)
    
    # Determine common color scale
    vmin = min(z1_grid.min(), z2_grid.min(), u_xx_grid.min(), u_tt_grid.min())
    vmax = max(z1_grid.max(), z2_grid.max(), u_xx_grid.max(), u_tt_grid.max())
    
    # Plot z1
    ax = axes[0, 0]
    im = ax.contourf(X, T, z1_grid, levels=50, cmap='RdBu_r', vmin=vmin, vmax=vmax)
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('t', fontsize=12)
    ax.set_title(f'z1 (matches {correspondence["z1_matches"]})', 
                 fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax)
    
    # Plot z2
    ax = axes[0, 1]
    im = ax.contourf(X, T, z2_grid, levels=50, cmap='RdBu_r', vmin=vmin, vmax=vmax)
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('t', fontsize=12)
    ax.set_title(f'z2 (matches {correspondence["z2_matches"]})', 
                 fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax)
    
    # Plot u_xx
    ax = axes[1, 0]
    im = ax.contourf(X, T, u_xx_grid, levels=50, cmap='RdBu_r', vmin=vmin, vmax=vmax)
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('t', fontsize=12)
    ax.set_title('u_xx (true)', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax)
    
    # Plot u_tt
    ax = axes[1, 1]
    im = ax.contourf(X, T, u_tt_grid, levels=50, cmap='RdBu_r', vmin=vmin, vmax=vmax)
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('t', fontsize=12)
    ax.set_title('u_tt (true)', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'heatmaps.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: heatmaps.png")
    plt.close()


def plot_error_distributions(results, correspondence, save_dir):
    """Plot error distributions for best matches"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Error Analysis: Best Correspondence', 
                 fontsize=16, fontweight='bold')
    
    z1 = results['z1'].flatten()
    z2 = results['z2'].flatten()
    u_xx = results['u_xx'].flatten()
    u_tt = results['u_tt'].flatten()
    
    # Determine errors for best matches
    if correspondence['z1_matches'] == 'u_xx':
        error_z1 = z1 - u_xx
        label_z1 = 'z1 - u_xx'
    else:
        error_z1 = z1 - u_tt
        label_z1 = 'z1 - u_tt'
    
    if correspondence['z2_matches'] == 'u_xx':
        error_z2 = z2 - u_xx
        label_z2 = 'z2 - u_xx'
    else:
        error_z2 = z2 - u_tt
        label_z2 = 'z2 - u_tt'
    
    # Error histogram for z1
    ax = axes[0, 0]
    ax.hist(error_z1, bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax.axvline(0, color='r', linestyle='--', linewidth=2)
    ax.set_xlabel('Error', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(f'{label_z1}: Error Distribution', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Error stats for z1
    mean_err = np.mean(error_z1)
    std_err = np.std(error_z1)
    ax.text(0.02, 0.98, f'Mean: {mean_err:.6f}\nStd: {std_err:.6f}',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Error histogram for z2
    ax = axes[0, 1]
    ax.hist(error_z2, bins=50, alpha=0.7, color='green', edgecolor='black')
    ax.axvline(0, color='r', linestyle='--', linewidth=2)
    ax.set_xlabel('Error', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(f'{label_z2}: Error Distribution', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Error stats for z2
    mean_err = np.mean(error_z2)
    std_err = np.std(error_z2)
    ax.text(0.02, 0.98, f'Mean: {mean_err:.6f}\nStd: {std_err:.6f}',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Absolute error comparison
    ax = axes[1, 0]
    abs_err_z1 = np.abs(error_z1)
    abs_err_z2 = np.abs(error_z2)
    ax.hist(abs_err_z1, bins=50, alpha=0.5, color='blue', 
            edgecolor='black', label=label_z1)
    ax.hist(abs_err_z2, bins=50, alpha=0.5, color='green', 
            edgecolor='black', label=label_z2)
    ax.set_xlabel('Absolute Error', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Absolute Error Comparison', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # Q-Q plot or residual plot
    ax = axes[1, 1]
    ax.scatter(range(len(error_z1[::10])), error_z1[::10], 
               alpha=0.5, s=1, color='blue', label=label_z1)
    ax.scatter(range(len(error_z2[::10])), error_z2[::10], 
               alpha=0.5, s=1, color='green', label=label_z2)
    ax.axhline(0, color='r', linestyle='--', linewidth=2)
    ax.set_xlabel('Sample Index (subsampled)', fontsize=12)
    ax.set_ylabel('Error', fontsize=12)
    ax.set_title('Error vs Sample Index', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'error_distributions.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: error_distributions.png")
    plt.close()


def print_summary(correspondence):
    """Print summary of analysis"""
    print("\n" + "=" * 70)
    print("ANALYSIS SUMMARY")
    print("=" * 70)
    
    print(f"\nBEST CORRESPONDENCE:")
    print(f"  z1 ↔ {correspondence['z1_matches']}")
    print(f"  z2 ↔ {correspondence['z2_matches']}")
    
    print(f"\nCORRELATIONS:")
    for key, val in correspondence['correlations'].items():
        print(f"  {key}: {val:.6f}")
    
    print(f"\nMEAN SQUARED ERROR:")
    for key, val in correspondence['mse'].items():
        print(f"  {key}: {val:.6e}")
    
    print(f"\nMEAN ABSOLUTE ERROR:")
    for key, val in correspondence['mae'].items():
        print(f"  {key}: {val:.6e}")
    
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description='Model 1 Interpretation Analysis')
    parser.add_argument('--checkpoint', type=str, 
                        default='./checkpoints/model1_final_epoch_2000.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--dataset', type=str, choices=['train', 'test', 'grid'],
                        default='grid',
                        help='Which dataset to use for analysis')
    parser.add_argument('--save_dir', type=str, 
                        default='./results/interpretation',
                        help='Directory to save results')
    
    args = parser.parse_args()
    
    # Check if checkpoint exists
    if not Path(args.checkpoint).exists():
        print(f"ERROR: Checkpoint not found at {args.checkpoint}")
        print("Please train Model 1 first or specify correct checkpoint path.")
        return
    
    # Run analysis
    results = analyze_model(args.checkpoint, args.dataset)
    
    # Determine correspondence
    print("\nAnalyzing correspondence between outputs and derivatives...")
    correspondence = determine_correspondence(results)
    
    # Print summary
    print_summary(correspondence)
    
    # Generate plots
    plot_analysis(results, correspondence, args.save_dir)
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()

