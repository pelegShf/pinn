"""
Script to compare Model 1 and Model 2 architectures and loss functions visually
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from models import Model1, Model2
import config
import torch

def visualize_architectures():
    """Create visual comparison of the two model architectures"""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle('PINN Model Architectures Comparison', fontsize=16, fontweight='bold')

    # Model 1 visualization
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis('off')
    ax1.set_title('Model 1: Balanced Approach\n3 layers × 50 neurons',
                  fontsize=14, fontweight='bold', color='blue')

    # Draw Model 1 layers
    layer_x = [2, 3.5, 5, 6.5, 8]
    layer_names = ['Input\n(x, t)\n[2]', 'Hidden 1\n[50]', 'Hidden 2\n[50]',
                   'Hidden 3\n[50]', 'Output\nu\n[1]']
    layer_sizes = [0.3, 1.0, 1.0, 1.0, 0.3]

    for i, (x, name, size) in enumerate(zip(layer_x, layer_names, layer_sizes)):
        rect = patches.FancyBboxPatch((x-0.3, 5-size/2), 0.6, size,
                                      boxstyle="round,pad=0.1",
                                      edgecolor='blue', facecolor='lightblue',
                                      linewidth=2)
        ax1.add_patch(rect)
        ax1.text(x, 5, name, ha='center', va='center', fontsize=9, fontweight='bold')

        # Draw arrows
        if i < len(layer_x) - 1:
            ax1.annotate('', xy=(layer_x[i+1]-0.3, 5), xytext=(x+0.3, 5),
                        arrowprops=dict(arrowstyle='->', lw=2, color='blue'))
            if i > 0 and i < len(layer_x) - 1:
                ax1.text((x + layer_x[i+1])/2, 5.5, 'tanh',
                        ha='center', fontsize=8, style='italic', color='darkblue')

    # Model 1 stats
    stats1 = "Loss Function:\n"
    stats1 += "L = 1.0×L_data + 1.0×L_physics\n\n"
    stats1 += "Parameters: ~7,600\n"
    stats1 += "Training: Faster\n"
    stats1 += "Philosophy: Balanced"

    ax1.text(5, 2, stats1, ha='center', va='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # Model 2 visualization
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis('off')
    ax2.set_title('Model 2: Physics-First Approach\n4 layers × 100 neurons',
                  fontsize=14, fontweight='bold', color='red')

    # Draw Model 2 layers
    layer_x = [1.5, 3, 4.5, 6, 7.5, 9]
    layer_names = ['Input\n(x, t)\n[2]', 'Hidden 1\n[100]', 'Hidden 2\n[100]',
                   'Hidden 3\n[100]', 'Hidden 4\n[100]', 'Output\nu\n[1]']
    layer_sizes = [0.3, 1.5, 1.5, 1.5, 1.5, 0.3]

    for i, (x, name, size) in enumerate(zip(layer_x, layer_names, layer_sizes)):
        rect = patches.FancyBboxPatch((x-0.3, 5-size/2), 0.6, size,
                                      boxstyle="round,pad=0.1",
                                      edgecolor='red', facecolor='lightcoral',
                                      linewidth=2)
        ax2.add_patch(rect)
        ax2.text(x, 5, name, ha='center', va='center', fontsize=8, fontweight='bold')

        # Draw arrows
        if i < len(layer_x) - 1:
            ax2.annotate('', xy=(layer_x[i+1]-0.3, 5), xytext=(x+0.3, 5),
                        arrowprops=dict(arrowstyle='->', lw=2, color='red'))
            if i > 0 and i < len(layer_x) - 1:
                ax2.text((x + layer_x[i+1])/2, 5.5, 'tanh',
                        ha='center', fontsize=7, style='italic', color='darkred')

    # Model 2 stats
    stats2 = "Loss Function:\n"
    stats2 += "L = 0.5×L_data + 1.5×L_physics\n\n"
    stats2 += "Parameters: ~40,400\n"
    stats2 += "Training: Slower\n"
    stats2 += "Philosophy: Physics-focused"

    ax2.text(5, 2, stats2, ha='center', va='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    plt.savefig('./data/model_architectures_comparison.png', dpi=150, bbox_inches='tight')
    print("✓ Architecture comparison saved to: ./data/model_architectures_comparison.png")
    plt.show()


def visualize_loss_components():
    """Visualize the loss function components"""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Loss Function Components and Weighting', fontsize=16, fontweight='bold')

    # Loss weighting comparison
    ax1 = axes[0, 0]
    models = ['Model 1', 'Model 2']
    data_weights = [1.0, 0.5]
    physics_weights = [1.0, 1.5]

    x = np.arange(len(models))
    width = 0.35

    bars1 = ax1.bar(x - width/2, data_weights, width, label='λ_data',
                    color='skyblue', edgecolor='black', linewidth=1.5)
    bars2 = ax1.bar(x + width/2, physics_weights, width, label='λ_physics',
                    color='salmon', edgecolor='black', linewidth=1.5)

    ax1.set_ylabel('Weight Value', fontsize=12, fontweight='bold')
    ax1.set_title('Loss Weighting Comparison', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=11)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}', ha='center', va='bottom', fontweight='bold')

    # Loss components pie chart - Model 1
    ax2 = axes[0, 1]
    sizes1 = [1.0, 1.0]  # Equal weights
    labels1 = ['Data Loss\n(50%)', 'Physics Loss\n(50%)']
    colors1 = ['skyblue', 'salmon']
    explode1 = (0.05, 0.05)

    ax2.pie(sizes1, explode=explode1, labels=labels1, colors=colors1,
           autopct='%1.0f%%', shadow=True, startangle=90,
           textprops={'fontsize': 11, 'fontweight': 'bold'})
    ax2.set_title('Model 1: Balanced Loss\n(Equal weighting)',
                 fontsize=13, fontweight='bold')

    # Loss components pie chart - Model 2
    ax3 = axes[1, 0]
    sizes2 = [0.5, 1.5]  # Physics-focused
    labels2 = ['Data Loss\n(25%)', 'Physics Loss\n(75%)']
    colors2 = ['skyblue', 'salmon']
    explode2 = (0.05, 0.05)

    ax3.pie(sizes2, explode=explode2, labels=labels2, colors=colors2,
           autopct='%1.0f%%', shadow=True, startangle=90,
           textprops={'fontsize': 11, 'fontweight': 'bold'})
    ax3.set_title('Model 2: Physics-First Loss\n(Physics emphasized)',
                 fontsize=13, fontweight='bold')

    # Explanation text
    ax4 = axes[1, 1]
    ax4.axis('off')

    explanation = """
LOSS FUNCTION BREAKDOWN

Total Loss = λ_data × L_data + λ_physics × L_physics

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Data Loss (L_data):
  • Measures fit to training data
  • MSE(u_pred, u_true)
  • Supervised learning component

Physics Loss (L_physics):
  • Measures PDE satisfaction
  • mean((∂²u/∂t² - c²∂²u/∂x²)²)
  • Physics-informed component

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Model 1 Philosophy:
  "Trust data and physics equally"
  → Balanced approach

Model 2 Philosophy:
  "Trust physics more than data"
  → Better for noisy data
  → Better generalization
"""

    ax4.text(0.1, 0.9, explanation, transform=ax4.transAxes,
            fontsize=10, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

    plt.tight_layout()
    plt.savefig('./data/loss_functions_comparison.png', dpi=150, bbox_inches='tight')
    print("✓ Loss functions comparison saved to: ./data/loss_functions_comparison.png")
    plt.show()


def print_model_info():
    """Print detailed information about both models"""

    print("=" * 70)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 70)

    # Create models
    model1 = Model1(config.MODEL1_CONFIG)
    model2 = Model2(config.MODEL2_CONFIG)

    # Count parameters
    params1 = sum(p.numel() for p in model1.parameters())
    params2 = sum(p.numel() for p in model2.parameters())

    print("\nMODEL 1: Balanced Approach")
    print("-" * 70)
    print(f"Architecture: {config.MODEL1_CONFIG['hidden_layers']}")
    print(f"Activation: {config.MODEL1_CONFIG['activation']}")
    print(f"Total Parameters: {params1:,}")
    print(f"Learning Rate: {config.MODEL1_CONFIG['learning_rate']}")
    print(f"Batch Size: {config.MODEL1_CONFIG['batch_size']}")
    print(f"Epochs: {config.MODEL1_CONFIG['epochs']}")
    print(f"\nLoss Weighting:")
    print(f"  λ_data = 1.0 (50%)")
    print(f"  λ_physics = 1.0 (50%)")
    print(f"\nPhilosophy: Equal trust in data and physics")
    print(f"Best for: Fast training, simple problems, good data quality")

    print("\n" + "=" * 70)
    print("\nMODEL 2: Physics-First Approach")
    print("-" * 70)
    print(f"Architecture: {config.MODEL2_CONFIG['hidden_layers']}")
    print(f"Activation: {config.MODEL2_CONFIG['activation']}")
    print(f"Total Parameters: {params2:,}")
    print(f"Learning Rate: {config.MODEL2_CONFIG['learning_rate']}")
    print(f"Batch Size: {config.MODEL2_CONFIG['batch_size']}")
    print(f"Epochs: {config.MODEL2_CONFIG['epochs']}")
    print(f"\nLoss Weighting:")
    print(f"  λ_data = 0.5 (25%)")
    print(f"  λ_physics = 1.5 (75%)")
    print(f"\nPhilosophy: Trust physics more than data")
    print(f"Best for: Complex problems, noisy data, better generalization")

    print("\n" + "=" * 70)
    print("\nKEY DIFFERENCES")
    print("-" * 70)
    print(f"Model 2 has {params2/params1:.1f}× more parameters than Model 1")
    print(f"Model 2 emphasizes physics {1.5/1.0:.1f}× more than Model 1")
    print(f"Model 2 de-emphasizes data {0.5/1.0:.1f}× compared to Model 1")

    print("\n" + "=" * 70)
    print("\nWAVE EQUATION BEING SOLVED")
    print("-" * 70)
    print(f"PDE: ∂²u/∂t² = c² ∂²u/∂x²")
    print(f"Wave speed (c): {config.WAVE_SPEED}")
    print(f"Domain: x ∈ [{config.X_MIN}, {config.X_MAX}], t ∈ [{config.T_MIN}, {config.T_MAX}]")
    print(f"Initial condition: u(x, 0) = 0")
    print(f"Solution: u(x,t) = sin(kx) × sin(ωt)")
    print(f"  where k = 2π/λ = {2*np.pi/config.WAVELENGTH:.4f}")
    print(f"        ω = kc = {2*np.pi/config.WAVELENGTH * config.WAVE_SPEED:.4f}")

    print("\n" + "=" * 70)


def main():
    print("Comparing PINN Models...\n")

    # Print text summary
    print_model_info()

    print("\nGenerating visualizations...\n")

    # Create visualizations
    visualize_architectures()
    visualize_loss_components()

    print("\n" + "=" * 70)
    print("COMPARISON COMPLETE!")
    print("=" * 70)
    print("\nGenerated files:")
    print("  - ./data/model_architectures_comparison.png")
    print("  - ./data/loss_functions_comparison.png")
    print("\nFor detailed explanation, see: MODELS_EXPLAINED.md")


if __name__ == "__main__":
    main()
