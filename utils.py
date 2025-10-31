"""
Utility functions for visualization and analysis
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import config


def plot_training_history(history, save_path=None):
    """
    Plot training metrics (loss curves)

    Args:
        history: dictionary with training history
        save_path: path to save the figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Training History', fontsize=16)

    # Total loss
    axes[0, 0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0, 0].plot(history['test_loss'], label='Test Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_yscale('log')

    # Data loss
    axes[0, 1].plot(history['train_data_loss'], label='Data Loss', color='green', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Data Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_yscale('log')

    # Physics loss
    axes[1, 0].plot(
        history['train_physics_loss'], label='Physics Loss', color='red', linewidth=2
    )
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_title('Physics Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_yscale('log')

    # Loss components comparison
    axes[1, 1].plot(history['train_data_loss'], label='Data Loss', linewidth=2)
    axes[1, 1].plot(history['train_physics_loss'], label='Physics Loss', linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].set_title('Loss Components')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_yscale('log')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training history plot saved: {save_path}")

    plt.show()


def plot_predictions(model, grid_dataset, save_path=None):
    """
    Plot wave visualization (ground truth vs predictions)

    Args:
        model: trained PINN model
        grid_dataset: WaveDatasetGrid instance
        save_path: path to save the figure
    """
    model.eval()

    with torch.no_grad():
        # Get predictions on grid
        x = grid_dataset.x
        t = grid_dataset.t
        u_true = grid_dataset.u

        u_pred = model(x, t)

        # Reshape for plotting
        nx, nt = grid_dataset.nx, grid_dataset.nt
        X = grid_dataset.X
        T = grid_dataset.T
        U_true = u_true.cpu().numpy().reshape(nt, nx)
        U_pred = u_pred.cpu().numpy().reshape(nt, nx)

    # Create figure
    fig = plt.figure(figsize=(15, 10))

    # Plot ground truth
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    surf1 = ax1.plot_surface(X, T, U_true, cmap=cm.viridis, alpha=0.8)
    ax1.set_xlabel('x')
    ax1.set_ylabel('t')
    ax1.set_zlabel('u(x,t)')
    ax1.set_title('Ground Truth')
    fig.colorbar(surf1, ax=ax1, shrink=0.5)

    # Plot prediction
    ax2 = fig.add_subplot(2, 2, 2, projection='3d')
    surf2 = ax2.plot_surface(X, T, U_pred, cmap=cm.viridis, alpha=0.8)
    ax2.set_xlabel('x')
    ax2.set_ylabel('t')
    ax2.set_zlabel('u(x,t)')
    ax2.set_title('Model Prediction')
    fig.colorbar(surf2, ax=ax2, shrink=0.5)

    # Plot 2D heatmap - Ground truth
    ax3 = fig.add_subplot(2, 2, 3)
    im1 = ax3.contourf(X, T, U_true, levels=50, cmap='viridis')
    ax3.set_xlabel('x')
    ax3.set_ylabel('t')
    ax3.set_title('Ground Truth (2D)')
    fig.colorbar(im1, ax=ax3)

    # Plot 2D heatmap - Prediction
    ax4 = fig.add_subplot(2, 2, 4)
    im2 = ax4.contourf(X, T, U_pred, levels=50, cmap='viridis')
    ax4.set_xlabel('x')
    ax4.set_ylabel('t')
    ax4.set_title('Model Prediction (2D)')
    fig.colorbar(im2, ax=ax4)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Predictions plot saved: {save_path}")

    plt.show()


def plot_error_analysis(model, grid_dataset, save_path=None):
    """
    Plot error analysis (residuals, error distribution)

    Args:
        model: trained PINN model
        grid_dataset: WaveDatasetGrid instance
        save_path: path to save the figure
    """
    model.eval()

    with torch.no_grad():
        # Get predictions
        x = grid_dataset.x
        t = grid_dataset.t
        u_true = grid_dataset.u

        u_pred = model(x, t)

        # Compute error
        error = torch.abs(u_true - u_pred)

        # Reshape for plotting
        nx, nt = grid_dataset.nx, grid_dataset.nt
        X = grid_dataset.X
        T = grid_dataset.T
        U_true = u_true.cpu().numpy().reshape(nt, nx)
        U_pred = u_pred.cpu().numpy().reshape(nt, nx)
        Error = error.cpu().numpy().reshape(nt, nx)

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Error Analysis', fontsize=16)

    # Absolute error heatmap
    im1 = axes[0, 0].contourf(X, T, Error, levels=50, cmap='hot')
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('t')
    axes[0, 0].set_title('Absolute Error |u_true - u_pred|')
    fig.colorbar(im1, ax=axes[0, 0])

    # Relative error
    relative_error = np.abs((U_true - U_pred) / (U_true + 1e-8))
    im2 = axes[0, 1].contourf(X, T, relative_error, levels=50, cmap='hot')
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('t')
    axes[0, 1].set_title('Relative Error')
    fig.colorbar(im2, ax=axes[0, 1])

    # Error distribution
    axes[1, 0].hist(Error.flatten(), bins=50, alpha=0.7, color='blue', edgecolor='black')
    axes[1, 0].set_xlabel('Absolute Error')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Error Distribution')
    axes[1, 0].grid(True, alpha=0.3)

    # Error statistics
    mean_error = np.mean(Error)
    max_error = np.max(Error)
    l2_error = np.sqrt(np.mean(Error ** 2))

    # Comparison plot at fixed time slices
    time_slices = [0, nt // 3, 2 * nt // 3, nt - 1]
    for i, idx in enumerate(time_slices):
        t_val = T[idx, 0]
        axes[1, 1].plot(
            X[idx, :], U_true[idx, :], 'o-', label=f't={t_val:.2f} (true)', alpha=0.6
        )
        axes[1, 1].plot(
            X[idx, :], U_pred[idx, :], '--', label=f't={t_val:.2f} (pred)', alpha=0.8
        )

    axes[1, 1].set_xlabel('x')
    axes[1, 1].set_ylabel('u(x,t)')
    axes[1, 1].set_title('Snapshots at Different Times')
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)

    # Add error statistics as text
    textstr = f'Mean Error: {mean_error:.6f}\nMax Error: {max_error:.6f}\nL2 Error: {l2_error:.6f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    axes[1, 0].text(
        0.95, 0.95, textstr, transform=axes[1, 0].transAxes, fontsize=10,
        verticalalignment='top', horizontalalignment='right', bbox=props
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Error analysis plot saved: {save_path}")

    plt.show()

    # Print error statistics
    print("\nError Statistics:")
    print(f"Mean Absolute Error: {mean_error:.6f}")
    print(f"Max Absolute Error: {max_error:.6f}")
    print(f"L2 Error: {l2_error:.6f}")


def plot_wave_animation(model, grid_dataset, num_frames=50, save_path=None):
    """
    Create an animation of the wave evolution

    Args:
        model: trained PINN model
        grid_dataset: WaveDatasetGrid instance
        num_frames: number of frames in the animation
        save_path: path to save the animation
    """
    model.eval()

    with torch.no_grad():
        x = grid_dataset.x
        t = grid_dataset.t
        u_true = grid_dataset.u

        u_pred = model(x, t)

        nx, nt = grid_dataset.nx, grid_dataset.nt
        X = grid_dataset.X
        U_true = u_true.cpu().numpy().reshape(nt, nx)
        U_pred = u_pred.cpu().numpy().reshape(nt, nx)

    # Select time frames
    frame_indices = np.linspace(0, nt - 1, num_frames, dtype=int)

    fig, ax = plt.subplots(figsize=(10, 6))

    for idx in frame_indices:
        ax.clear()
        t_val = grid_dataset.T[idx, 0]

        ax.plot(X[idx, :], U_true[idx, :], 'o-', label='Ground Truth', linewidth=2, markersize=4)
        ax.plot(X[idx, :], U_pred[idx, :], 's--', label='Prediction', linewidth=2, markersize=4)

        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('u(x,t)', fontsize=12)
        ax.set_title(f'Wave Evolution at t = {t_val:.3f}', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([U_true.min() - 0.1, U_true.max() + 0.1])

        plt.pause(0.1)

    if save_path:
        print(f"Animation displayed (save functionality requires additional setup)")

    plt.show()


def plot_model1_training(history, save_path=None):
    """
    Plot training curves for Model 1

    Model 1 outputs scalar Z and minimizes: L = λ_z*mean(|Z|) + λ_norm*(L1_norm - target)²

    Args:
        history: dictionary with training history
        save_path: path to save the figure
    """
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    fig.suptitle('Model 1 Training: L = λ_z*mean(|Z|) + λ_norm*(L1_norm - target)²',
                 fontsize=16, fontweight='bold')

    epochs = np.arange(1, len(history['train_loss']) + 1)

    # Total loss (train and test)
    ax1 = axes[0, 0]
    ax1.plot(epochs, history['train_loss'], label='Train Loss (Total)', linewidth=2, color='blue')
    ax1.plot(epochs, history['test_loss'], label='Test Loss', linewidth=2, color='red')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Total Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')

    # Loss components
    ax2 = axes[0, 1]
    if 'loss_z' in history and 'loss_norm' in history:
        ax2.plot(epochs, history['loss_z'], label='Loss_z (mean|Z|)', linewidth=2, color='green')
        ax2.plot(epochs, history['loss_norm'], label='Loss_norm (deviation²)', linewidth=2, color='orange')
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Loss Component', fontsize=12)
        ax2.set_title('Loss Components', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
    else:
        ax2.text(0.5, 0.5, 'Loss components not tracked', ha='center', va='center',
                transform=ax2.transAxes, fontsize=12)

    # Z mean over time
    ax3 = axes[1, 0]
    ax3.plot(epochs, history['Z_mean'], linewidth=2, color='green')
    ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5, linewidth=1)
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Z_mean', fontsize=12)
    ax3.set_title('Mean of Z (should approach 0)', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # Z standard deviation
    ax4 = axes[1, 1]
    ax4.plot(epochs, history['Z_std'], linewidth=2, color='purple')
    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_ylabel('Z_std', fontsize=12)
    ax4.set_title('Standard Deviation of Z', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)

    # L1 norm of weights with target line
    ax5 = axes[2, 0]
    if 'l1_norm' in history:
        ax5.plot(epochs, history['l1_norm'], linewidth=2, color='red', label='Actual L1 Norm')
        # Add target norm as horizontal line
        if 'norm_deviation' in history:
            # Infer target from deviation: target = l1_norm - deviation
            target_norm = history['l1_norm'][-1] - history['norm_deviation'][-1]
            ax5.axhline(y=target_norm, color='green', linestyle='--', linewidth=2,
                       label=f'Target Norm ({target_norm:.1f})')
        ax5.set_xlabel('Epoch', fontsize=12)
        ax5.set_ylabel('sum(|weights|)', fontsize=12)
        ax5.set_title('L1 Norm vs Target (should converge to target)',
                     fontsize=14, fontweight='bold')
        ax5.legend(fontsize=10)
        ax5.grid(True, alpha=0.3)
    else:
        ax5.text(0.5, 0.5, 'L1 norm not tracked', ha='center', va='center',
                transform=ax5.transAxes, fontsize=12)

    # Z abs mean
    ax6 = axes[2, 1]
    ax6.plot(epochs, history['Z_abs_mean'], linewidth=2, color='orange')
    ax6.set_xlabel('Epoch', fontsize=12)
    ax6.set_ylabel('mean(|Z|)', fontsize=12)
    ax6.set_title('Mean Absolute Value of Z', fontsize=14, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    ax6.set_yscale('log')

    # Add final values as text
    final_text = f"Final Values:\n"
    final_text += f"Total Loss: {history['train_loss'][-1]:.6f}\n"
    if 'loss_z' in history:
        final_text += f"Loss_z: {history['loss_z'][-1]:.6f}\n"
    if 'loss_norm' in history:
        final_text += f"Loss_norm: {history['loss_norm'][-1]:.6f}\n"
    if 'l1_norm' in history:
        final_text += f"L1 norm: {history['l1_norm'][-1]:.4f}\n"
    if 'norm_deviation' in history:
        final_text += f"Deviation: {history['norm_deviation'][-1]:.4f}\n"
    final_text += f"Z_mean: {history['Z_mean'][-1]:.6f}\n"
    final_text += f"Z_std: {history['Z_std'][-1]:.6f}"

    ax6.text(0.98, 0.98, final_text, transform=ax6.transAxes,
             fontsize=10, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Model 1 training plot saved: {save_path}")

    plt.show()


def plot_model1_weights(history, save_path=None):
    """
    Plot weight vector evolution for Model 1

    Shows how the 100 weights from the last hidden layer to output Z evolve during training

    Args:
        history: dictionary with training history (must contain 'weights')
        save_path: path to save the figure
    """
    if 'weights' not in history or len(history['weights']) == 0:
        print("No weight history found!")
        return

    weights_array = np.array(history['weights'])  # Shape: (n_epochs, 100)
    n_epochs, n_weights = weights_array.shape

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('Model 1: Weight Vector Evolution (Last Layer → Z)', fontsize=16, fontweight='bold')

    # Heatmap of weights over time
    ax1 = plt.subplot(2, 2, 1)
    im1 = ax1.imshow(weights_array.T, aspect='auto', cmap='RdBu_r',
                     interpolation='nearest', vmin=-weights_array.max(), vmax=weights_array.max())
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Weight Index (0-99)', fontsize=12)
    ax1.set_title('Weight Values Heatmap', fontsize=14, fontweight='bold')
    cbar1 = fig.colorbar(im1, ax=ax1)
    cbar1.set_label('Weight Value', fontsize=10)

    # Individual weight trajectories (sample every 10th weight)
    ax2 = plt.subplot(2, 2, 2)
    epochs = np.arange(1, n_epochs + 1)
    for i in range(0, n_weights, 10):  # Plot every 10th weight
        ax2.plot(epochs, weights_array[:, i], alpha=0.7, linewidth=1)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Weight Value', fontsize=12)
    ax2.set_title('Weight Trajectories (every 10th weight)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)

    # Weight distribution at start vs end
    ax3 = plt.subplot(2, 2, 3)
    ax3.hist(weights_array[0, :], bins=30, alpha=0.6, label='Initial', color='blue', edgecolor='black')
    ax3.hist(weights_array[-1, :], bins=30, alpha=0.6, label='Final', color='red', edgecolor='black')
    ax3.set_xlabel('Weight Value', fontsize=12)
    ax3.set_ylabel('Frequency', fontsize=12)
    ax3.set_title('Weight Distribution: Initial vs Final', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)

    # Weight statistics over time
    ax4 = plt.subplot(2, 2, 4)
    weight_means = np.mean(weights_array, axis=1)
    weight_stds = np.std(weights_array, axis=1)
    weight_mins = np.min(weights_array, axis=1)
    weight_maxs = np.max(weights_array, axis=1)

    ax4.plot(epochs, weight_means, label='Mean', linewidth=2, color='blue')
    ax4.fill_between(epochs, weight_means - weight_stds, weight_means + weight_stds,
                      alpha=0.3, label='±1 std', color='blue')
    ax4.plot(epochs, weight_mins, label='Min', linewidth=1, linestyle='--', color='green')
    ax4.plot(epochs, weight_maxs, label='Max', linewidth=1, linestyle='--', color='red')
    ax4.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_ylabel('Weight Value', fontsize=12)
    ax4.set_title('Weight Statistics Over Training', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Model 1 weights plot saved: {save_path}")

    plt.show()

    # Print summary statistics
    print("\n" + "=" * 60)
    print("WEIGHT VECTOR SUMMARY")
    print("=" * 60)
    print(f"Number of weights: {n_weights}")
    print(f"Epochs tracked: {n_epochs}")
    print(f"\nInitial weights:")
    print(f"  Mean: {weights_array[0, :].mean():.6f}")
    print(f"  Std:  {weights_array[0, :].std():.6f}")
    print(f"  Min:  {weights_array[0, :].min():.6f}")
    print(f"  Max:  {weights_array[0, :].max():.6f}")
    print(f"\nFinal weights:")
    print(f"  Mean: {weights_array[-1, :].mean():.6f}")
    print(f"  Std:  {weights_array[-1, :].std():.6f}")
    print(f"  Min:  {weights_array[-1, :].min():.6f}")
    print(f"  Max:  {weights_array[-1, :].max():.6f}")
    print("=" * 60)


def plot_model2_training(history, save_path=None):
    """
    Plot training curves for Model 2

    Model 2 generates u_tilda(x,t) and is trained using frozen Model1 as validator
    Loss = mean(Z) from Model1(x, t, u_tilda), where we want Z -> 0

    Args:
        history: dictionary with training history
        save_path: path to save the figure
    """
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    fig.suptitle('Model 2 Training: Learned u_tilda using Model1 as Validator',
                 fontsize=16, fontweight='bold')

    epochs = np.arange(1, len(history['train_loss']) + 1)

    # Total loss (Z_mean from Model1)
    ax1 = axes[0, 0]
    ax1.plot(epochs, history['train_loss'], label='Train Loss (Z_mean)', linewidth=2, color='blue')
    ax1.plot(epochs, history['test_loss'], label='Test Loss', linewidth=2, color='red')
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.5, linewidth=1)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss (Z_mean)', fontsize=12)
    ax1.set_title('Loss: Z_mean from Model1 (should approach 0)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # MSE vs true solution (monitoring only)
    ax2 = axes[0, 1]
    if 'mse_vs_true' in history:
        ax2.plot(epochs, history['mse_vs_true'], linewidth=2, color='green')
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('MSE', fontsize=12)
        ax2.set_title('MSE(u_tilda, u_true) - Monitoring Only', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
    else:
        ax2.text(0.5, 0.5, 'MSE not tracked', ha='center', va='center',
                transform=ax2.transAxes, fontsize=12)

    # Z mean (from Model1 evaluation)
    ax3 = axes[1, 0]
    if 'Z_mean' in history:
        ax3.plot(epochs, history['Z_mean'], linewidth=2, color='purple')
        ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5, linewidth=1)
        ax3.set_xlabel('Epoch', fontsize=12)
        ax3.set_ylabel('Z_mean', fontsize=12)
        ax3.set_title('Mean of Z from Model1 (target: 0)', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'Z_mean not tracked', ha='center', va='center',
                transform=ax3.transAxes, fontsize=12)

    # Z standard deviation
    ax4 = axes[1, 1]
    if 'Z_std' in history:
        ax4.plot(epochs, history['Z_std'], linewidth=2, color='orange')
        ax4.set_xlabel('Epoch', fontsize=12)
        ax4.set_ylabel('Z_std', fontsize=12)
        ax4.set_title('Standard Deviation of Z from Model1', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'Z_std not tracked', ha='center', va='center',
                transform=ax4.transAxes, fontsize=12)

    # u_tilda mean
    ax5 = axes[2, 0]
    if 'u_tilda_mean' in history:
        ax5.plot(epochs, history['u_tilda_mean'], linewidth=2, color='teal')
        ax5.set_xlabel('Epoch', fontsize=12)
        ax5.set_ylabel('u_tilda_mean', fontsize=12)
        ax5.set_title('Mean of Generated u_tilda(x,t)', fontsize=14, fontweight='bold')
        ax5.grid(True, alpha=0.3)
    else:
        ax5.text(0.5, 0.5, 'u_tilda_mean not tracked', ha='center', va='center',
                transform=ax5.transAxes, fontsize=12)

    # u_tilda standard deviation
    ax6 = axes[2, 1]
    if 'u_tilda_std' in history:
        ax6.plot(epochs, history['u_tilda_std'], linewidth=2, color='brown')
        ax6.set_xlabel('Epoch', fontsize=12)
        ax6.set_ylabel('u_tilda_std', fontsize=12)
        ax6.set_title('Std Dev of Generated u_tilda(x,t)', fontsize=14, fontweight='bold')
        ax6.grid(True, alpha=0.3)
    else:
        ax6.text(0.5, 0.5, 'u_tilda_std not tracked', ha='center', va='center',
                transform=ax6.transAxes, fontsize=12)

    # Add final values as text
    final_text = f"Final Values:\n"
    final_text += f"Loss (Z_mean): {history['train_loss'][-1]:.6f}\n"
    if 'mse_vs_true' in history:
        final_text += f"MSE vs True: {history['mse_vs_true'][-1]:.6f}\n"
    if 'Z_mean' in history:
        final_text += f"Z_mean: {history['Z_mean'][-1]:.6f}\n"
    if 'Z_std' in history:
        final_text += f"Z_std: {history['Z_std'][-1]:.6f}\n"
    if 'u_tilda_mean' in history:
        final_text += f"u_tilda_mean: {history['u_tilda_mean'][-1]:.6f}\n"
    if 'u_tilda_std' in history:
        final_text += f"u_tilda_std: {history['u_tilda_std'][-1]:.6f}"

    ax6.text(0.98, 0.98, final_text, transform=ax6.transAxes,
             fontsize=10, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Model 2 training plot saved: {save_path}")

    plt.show()


if __name__ == "__main__":
    # Test plotting functions with dummy data
    print("Testing plotting functions...")

    # Create dummy history for Model 2
    history = {
        'train_loss': np.random.exponential(1, 100)[::-1].cumsum() / 100,
        'train_data_loss': np.random.exponential(1, 100)[::-1].cumsum() / 100,
        'train_physics_loss': np.random.exponential(1, 100)[::-1].cumsum() / 100,
        'test_loss': np.random.exponential(1, 100)[::-1].cumsum() / 100,
    }

    plot_training_history(history)
    print("Plotting test completed!")
