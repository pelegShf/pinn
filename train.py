"""
Training script for 1D Wave Equation PINNs
Contains separate training loops for Model 1 and Model 2
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from pathlib import Path
from datetime import datetime

import config
from models import Model1, Model2, wave_equation_residual
from dataset import get_dataloaders
from utils import plot_training_history, plot_predictions, plot_error_analysis


def loss_function_model1(model, x, t, u_true):
    """
    Loss function for Model 1

    Model 1 takes triplet inputs (x, t, u) and outputs (z1, z2)
    Loss: L = λ_z * mean(|z1 - z2|) + λ_norm * (output_L1_norm - target_norm)²

    The target norm term encourages the output vector (z1, z2) to have
    an L1 norm near the target, preventing collapse to zero and unbounded growth.

    Args:
        model: Model1 instance
        x: spatial coordinate (batch_size, 1)
        t: temporal coordinate (batch_size, 1)
        u_true: ground truth solution (batch_size, 1)

    Returns:
        loss: total loss
        loss_dict: dictionary with loss components
    """
    # Forward pass with triplet input (x, t, u)
    Z = model(x, t, u_true)  # Shape: (batch_size, 2)
    
    # Split into z1 and z2
    z1 = Z[:, 0:1]  # Shape: (batch_size, 1)
    z2 = Z[:, 1:2]  # Shape: (batch_size, 1)

    # Term 1: mean absolute value of (z1 - z2)
    loss_z = torch.mean(torch.abs(z1 - z2))

    # Term 2: Target norm regularization for output vector (z1, z2)
    # Compute L1 norm of output: |z1| + |z2| for each sample
    output_l1_norm = torch.abs(z1) + torch.abs(z2)  # Shape: (batch_size, 1)
    mean_output_l1_norm = torch.mean(output_l1_norm)  # Scalar

    # Get hyperparameters from config
    lambda_z = config.MODEL1_CONFIG.get('lambda_z', 1.0)
    lambda_norm = config.MODEL1_CONFIG.get('lambda_norm', 0.01)
    target_norm = config.MODEL1_CONFIG.get('target_norm', 50.0)

    # Target norm penalty: (mean_output_l1_norm - target_norm)²
    # This creates a "sweet spot" at target_norm
    norm_deviation = mean_output_l1_norm - target_norm
    loss_norm = norm_deviation ** 2

    # Total loss
    loss = lambda_z * loss_z + lambda_norm * loss_norm

    loss_dict = {
        'total': loss.item(),
        'loss_z': loss_z.item(),
        'loss_norm': loss_norm.item(),
        'output_l1_norm': mean_output_l1_norm.item(),
        'norm_deviation': norm_deviation.item(),  # Signed difference from target
        'z1_mean': torch.mean(z1).item(),
        'z1_std': torch.std(z1).item(),
        'z2_mean': torch.mean(z2).item(),
        'z2_std': torch.std(z2).item(),
        'z1_z2_diff_mean': torch.mean(z1 - z2).item(),
        'z1_z2_diff_abs_mean': loss_z.item(),
    }

    return loss, loss_dict


def loss_function_model2(model2, x, t, u_true, model1_frozen):
    """
    Loss function for Model 2

    Model 2 predicts u_tilda(x,t) and is trained to make Model 1's output Z close to 0

    Strategy:
    1. Model2 generates u_tilda(x, t)
    2. Create triplets (x, t, u_tilda)
    3. Pass to frozen pretrained Model1 -> get Z
    4. Loss = mean(Z) (we want Z -> 0, indicating u_tilda is close to true solution)

    Args:
        model2: Model2 instance (trainable)
        x: spatial coordinate (batch_size, 1)
        t: temporal coordinate (batch_size, 1)
        u_true: ground truth solution (batch_size, 1) - used for metrics only
        model1_frozen: Pretrained Model1 (frozen, used as validator)

    Returns:
        loss: total loss
        loss_dict: dictionary with loss components
    """
    # Model2 generates u_tilda
    u_tilda = model2(x, t)

    # Pass triplet (x, t, u_tilda) to frozen Model1
    with torch.no_grad():
        model1_frozen.eval()

    Z = model1_frozen(x, t, u_tilda)

    # Loss: mean of Z (we want Z -> 0)
    # Note: We use mean(Z) not mean(|Z|) because Model1 was trained with mean(|Z|)
    # and should output values close to 0 for correct triplets
    loss = torch.mean(Z)

    # Compute MSE with true solution for monitoring (not used in backprop)
    with torch.no_grad():
        mse = nn.MSELoss()(u_tilda, u_true)

    loss_dict = {
        'total': loss.item(),
        'Z_mean': torch.mean(Z).item(),
        'Z_std': torch.std(Z).item(),
        'mse_vs_true': mse.item(),  # For monitoring convergence
        'u_tilda_mean': torch.mean(u_tilda).item(),
        'u_tilda_std': torch.std(u_tilda).item(),
    }

    return loss, loss_dict


def train_model(
    model,
    train_loader,
    test_loader,
    loss_function,
    optimizer,
    epochs,
    model_name="model",
    track_weights=False,
    model1_frozen=None,
):
    """
    Generic training loop

    Args:
        model: PINN model
        train_loader: training dataloader
        test_loader: test dataloader
        loss_function: custom loss function
        optimizer: optimizer
        epochs: number of epochs
        model_name: name for saving checkpoints
        track_weights: if True, track last layer weights (for Model1)
        model1_frozen: Pretrained frozen Model1 (required for Model2 training)

    Returns:
        history: training history dictionary
    """
    history = {
        'train_loss': [],
        'test_loss': [],
    }

    # For Model1, track additional metrics
    if model_name == "model1":
        history['z1_mean'] = []
        history['z1_std'] = []
        history['z2_mean'] = []
        history['z2_std'] = []
        history['z1_z2_diff_mean'] = []
        history['z1_z2_diff_abs_mean'] = []
        history['loss_z'] = []  # Mean |z1 - z2| component
        history['loss_norm'] = []  # Target norm penalty component
        history['output_l1_norm'] = []  # L1 norm of output vector (|z1| + |z2|)
        history['norm_deviation'] = []  # Signed difference from target
        if track_weights:
            history['weights'] = []  # Track weight matrix from last layer

    # For Model2, track Z metrics and MSE vs true
    if model_name == "model2":
        history['Z_mean'] = []
        history['Z_std'] = []
        history['mse_vs_true'] = []
        history['u_tilda_mean'] = []
        history['u_tilda_std'] = []

    print(f"\nTraining {model_name}...")
    print("=" * 60)

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_losses = []
        epoch_loss_dicts = []

        for batch_idx, (x, t, u) in enumerate(train_loader):
            x, t, u = x.to(config.DEVICE), t.to(config.DEVICE), u.to(config.DEVICE)

            # Zero gradients
            optimizer.zero_grad()

            # Compute loss
            if model_name == "model2" and model1_frozen is not None:
                loss, loss_dict = loss_function(model, x, t, u, model1_frozen)
            else:
                loss, loss_dict = loss_function(model, x, t, u)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Record losses
            train_losses.append(loss_dict['total'])
            epoch_loss_dicts.append(loss_dict)

        # Average training losses
        avg_train_loss = np.mean(train_losses)

        # Compute averages for all loss components
        avg_loss_dict = {}
        for key in epoch_loss_dicts[0].keys():
            avg_loss_dict[key] = np.mean([d[key] for d in epoch_loss_dicts])

        # Test phase
        model.eval()
        test_losses = []
        with torch.no_grad():
            for x, t, u in test_loader:
                x, t, u = x.to(config.DEVICE), t.to(config.DEVICE), u.to(config.DEVICE)

                # Different test loss for Model1 vs Model2
                if model_name == "model1":
                    Z = model(x, t, u)  # Shape: (batch_size, 2)
                    z1 = Z[:, 0:1]
                    z2 = Z[:, 1:2]
                    test_loss = torch.mean(torch.abs(z1 - z2))
                elif model_name == "model2" and model1_frozen is not None:
                    # For Model2, test loss is Z mean from Model1
                    u_tilda = model(x, t)
                    Z = model1_frozen(x, t, u_tilda)
                    test_loss = torch.mean(Z)
                else:
                    # Fallback: MSE with true solution
                    u_pred = model(x, t)
                    test_loss = nn.MSELoss()(u_pred, u)

                test_losses.append(test_loss.item())

        avg_test_loss = np.mean(test_losses)

        # Record history
        history['train_loss'].append(avg_train_loss)
        history['test_loss'].append(avg_test_loss)

        # Record model-specific metrics
        if model_name == "model1":
            history['z1_mean'].append(avg_loss_dict.get('z1_mean', 0))
            history['z1_std'].append(avg_loss_dict.get('z1_std', 0))
            history['z2_mean'].append(avg_loss_dict.get('z2_mean', 0))
            history['z2_std'].append(avg_loss_dict.get('z2_std', 0))
            history['z1_z2_diff_mean'].append(avg_loss_dict.get('z1_z2_diff_mean', 0))
            history['z1_z2_diff_abs_mean'].append(avg_loss_dict.get('z1_z2_diff_abs_mean', 0))
            history['loss_z'].append(avg_loss_dict.get('loss_z', 0))
            history['loss_norm'].append(avg_loss_dict.get('loss_norm', 0))
            history['output_l1_norm'].append(avg_loss_dict.get('output_l1_norm', 0))
            history['norm_deviation'].append(avg_loss_dict.get('norm_deviation', 0))

            # Track weights if requested
            if track_weights and hasattr(model, 'get_last_layer_weights'):
                weights, bias = model.get_last_layer_weights()
                history['weights'].append(weights.copy())

        elif model_name == "model2":
            history['Z_mean'].append(avg_loss_dict.get('Z_mean', 0))
            history['Z_std'].append(avg_loss_dict.get('Z_std', 0))
            history['mse_vs_true'].append(avg_loss_dict.get('mse_vs_true', 0))
            history['u_tilda_mean'].append(avg_loss_dict.get('u_tilda_mean', 0))
            history['u_tilda_std'].append(avg_loss_dict.get('u_tilda_std', 0))

        # Print progress
        if (epoch + 1) % config.PLOT_EVERY == 0 or epoch == 0:
            if model_name == "model1":
                print(
                    f"Epoch [{epoch+1}/{epochs}] "
                    f"Train Loss: {avg_train_loss:.6f} "
                    f"z1_mean: {avg_loss_dict.get('z1_mean', 0):.6f} "
                    f"z2_mean: {avg_loss_dict.get('z2_mean', 0):.6f} "
                    f"|z1-z2|: {avg_loss_dict.get('z1_z2_diff_abs_mean', 0):.6f} "
                    f"output_L1: {avg_loss_dict.get('output_l1_norm', 0):.6f} "
                    f"loss_norm: {avg_loss_dict.get('loss_norm', 0):.6f} "
                    f"Test Loss: {avg_test_loss:.6f}"
                )
            else:
                print(
                    f"Epoch [{epoch+1}/{epochs}] "
                    f"Train Loss (Z_mean): {avg_train_loss:.6f} "
                    f"MSE vs True: {avg_loss_dict.get('mse_vs_true', 0):.6f} "
                    f"Test Loss: {avg_test_loss:.6f}"
                )

        # Save checkpoint
        if (epoch + 1) % config.SAVE_EVERY == 0:
            save_checkpoint(model, optimizer, epoch, history, model_name)

    print("=" * 60)
    print(f"{model_name} training completed!\n")

    return history


def save_checkpoint(model, optimizer, epoch, history, model_name):
    """Save model checkpoint"""
    checkpoint_dir = Path(config.CHECKPOINT_DIR)
    checkpoint_dir.mkdir(exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history,
    }

    path = checkpoint_dir / f"{model_name}_epoch_{epoch+1}.pt"
    torch.save(checkpoint, path)
    print(f"Checkpoint saved: {path}")


def main():
    """Main training function"""
    # Set random seeds
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)

    # Create directories
    Path(config.CHECKPOINT_DIR).mkdir(exist_ok=True)
    Path(config.RESULTS_DIR).mkdir(exist_ok=True)

    # Get dataloaders
    # Set load_from_disk=True to load saved datasets (faster)
    # Set save_to_disk=True to save generated datasets for future use
    print("Loading datasets...")
    train_loader, test_loader, grid_dataset = get_dataloaders(
        load_from_disk=True,
        save_to_disk=True
    )
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")

    # ==================== Train Model 1 ====================
    print("\n" + "=" * 60)
    print("TRAINING MODEL 1")
    print("=" * 60)

    model1 = Model1(config.MODEL1_CONFIG).to(config.DEVICE)
    optimizer1 = optim.Adam(model1.parameters(), lr=config.MODEL1_CONFIG['learning_rate'])

    history1 = train_model(
        model=model1,
        train_loader=train_loader,
        test_loader=test_loader,
        loss_function=loss_function_model1,
        optimizer=optimizer1,
        epochs=config.MODEL1_CONFIG['epochs'],
        model_name="model1",
        track_weights=True,  # Enable weight tracking for Model1
    )

    # Save final model
    save_checkpoint(
        model1, optimizer1, config.MODEL1_CONFIG['epochs'] - 1, history1, "model1_final"
    )

    # Plot results for Model1
    from utils import plot_model1_training, plot_model1_weights, plot_z1_z2_evolution
    from datetime import datetime
    # Save each on a date/time folder
    date_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(config.RESULTS_DIR) / date_time
    results_dir.mkdir(exist_ok=True)
    plot_model1_training(history1, save_path=f"{results_dir}/model1_training.png")
    plot_z1_z2_evolution(history1, save_path=f"{results_dir}/model1_z1_z2_evolution.png")
    if 'weights' in history1:
        plot_model1_weights(history1, save_path=f"{results_dir}/model1_weights.png")

    # ==================== Train Model 2 ====================
    print("\n" + "=" * 60)
    print("TRAINING MODEL 2")
    print("=" * 60)

    # Load pretrained Model1 as frozen validator
    print("Loading pretrained Model1 as validator...")
    model1_checkpoint_path = Path(config.MODEL2_CONFIG['model1_checkpoint'])

    if not model1_checkpoint_path.exists():
        print(f"ERROR: Pretrained Model1 checkpoint not found at {model1_checkpoint_path}")
        print("Please train Model1 first or update the checkpoint path in config.py")
        print("Skipping Model2 training...")
    else:
        # Load Model1 checkpoint
        checkpoint = torch.load(model1_checkpoint_path, map_location=config.DEVICE)
        model1_frozen = Model1(config.MODEL1_CONFIG).to(config.DEVICE)
        model1_frozen.load_state_dict(checkpoint['model_state_dict'])
        model1_frozen.eval()

        # Freeze Model1 parameters
        for param in model1_frozen.parameters():
            param.requires_grad = False

        print(f"Loaded pretrained Model1 from {model1_checkpoint_path}")

        # Create Model2
        model2 = Model2(config.MODEL2_CONFIG).to(config.DEVICE)
        optimizer2 = optim.Adam(model2.parameters(), lr=config.MODEL2_CONFIG['learning_rate'])

        # Train Model2
        history2 = train_model(
            model=model2,
            train_loader=train_loader,
            test_loader=test_loader,
            loss_function=loss_function_model2,
            optimizer=optimizer2,
            epochs=config.MODEL2_CONFIG['epochs'],
            model_name="model2",
            track_weights=False,
            model1_frozen=model1_frozen,  # Pass frozen Model1
        )

        # Save final model
        save_checkpoint(
            model2, optimizer2, config.MODEL2_CONFIG['epochs'] - 1, history2, "model2_final"
        )

        # Plot results for Model2
        from utils import plot_model2_training
        plot_model2_training(history2, save_path=f"{results_dir}/model2_training.png")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETED!")
    print("=" * 60)


if __name__ == "__main__":
    main()
