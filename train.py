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
from utils import plot_model1_training, plot_model1_weights

import config
from models import Model1, Model2, wave_equation_residual
from dataset import get_dataloaders, get_patch_dataloaders
from utils import plot_training_history, plot_predictions, plot_error_analysis


def loss_function_model1(model, x, t, u_true):
    """
    Loss function for Model 1

    Model 1 takes triplet inputs (x, t, u) and outputs scalar Z
    Loss: L = λ_z * mean(|Z|) + λ_norm * (L1_norm - target_norm)²

    The target norm term encourages weights to stay near a target magnitude,
    preventing both collapse to zero and unbounded growth.

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
    Z = model(x, t, u_true)

    # Term 1: mean absolute value of Z
    loss_z = torch.mean(torch.abs(Z))

    # Term 2: Target norm regularization for last layer weights
    # Get the last linear layer (output layer)
    last_layer = model.network[-1]  # This is the Linear layer that outputs Z
    weights = last_layer.weight  # Shape: (1, 100)

    # L1 norm: sum of absolute values
    l1_norm = torch.sum(torch.abs(weights))

    # Get hyperparameters from config
    lambda_z = config.MODEL1_CONFIG.get('lambda_z', 1.0)
    lambda_norm = config.MODEL1_CONFIG.get('lambda_norm', 0.01)
    target_norm = config.MODEL1_CONFIG.get('target_norm', 50.0)

    # Target norm penalty: (current_norm - target_norm)²
    # This creates a "sweet spot" at target_norm
    norm_deviation = l1_norm - target_norm
    loss_norm = norm_deviation ** 2

    # Total loss
    loss = lambda_z * loss_z + lambda_norm * loss_norm

    loss_dict = {
        'total': loss.item(),
        'loss_z': loss_z.item(),
        'loss_norm': loss_norm.item(),
        'l1_norm': l1_norm.item(),
        'norm_deviation': norm_deviation.item(),  # Signed difference from target
        'Z_mean': torch.mean(Z).item(),
        'Z_std': torch.std(Z).item(),
        'Z_abs_mean': loss_z.item(),
    }

    return loss, loss_dict


def loss_function_model1_patches(model, patches):
    """
    Loss function for Model 1 when using N×N patches.

    patches: (batch_size, 3, N, N) -> flatten to (batch_size, 3*N*N)
    """
    B = patches.shape[0]
    vec = patches.view(B, -1)

    # Forward pass with flattened patch vector
    Z = model.forward_vector(vec)

    # Term 1: mean absolute value of Z
    loss_z = torch.mean(torch.abs(Z))

    # Term 2: Target norm regularization for last layer weights
    last_layer = model.network[-1]
    weights = last_layer.weight

    l1_norm = torch.sum(torch.abs(weights))
    lambda_z = config.MODEL1_CONFIG.get('lambda_z', 1.0)
    lambda_norm = config.MODEL1_CONFIG.get('lambda_norm', 0.01)
    target_norm = config.MODEL1_CONFIG.get('target_norm', 50.0)

    norm_deviation = l1_norm - target_norm
    loss_norm = norm_deviation ** 2

    loss = lambda_z * loss_z + lambda_norm * loss_norm

    loss_dict = {
        'total': loss.item(),
        'loss_z': loss_z.item(),
        'loss_norm': loss_norm.item(),
        'l1_norm': l1_norm.item(),
        'norm_deviation': norm_deviation.item(),
        'Z_mean': torch.mean(Z).item(),
        'Z_std': torch.std(Z).item(),
        'Z_abs_mean': loss_z.item(),
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

    with torch.no_grad():
        loader = WavePatchDataset(train_grid, patch_size=N)
        model1_frozen.eval()
        for batch in test_loader:
            if model_name == "model1" and getattr(config, 'USE_PATCHES_FOR_MODEL1', False):
                patches = batch.to(config.DEVICE)
                B = patches.shape[0]
                vec = patches.view(B, -1)
                Z = model1_frozen.forward_vector(vec)

    # Pass triplet (x, t, u_tilda) to frozen Model1
    with torch.no_grad():
        model1_frozen.eval()

    Z = model1_frozen(x, t, u_tilda)

    # Loss: mean of Z (we want Z -> 0)
    # Note: We use mean(Z) not mean(|Z|) because Model1 was trained with mean(|Z|)
    # and should output values close to 0 for correct triplets
    loss = torch.mean(torch.abs(Z))

    # Debug metrics (monitoring only; no gradients)
    with torch.no_grad():
        mse = nn.MSELoss()(u_tilda, u_true)
        Z_abs_mean = torch.mean(torch.abs(Z))
        # Compare validator outputs on true u and on zero baseline
        Z_true = model1_frozen(x, t, u_true)
        Z_zero = model1_frozen(x, t, torch.zeros_like(u_true))
        Z_true_mean = torch.mean(Z_true)
        Z_zero_mean = torch.mean(Z_zero)
        # Scale-invariant relation diagnostic: correlation between u_tilda and u_true
        ut = u_tilda.view(-1)
        u = u_true.view(-1)
        ut_c = ut - ut.mean()
        u_c = u - u.mean()
        corr = torch.dot(ut_c, u_c) / (ut_c.norm() * u_c.norm() + 1e-8)

    loss_dict = {
        'total': loss.item(),
        'Z_mean': torch.mean(Z).item(),
        'Z_std': torch.std(Z).item(),
        'Z_abs_mean': Z_abs_mean.item(),
        'Z_true_mean': Z_true_mean.item(),
        'Z_zero_mean': Z_zero_mean.item(),
        'corr': corr.item(),
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
        history['Z_mean'] = []
        history['Z_std'] = []
        history['Z_abs_mean'] = []
        history['loss_z'] = []  # Mean |Z| component
        history['loss_norm'] = []  # Target norm penalty component
        history['l1_norm'] = []  # L1 norm value (positive)
        history['norm_deviation'] = []  # Signed difference from target
        if track_weights:
            history['weights'] = []  # Track weight vector from last layer

    # For Model2, track Z metrics and MSE vs true
    if model_name == "model2":
        history['Z_mean'] = []
        history['Z_std'] = []
        history['Z_abs_mean'] = []
        history['Z_true_mean'] = []
        history['Z_zero_mean'] = []
        history['corr'] = []
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

        for batch_idx, batch in enumerate(train_loader):
            # Zero gradients
            optimizer.zero_grad()

            if model_name == "model1" and getattr(config, 'USE_PATCHES_FOR_MODEL1', False):
                # Batch of patches: (B, 3, N, N)
                patches = batch.to(config.DEVICE)
                loss, loss_dict = loss_function_model1_patches(model, patches)
            else:
                # Standard triplet batches (x, t, u)
                x, t, u = batch
                x, t, u = x.to(config.DEVICE), t.to(config.DEVICE), u.to(config.DEVICE)
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
            for batch in test_loader:
                if model_name == "model1" and getattr(config, 'USE_PATCHES_FOR_MODEL1', False):
                    patches = batch.to(config.DEVICE)
                    B = patches.shape[0]
                    vec = patches.view(B, -1)
                    Z = model.forward_vector(vec)
                    test_loss = torch.mean(torch.abs(Z))
                else:
                    x, t, u = batch
                    x, t, u = x.to(config.DEVICE), t.to(config.DEVICE), u.to(config.DEVICE)

                    if model_name == "model1":
                        Z = model(x, t, u)
                        test_loss = torch.mean(torch.abs(Z))
                    elif model_name == "model2" and model1_frozen is not None:
                        u_tilda = model(x, t)
                        Z = model1_frozen(x, t, u_tilda)
                        test_loss = torch.mean(torch.abs(Z))
                    else:
                        u_pred = model(x, t)
                        test_loss = nn.MSELoss()(u_pred, u)

                test_losses.append(test_loss.item())

        avg_test_loss = np.mean(test_losses)

        # Record history
        history['train_loss'].append(avg_train_loss)
        history['test_loss'].append(avg_test_loss)

        # Record model-specific metrics
        if model_name == "model1":
            history['Z_mean'].append(avg_loss_dict.get('Z_mean', 0))
            history['Z_std'].append(avg_loss_dict.get('Z_std', 0))
            history['Z_abs_mean'].append(avg_loss_dict.get('Z_abs_mean', 0))
            history['loss_z'].append(avg_loss_dict.get('loss_z', 0))
            history['loss_norm'].append(avg_loss_dict.get('loss_norm', 0))
            history['l1_norm'].append(avg_loss_dict.get('l1_norm', 0))
            history['norm_deviation'].append(avg_loss_dict.get('norm_deviation', 0))

            # Track weights if requested
            if track_weights and hasattr(model, 'get_last_layer_weights'):
                weights, bias = model.get_last_layer_weights()
                history['weights'].append(weights.copy())

        elif model_name == "model2":
            history['Z_mean'].append(avg_loss_dict.get('Z_mean', 0))
            history['Z_std'].append(avg_loss_dict.get('Z_std', 0))
            history['Z_abs_mean'].append(avg_loss_dict.get('Z_abs_mean', 0))
            history['Z_true_mean'].append(avg_loss_dict.get('Z_true_mean', 0))
            history['Z_zero_mean'].append(avg_loss_dict.get('Z_zero_mean', 0))
            history['corr'].append(avg_loss_dict.get('corr', 0))
            history['mse_vs_true'].append(avg_loss_dict.get('mse_vs_true', 0))
            history['u_tilda_mean'].append(avg_loss_dict.get('u_tilda_mean', 0))
            history['u_tilda_std'].append(avg_loss_dict.get('u_tilda_std', 0))

        # Print progress
        if (epoch + 1) % config.PLOT_EVERY == 0 or epoch == 0:
            # Batch-scale snapshot
            try:
                if model_name == "model1" and getattr(config, 'USE_PATCHES_FOR_MODEL1', False):
                    # Recompute a quick range from last seen training loss inputs if available
                    # Fetch one small batch from train_loader for logging
                    patches_sample = next(iter(train_loader))
                    ps = patches_sample.to(config.DEVICE)
                    x_ch = ps[:, 0, :, :]
                    t_ch = ps[:, 1, :, :]
                    u_ch = ps[:, 2, :, :]
                    print(
                        f"[epoch {epoch+1}] x[{x_ch.min().item():.4f},{x_ch.max().item():.4f}] "
                        f"t[{t_ch.min().item():.4f},{t_ch.max().item():.4f}] "
                        f"u[{u_ch.min().item():.4f},{u_ch.max().item():.4f}]"
                    )
                else:
                    print(
                        f"[epoch {epoch+1}] x[{x.min().item():.4f},{x.max().item():.4f}] "
                        f"t[{t.min().item():.4f},{t.max().item():.4f}] "
                        f"u[{u.min().item():.4f},{u.max().item():.4f}]"
                    )
            except Exception:
                pass
            if model_name == "model1":
                print(
                    f"Epoch [{epoch+1}/{epochs}] "
                    f"Train Loss: {avg_train_loss:.6f} "
                    f"Z_mean: {avg_loss_dict.get('Z_mean', 0):.6f} "
                    f"Test Loss: {avg_test_loss:.6f} "
                    f"Abs weight sum: {abs(weights).sum():.6f}"
                )
            else:
                print(
                    f"Epoch [{epoch+1}/{epochs}] "
                    f"Train Loss (Z_mean): {avg_train_loss:.6f} "
                    f"|Z|_mean: {avg_loss_dict.get('Z_abs_mean', 0):.6f} "
                    f"Z(true)_mean: {avg_loss_dict.get('Z_true_mean', 0):.6f} "
                    f"Z(0)_mean: {avg_loss_dict.get('Z_zero_mean', 0):.6f} "
                    f"corr(u_tilda,u): {avg_loss_dict.get('corr', 0):.4f} "
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
    
    run_both_models = config.RUN_BOTH_MODELS # 0 runs both, 1 runs only Model 1, 2 runs only Model 2

    # Create directories
    Path(config.CHECKPOINT_DIR).mkdir(exist_ok=True)
    Path(config.RESULTS_DIR).mkdir(exist_ok=True)

    # Get dataloaders
    # Model1 can use patch loaders; Model2 continues to use standard loaders
    print("Loading datasets...")
    use_patches = getattr(config, 'USE_PATCHES_FOR_MODEL1', False)
    if use_patches:
        train_loader_m1, test_loader_m1, grid_dataset = get_patch_dataloaders()
    else:
        train_loader_m1, test_loader_m1, grid_dataset = get_dataloaders(
            load_from_disk=True,
            save_to_disk=True
        )

    # Separate standard loaders for Model2 (unchanged)
    train_loader_std, test_loader_std, grid_dataset_std = get_dataloaders(
        load_from_disk=True,
        save_to_disk=True
    )
    # Scale diagnostics for datasets
    try:
        x_train = train_loader_std.dataset.x.detach().cpu().numpy().flatten()
        t_train = train_loader_std.dataset.t.detach().cpu().numpy().flatten()
        u_train = train_loader_std.dataset.u.detach().cpu().numpy().flatten()
        print(
            f"x_train range: [{x_train.min():.6f}, {x_train.max():.6f}] "
            f"mean={x_train.mean():.6f} std={x_train.std():.6f}"
        )
        print(
            f"t_train range: [{t_train.min():.6f}, {t_train.max():.6f}] "
            f"mean={t_train.mean():.6f} std={t_train.std():.6f}"
        )
        print(
            f"u_train range: [{u_train.min():.6f}, {u_train.max():.6f}] "
            f"mean={u_train.mean():.6f} std={u_train.std():.6f}"
        )
    except Exception as e:
        print(f"Could not compute dataset scale diagnostics: {e}")
    model1_final_path = Path(config.CHECKPOINT_DIR) / f"model1_final_epoch_{config.MODEL1_CONFIG['epochs']}.pt"

    print(f"Train samples (Model1 path): {len(train_loader_m1.dataset)}")
    print(f"Test samples (Model1 path): {len(test_loader_m1.dataset)}")
    
    date_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(config.RESULTS_DIR) / date_time
    results_dir.mkdir(exist_ok=True)
    if run_both_models == 0 or run_both_models == 1:
        # ==================== Train Model 1 ====================
        print("\n" + "=" * 60)
        print("TRAINING MODEL 1")
        print("=" * 60)
        model1 = Model1(config.MODEL1_CONFIG).to(config.DEVICE)
        optimizer1 = optim.Adam(model1.parameters(), lr=config.MODEL1_CONFIG['learning_rate'])

        history1 = train_model(
            model=model1,
            train_loader=train_loader_m1,
            test_loader=test_loader_m1,
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
        # Path to the freshly saved final Model1 checkpoint
        model1_final_path = Path(config.CHECKPOINT_DIR) / f"model1_final_epoch_{config.MODEL1_CONFIG['epochs']}.pt"

        # Plot results for Model1
        # Save each on a date/time folder

        plot_model1_training(history1, save_path=f"{results_dir}/model1_training.png")
        if 'weights' in history1:
            plot_model1_weights(history1, save_path=f"{results_dir}/model1_weights.png")

    # ==================== Train Model 2 ====================
    
    if run_both_models == 0 or run_both_models == 2:
        print("\n" + "=" * 60)
        print("TRAINING MODEL 2")
        print("=" * 60)

        # Load pretrained Model1 as frozen validator
        print("Loading pretrained Model1 as validator...")
        # Prefer the freshly saved Model1 from this run; fallback to configured path
        model1_checkpoint_path = (
            model1_final_path if model1_final_path.exists() else Path(config.MODEL2_CONFIG['model1_checkpoint'])
        )

        if not model1_checkpoint_path.exists():
            print(f"ERROR: Pretrained Model1 checkpoint not found at {model1_checkpoint_path}")
            print("Please train Model1 first or update the checkpoint path in config.py")
            print("Skipping Model2 training...")
        else:
            # Load Model1 checkpoint
            checkpoint = torch.load(
                model1_checkpoint_path,
                map_location=config.DEVICE,
                weights_only=False,  # allow full unpickling of trusted local checkpoint
            )
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
                train_loader=train_loader_std,
                test_loader=test_loader_std,
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
            from utils import plot_model2_training, plot_model2_u_comparison
            plot_model2_training(history2, save_path=f"{results_dir}/model2_training.png")
            # Compare u_true vs u_tilda with heatmaps and difference
            plot_model2_u_comparison(model2, grid_dataset, save_path=f"{results_dir}/model2_u_vs_utilda.png")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETED!")
    print("=" * 60)


if __name__ == "__main__":
    main()
