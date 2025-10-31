"""
Grid Search Script for Model 1

Performs hyperparameter grid search using YAML configuration
"""
import torch
import torch.optim as optim
import numpy as np
import yaml
from pathlib import Path
from itertools import product
import json
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd

import config
from models import Model1
from dataset import get_dataloaders
from train import loss_function_model1, train_model, save_checkpoint
from utils import plot_model1_training, plot_model1_weights


def load_config(config_path='grid_search_config.yaml'):
    """Load grid search configuration from YAML file"""
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg


def create_hyperparameter_combinations(grid_config):
    """
    Create all combinations of hyperparameters from grid search config

    Args:
        grid_config: dictionary with lists of values for each hyperparameter

    Returns:
        list of dictionaries, each containing one hyperparameter combination
    """
    # Get parameter names and their possible values
    param_names = list(grid_config.keys())
    param_values = [grid_config[name] for name in param_names]

    # Generate all combinations
    combinations = []
    for values in product(*param_values):
        combo = dict(zip(param_names, values))
        combinations.append(combo)

    return combinations


def run_experiment(exp_id, hyperparams, cfg, train_loader, test_loader, results_dir):
    """
    Run a single training experiment with given hyperparameters

    Args:
        exp_id: experiment ID
        hyperparams: dictionary of hyperparameters for this experiment
        cfg: full config dictionary
        train_loader: training dataloader
        test_loader: test dataloader
        results_dir: directory to save results

    Returns:
        results: dictionary with training history and final metrics
    """
    print("\n" + "=" * 70)
    print(f"EXPERIMENT {exp_id}")
    print("=" * 70)
    print("Hyperparameters:")
    for key, value in hyperparams.items():
        print(f"  {key}: {value}")
    print("=" * 70)

    # Set random seed
    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])

    # Create model
    model = Model1().to(config.DEVICE)

    # Create optimizer with specified learning rate
    optimizer = optim.Adam(model.parameters(), lr=hyperparams['learning_rate'])

    # Update config with hyperparameters for this experiment
    temp_config = config.MODEL1_CONFIG.copy()
    temp_config['lambda_z'] = hyperparams['lambda_z']
    temp_config['lambda_norm'] = hyperparams['lambda_norm']
    temp_config['target_norm'] = hyperparams['target_norm']
    temp_config['learning_rate'] = hyperparams['learning_rate']

    # Temporarily update the global config
    original_config = config.MODEL1_CONFIG.copy()
    config.MODEL1_CONFIG.update(temp_config)

    # Train model
    history = train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        loss_function=loss_function_model1,
        optimizer=optimizer,
        epochs=cfg['training']['epochs'],
        model_name=f"exp_{exp_id}",
        track_weights=cfg['output']['track_weights'],
    )

    # Restore original config
    config.MODEL1_CONFIG.update(original_config)

    # Create experiment directory
    exp_dir = results_dir / f"exp_{exp_id}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Save hyperparameters
    with open(exp_dir / "hyperparameters.json", 'w') as f:
        json.dump(hyperparams, f, indent=2)

    # Save training history
    with open(exp_dir / "history.json", 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        history_serializable = {}
        for key, value in history.items():
            if key == 'weights':
                continue  # Skip weights array
            history_serializable[key] = value if isinstance(value, list) else [float(v) for v in value]
        json.dump(history_serializable, f, indent=2)

    # Save plots if requested
    if cfg['output']['save_plots']:
        plot_model1_training(history, save_path=str(exp_dir / "training_curves.png"))
        if 'weights' in history and cfg['output']['track_weights']:
            plot_model1_weights(history, save_path=str(exp_dir / "weight_evolution.png"))

    # Save checkpoint if requested
    if cfg['output']['save_checkpoints']:
        checkpoint_path = exp_dir / "model_final.pt"
        checkpoint = {
            'epoch': cfg['training']['epochs'] - 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'history': history,
            'hyperparameters': hyperparams,
        }
        torch.save(checkpoint, checkpoint_path)

    # Compute final metrics
    final_metrics = {
        'final_train_loss': history['train_loss'][-1],
        'final_test_loss': history['test_loss'][-1],
        'final_Z_mean': history['Z_mean'][-1],
        'final_Z_std': history['Z_std'][-1],
        'final_l1_norm': history['l1_norm'][-1] if 'l1_norm' in history else None,
        'min_train_loss': min(history['train_loss']),
        'min_test_loss': min(history['test_loss']),
    }

    results = {
        'exp_id': exp_id,
        'hyperparameters': hyperparams,
        'metrics': final_metrics,
        'exp_dir': str(exp_dir),
    }

    print(f"\nFinal Metrics:")
    print(f"  Train Loss: {final_metrics['final_train_loss']:.6f}")
    print(f"  Test Loss: {final_metrics['final_test_loss']:.6f}")
    print(f"  Z Mean: {final_metrics['final_Z_mean']:.6f}")
    print(f"  L1 Norm: {final_metrics['final_l1_norm']:.4f}" if final_metrics['final_l1_norm'] else "")

    return results


def plot_grid_search_results(all_results, results_dir):
    """
    Create summary plots comparing all experiments

    Args:
        all_results: list of result dictionaries from all experiments
        results_dir: directory to save plots
    """
    print("\nCreating summary plots...")

    # Extract data for plotting
    exp_ids = [r['exp_id'] for r in all_results]
    train_losses = [r['metrics']['final_train_loss'] for r in all_results]
    test_losses = [r['metrics']['final_test_loss'] for r in all_results]
    z_means = [r['metrics']['final_Z_mean'] for r in all_results]
    l1_norms = [r['metrics']['final_l1_norm'] for r in all_results]

    # Extract hyperparameters
    lrs = [r['hyperparameters']['learning_rate'] for r in all_results]
    lambda_zs = [r['hyperparameters']['lambda_z'] for r in all_results]
    lambda_norms = [r['hyperparameters']['lambda_norm'] for r in all_results]
    target_norms = [r['hyperparameters']['target_norm'] for r in all_results]

    # Create summary figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Grid Search Results Summary', fontsize=16, fontweight='bold')

    # Plot 1: Final train loss by experiment
    ax1 = axes[0, 0]
    ax1.bar(exp_ids, train_losses, color='skyblue', edgecolor='black')
    ax1.set_xlabel('Experiment ID', fontsize=12)
    ax1.set_ylabel('Final Train Loss', fontsize=12)
    ax1.set_title('Final Training Loss', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')

    # Plot 2: Final test loss by experiment
    ax2 = axes[0, 1]
    ax2.bar(exp_ids, test_losses, color='lightcoral', edgecolor='black')
    ax2.set_xlabel('Experiment ID', fontsize=12)
    ax2.set_ylabel('Final Test Loss', fontsize=12)
    ax2.set_title('Final Test Loss', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    # Plot 3: Final Z mean by experiment
    ax3 = axes[0, 2]
    ax3.bar(exp_ids, z_means, color='lightgreen', edgecolor='black')
    ax3.axhline(y=0, color='red', linestyle='--', linewidth=2)
    ax3.set_xlabel('Experiment ID', fontsize=12)
    ax3.set_ylabel('Final Z Mean', fontsize=12)
    ax3.set_title('Final Z Mean (target: 0)', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')

    # Plot 4: Learning rate effect
    ax4 = axes[1, 0]
    scatter = ax4.scatter(lrs, train_losses, c=lambda_norms, cmap='viridis', s=100, edgecolor='black')
    ax4.set_xlabel('Learning Rate', fontsize=12)
    ax4.set_ylabel('Final Train Loss', fontsize=12)
    ax4.set_title('Learning Rate vs Loss (colored by λ_norm)', fontsize=14, fontweight='bold')
    ax4.set_xscale('log')
    ax4.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax4, label='λ_norm')

    # Plot 5: Lambda_norm vs Target_norm
    ax5 = axes[1, 1]
    scatter = ax5.scatter(lambda_norms, target_norms, c=train_losses, cmap='coolwarm', s=100, edgecolor='black')
    ax5.set_xlabel('λ_norm', fontsize=12)
    ax5.set_ylabel('target_norm', fontsize=12)
    ax5.set_title('Hyperparameter Space (colored by loss)', fontsize=14, fontweight='bold')
    ax5.set_xscale('log')
    ax5.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax5, label='Train Loss')

    # Plot 6: L1 norm by experiment
    ax6 = axes[1, 2]
    ax6.bar(exp_ids, l1_norms, color='orange', edgecolor='black')
    ax6.set_xlabel('Experiment ID', fontsize=12)
    ax6.set_ylabel('Final L1 Norm', fontsize=12)
    ax6.set_title('Final Weight L1 Norm', fontsize=14, fontweight='bold')
    ax6.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(results_dir / 'grid_search_summary.png', dpi=150, bbox_inches='tight')
    print(f"Summary plot saved to: {results_dir / 'grid_search_summary.png'}")
    plt.show()


def create_results_table(all_results, results_dir):
    """Create a CSV table with all results"""
    print("\nCreating results table...")

    rows = []
    for r in all_results:
        row = {
            'exp_id': r['exp_id'],
            'learning_rate': r['hyperparameters']['learning_rate'],
            'lambda_z': r['hyperparameters']['lambda_z'],
            'lambda_norm': r['hyperparameters']['lambda_norm'],
            'target_norm': r['hyperparameters']['target_norm'],
            'final_train_loss': r['metrics']['final_train_loss'],
            'final_test_loss': r['metrics']['final_test_loss'],
            'final_Z_mean': r['metrics']['final_Z_mean'],
            'final_Z_std': r['metrics']['final_Z_std'],
            'final_l1_norm': r['metrics']['final_l1_norm'],
            'min_train_loss': r['metrics']['min_train_loss'],
            'min_test_loss': r['metrics']['min_test_loss'],
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    # Sort by final train loss
    df = df.sort_values('final_train_loss')

    # Save to CSV
    csv_path = results_dir / 'grid_search_results.csv'
    df.to_csv(csv_path, index=False)
    print(f"Results table saved to: {csv_path}")

    # Print top 5 experiments
    print("\n" + "=" * 70)
    print("TOP 5 EXPERIMENTS (by final train loss)")
    print("=" * 70)
    print(df.head(5).to_string(index=False))
    print("=" * 70)

    return df


def main():
    """Main grid search function"""
    # Load configuration
    print("=" * 70)
    print("GRID SEARCH FOR MODEL 1")
    print("=" * 70)

    cfg = load_config('grid_search_config.yaml')
    print("\nLoaded configuration from: grid_search_config.yaml")

    # Create results directory
    results_dir = Path(cfg['output']['results_dir'])
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results will be saved to: {results_dir}")

    # Save config to results directory
    with open(results_dir / 'config.yaml', 'w') as f:
        yaml.dump(cfg, f)

    # Generate all hyperparameter combinations
    combinations = create_hyperparameter_combinations(cfg['grid_search'])
    print(f"\nTotal number of experiments: {len(combinations)}")
    print("\nHyperparameter combinations:")
    for i, combo in enumerate(combinations):
        print(f"  Exp {i+1}: {combo}")

    # Load datasets once (shared across all experiments)
    print("\n" + "=" * 70)
    print("LOADING DATASETS")
    print("=" * 70)
    train_loader, test_loader, grid_dataset = get_dataloaders(
        load_from_disk=cfg['data']['load_from_disk'],
        save_to_disk=cfg['data']['save_to_disk']
    )
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")

    # Run all experiments
    all_results = []
    start_time = datetime.now()

    for exp_id, hyperparams in enumerate(combinations, start=1):
        try:
            results = run_experiment(
                exp_id=exp_id,
                hyperparams=hyperparams,
                cfg=cfg,
                train_loader=train_loader,
                test_loader=test_loader,
                results_dir=results_dir,
            )
            all_results.append(results)

        except Exception as e:
            print(f"\nERROR in experiment {exp_id}: {e}")
            print("Continuing with next experiment...")

    end_time = datetime.now()
    duration = end_time - start_time

    # Save all results
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)

    results_file = results_dir / 'all_results.json'
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"All results saved to: {results_file}")

    # Create summary plots
    plot_grid_search_results(all_results, results_dir)

    # Create results table
    df = create_results_table(all_results, results_dir)

    # Print summary
    print("\n" + "=" * 70)
    print("GRID SEARCH COMPLETED")
    print("=" * 70)
    print(f"Total experiments: {len(all_results)}")
    print(f"Total duration: {duration}")
    print(f"Average time per experiment: {duration / len(all_results)}")
    print(f"\nBest experiment: {df.iloc[0]['exp_id']:.0f}")
    print(f"  Learning rate: {df.iloc[0]['learning_rate']}")
    print(f"  λ_z: {df.iloc[0]['lambda_z']}")
    print(f"  λ_norm: {df.iloc[0]['lambda_norm']}")
    print(f"  target_norm: {df.iloc[0]['target_norm']}")
    print(f"  Final train loss: {df.iloc[0]['final_train_loss']:.6f}")
    print(f"  Final test loss: {df.iloc[0]['final_test_loss']:.6f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
