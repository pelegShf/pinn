"""
Configuration file for 1D Wave Equation PINN
"""
import torch

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Wave equation parameters
WAVE_SPEED = 1.0  # c = 1
X_MIN = -5.0
X_MAX = 5.0
T_MIN = 0.0
T_MAX = 1.0  # 1 period

# Dataset parameters
N_TRAIN_POINTS = 10000  # Number of collocation points for training
N_TEST_POINTS = 2000    # Number of points for testing
WAVELENGTH = 2.0        # Wavelength of the wave (can be adjusted)

# Model 1 hyperparameters
# Architecture: Input (x,t,u) -> 5 hidden layers (100 each) -> Output Z
MODEL1_CONFIG = {
    'input_dim': 3,         # (x, t, u) triplet input
    'hidden_dim': 100,      # 100 neurons per layer
    'n_layers': 5,          # 5 hidden layers
    'output_dim': 1,        # Scalar Z
    'activation': 'tanh',
    'learning_rate': 1e-5,
    'epochs': 2000,
    'batch_size': 256,
    # Loss weights
    'lambda_z': 1.0,        # Weight for mean(|z1-z2|) term
    'lambda_norm': 0.01,    # Weight for target norm penalty
    'target_norm': 10.0,     # Target L1 norm for output vector (|z1| + |z2|)
                            # This controls the magnitude of the outputs
}

# Model 2 hyperparameters
MODEL2_CONFIG = {
    'hidden_layers': [100, 100, 100, 100],
    'activation': 'tanh',
    'learning_rate': 1e-3,
    'epochs': 5000,
    'batch_size': 256,
    # Pretrained Model1 for loss computation
    'model1_checkpoint': './checkpoints/model1_final.pt',  # Path to pretrained Model1
}

# Training parameters
SAVE_EVERY = 500  # Save checkpoint every N epochs
PLOT_EVERY = 100  # Plot results every N epochs
CHECKPOINT_DIR = './checkpoints'
RESULTS_DIR = './results'

# Random seed for reproducibility
SEED = 42
