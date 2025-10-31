"""
Configuration file for 1D Wave Equation PINN
"""
import torch

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Wave equation parameters
WAVE_SPEED = 1.0  # c = 1
X_MIN = -1.0
X_MAX = 1.0
T_MIN = 0.0
T_MAX = 1.0  # 1 period

# Use grid sampling (linspace) for train/test datasets (includes boundaries)
USE_GRID_SAMPLING = True

# Grid sizes when USE_GRID_SAMPLING is True
GRID_NX_TRAIN = 200
GRID_NT_TRAIN = 100
GRID_NX_TEST = 200
GRID_NT_TEST = 100

# Dataset parameters
N_TRAIN_POINTS = 10000  # Number of collocation points for training
N_TEST_POINTS = 2000    # Number of points for testing
WAVELENGTH = 2.0        # Wavelength of the wave (can be adjusted)

# Model 1 hyperparameters
# Architecture: Input (x,t,u) -> 5 hidden layers (100 each) -> Output Z
MODEL1_CONFIG = {
    'input_dim': 3,         # Overridden to 3*N*N when using patches
    'hidden_dim': 50,      # 100 neurons per layer
    'n_layers': 5,          # 5 hidden layers
    'output_dim': 1,        # Scalar Z
    'activation': 'tanh',
    'learning_rate': 1e-4,
    'epochs': 1000,
    'batch_size': 256,
    # Loss weights
    'lambda_z': 1.0,        # Weight for mean(|Z|) term
    'lambda_norm': 0.01,    # Weight for target norm penalty
    'target_norm': 100.0,     # Target L1 norm for last layer weights
                            # (100 weights -> avg 0.01 per weight)
}

# Model 2 hyperparameters
MODEL2_CONFIG = {
    'hidden_layers': [100, 100, 100, 100],
    'activation': 'sin',
    'learning_rate': 1e-6,
    'epochs': 1000,
    'batch_size': 256,
    # Pretrained Model1 for loss computation
    'model1_checkpoint': './checkpoints/model1_final_epoch_2500.pt',  # Path to pretrained Model1
}

# Training parameters
SAVE_EVERY = 500  # Save checkpoint every N epochs
PLOT_EVERY = 100  # Plot results every N epochs
CHECKPOINT_DIR = './checkpoints'
RESULTS_DIR = './results'

# Random seed for reproducibility
SEED = 42
RUN_BOTH_MODELS = 1 # "0" runs both, "1" runs only Model 1, "2" runs only Model 2

# Normalize x,t inputs to [-1, 1] in datasets

# Regional patches (x,t,u heatmap neighborhoods) for Model1
# When enabled, Model1 will receive N×N patches (flattened 3·N·N vector)
USE_PATCHES_FOR_MODEL1 = True
PATCH_SIZE = 5  # N in N×N patches (prefer odd for a clear center)
