# Physics-Informed Neural Networks for 1D Wave Equation

This project implements Physics-Informed Neural Networks (PINNs) to solve the 1D wave equation using PyTorch.

## Wave Equation

The 1D wave equation is:

```
∂²u/∂t² = c² ∂²u/∂x²
```

where:
- `u(x,t)` is the wave displacement
- `c` is the wave speed (set to 1.0)
- `x` is the spatial coordinate (range: -5 to 5)
- `t` is the temporal coordinate (range: 0 to 1 period)

## Project Structure

```
pinn/
├── config.py                  # Configuration parameters
├── dataset.py                 # Dataset generation (x, t, u(x,t)) triplets
├── models.py                  # Two PINN model architectures
├── train.py                   # Training loops for both models
├── utils.py                   # Plotting and visualization utilities
├── visualize_data.py          # Quick script to verify data quality
├── compare_models.py          # Compare model architectures visually
├── test_initial_condition.py  # Verify u(x,0) = 0
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── MODELS_EXPLAINED.md        # Detailed explanation of how models work
├── data/                      # Saved datasets (created by dataset.py)
├── checkpoints/               # Saved model checkpoints (created during training)
└── results/                   # Plots and visualizations (created during training)
```

## Installation

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Quick Start

Verify your data quality first (recommended):
```bash
python visualize_data.py
```
This creates:
- Detailed heatmap with quality metrics
- Wave animation (GIF)
- Complete dataset visualizations

Then train both models:
```bash
python train.py
```

### Test Individual Components

**Visualize and verify dataset quality:**
```bash
python visualize_data.py
```
Creates comprehensive visualizations:
- Heatmap with contours, gradients, and frequency analysis
- Wave animation showing time evolution
- Quality metrics and statistics

**Test dataset generation:**
```bash
python dataset.py
```
This will:
- Generate train, test, and grid datasets
- Save them to `./data/` directory
- Create all visualizations including heatmaps and animations
- Test loading from disk

**Test model architectures:**
```bash
python models.py
```

**Test plotting functions:**
```bash
python utils.py
```

## Configuration

Edit `config.py` to modify:
- Wave parameters (speed, domain, wavelength)
- Model architectures (hidden layers, activation functions)
- Training hyperparameters (learning rate, epochs, batch size)
- Dataset size

## Models

### Model 1: Balanced Approach
- **Architecture:** 3 hidden layers with 50 neurons each (~7,600 parameters)
- **Loss function:** L = 1.0×L_data + 1.0×L_physics (equal weights)
- **Philosophy:** Trust data and physics equally
- **Best for:** Fast training, simple problems, prototyping
- Customizable in `models.py` and `train.py`

### Model 2: Physics-First Approach
- **Architecture:** 4 hidden layers with 100 neurons each (~40,400 parameters)
- **Loss function:** L = 0.5×L_data + 1.5×L_physics (physics-focused)
- **Philosophy:** Trust physics more than data
- **Best for:** Complex problems, noisy data, better generalization
- Customizable in `models.py` and `train.py`

### Understanding the Models

For a detailed explanation of how these models work:
```bash
# Read the comprehensive guide
cat MODELS_EXPLAINED.md

# Or generate visual comparisons
python compare_models.py
```

This creates:
- Architecture comparison diagrams
- Loss function visualizations
- Detailed parameter counts
- Training philosophy explanations

## Dataset

The dataset generates triplets `(x, t, u(x,t))` where:
- `x`: spatial coordinates sampled uniformly from [-5, 5]
- `t`: temporal coordinates sampled uniformly from [0, 1 period]
- `u(x,t)`: analytical solution (standing wave by default)

Default analytical solution:
```python
u(x,t) = sin(k*x) * sin(ω*t)
```

where `k = 2π/λ` (wave number) and `ω = k*c` (angular frequency).

**Initial conditions:**
- `u(x, 0) = 0` (zero initial displacement everywhere)
- `∂u/∂t(x, 0) = ω*sin(k*x)` (non-zero initial velocity)

This represents a standing wave that starts from rest (zero displacement) and oscillates.

You can modify the analytical solution in `dataset.py` to match your specific problem.

### Dataset Saving and Loading

Datasets can be saved to disk to avoid regenerating them every time:

```python
# In your code
from dataset import get_dataloaders

# Generate and save datasets
train_loader, test_loader, grid_dataset = get_dataloaders(
    save_to_disk=True,
    data_dir='./data'
)

# Load previously saved datasets (much faster)
train_loader, test_loader, grid_dataset = get_dataloaders(
    load_from_disk=True,
    data_dir='./data'
)
```

The training script (`train.py`) automatically uses saved datasets if available, falling back to generation if not.

### Dataset Visualization

Visualize the dataset to understand the wave structure:

```python
from dataset import visualize_dataset, WaveDatasetGrid

# Create or load a dataset
grid_dataset = WaveDatasetGrid(nx=200, nt=100)

# Visualize it
visualize_dataset(
    grid_dataset,
    title="1D Wave Dataset",
    save_path="./data/dataset_viz.png"
)
```

The visualization includes:
- 3D scatter plot and surface plot
- 2D heatmap (x-t plane)
- Snapshots at different times
- Time evolution at different positions
- Solution distribution statistics

### Data Quality Verification

Use the enhanced visualization tools to verify your data quality:

```python
from dataset import create_heatmap_visualization, create_wave_animation, WaveDatasetGrid

# Create grid dataset
grid_dataset = WaveDatasetGrid(nx=200, nt=100)

# Create detailed heatmap with quality metrics
create_heatmap_visualization(
    grid_dataset,
    save_path="./data/quality_check.png"
)

# Create wave animation
create_wave_animation(
    grid_dataset,
    save_path="./data/wave.gif",
    fps=20,
    duration=5
)
```

**Heatmap visualization includes:**
- Wave field with contour lines
- Smooth heatmap view
- Gradient magnitude (shows wave propagation patterns)
- Time snapshots at multiple points
- Spatial frequency spectrum (FFT analysis)
- Comprehensive quality metrics:
  - Solution statistics (min, max, mean, std)
  - Smoothness indicator
  - Temporal and spatial variance
  - Symmetry checks
  - Domain and wave parameters

**Animation shows:**
- Real-time wave profile evolution
- Spatiotemporal heatmap with progress indicator
- Current time display
- Full wave dynamics over one period

These tools help you verify:
- Wave structure is physically correct
- No numerical artifacts or discontinuities
- Proper wave propagation
- Correct frequency content
- Symmetric behavior (if expected)

## Loss Functions

Both models use a combination of:

1. **Data Loss**: MSE between predicted and ground truth values
   ```
   L_data = ||u_pred - u_true||²
   ```

2. **Physics Loss**: Residual of the wave equation
   ```
   L_physics = ||∂²u/∂t² - c²∂²u/∂x²||²
   ```

Total loss: `L = λ_data * L_data + λ_physics * L_physics`

Customize the loss functions in `train.py`:
- `loss_function_model1()` for Model 1
- `loss_function_model2()` for Model 2

## Outputs

After training, the following files are generated:

### Checkpoints (saved in `checkpoints/`)
- `model1_epoch_*.pt`: Model 1 checkpoints
- `model2_epoch_*.pt`: Model 2 checkpoints
- `model1_final.pt`: Final Model 1 checkpoint
- `model2_final.pt`: Final Model 2 checkpoint

### Visualizations (saved in `results/`)
- `model1_history.png`: Training loss curves for Model 1
- `model1_predictions.png`: Wave visualization (3D and 2D)
- `model1_errors.png`: Error analysis
- Similar files for Model 2

## Visualization

The project includes three types of plots:

1. **Training History**: Loss curves over epochs
   - Total loss, data loss, physics loss
   - Train vs test loss

2. **Wave Predictions**: Comparison of ground truth vs predictions
   - 3D surface plots
   - 2D heatmaps
   - Snapshots at different times

3. **Error Analysis**: Model performance evaluation
   - Absolute and relative error heatmaps
   - Error distribution histogram
   - Time-slice comparisons

## Customization

### Modify the wave equation solution
Edit the `analytical_solution()` function in `dataset.py`:
```python
def analytical_solution(x, t, c=1.0):
    # Your custom solution here
    return u
```

### Change model architecture
Edit `MODEL1_CONFIG` or `MODEL2_CONFIG` in `config.py`:
```python
MODEL1_CONFIG = {
    'hidden_layers': [64, 64, 64],  # Number and size of hidden layers
    'activation': 'tanh',           # 'tanh', 'relu', or 'sigmoid'
    'learning_rate': 1e-3,
    'epochs': 10000,
    'batch_size': 256,
}
```

### Customize loss functions
Edit `loss_function_model1()` or `loss_function_model2()` in `train.py`:
```python
def loss_function_model1(model, x, t, u_true):
    # Your custom loss computation
    return loss, loss_dict
```

## Theory: Physics-Informed Neural Networks

PINNs embed the physics of the problem (wave equation) directly into the loss function:

1. The neural network learns to approximate `u(x,t)`
2. Automatic differentiation computes `∂²u/∂t²` and `∂²u/∂x²`
3. The physics loss penalizes violations of the wave equation
4. The data loss ensures the network fits the training data

This approach allows the network to:
- Learn from sparse data
- Satisfy physical constraints
- Generalize better to unseen regions

## Requirements

- Python 3.8+
- PyTorch 2.0+
- NumPy
- Matplotlib

See `requirements.txt` for full list.

## License

This project is provided as-is for educational and research purposes.

## Next Steps

1. Train the models: `python train.py`
2. Customize the loss functions for your specific requirements
3. Experiment with different architectures and hyperparameters
4. Modify the analytical solution to match your problem
5. Add boundary conditions or initial conditions if needed

## Troubleshooting

**Issue**: Import errors
- **Solution**: Install dependencies with `pip install -r requirements.txt`

**Issue**: CUDA out of memory
- **Solution**: Reduce batch size or model size in `config.py`

**Issue**: Poor convergence
- **Solution**:
  - Adjust learning rate
  - Change loss function weights
  - Increase number of training points
  - Try different activation functions

## References

- [Physics-Informed Neural Networks (PINNs)](https://www.sciencedirect.com/science/article/pii/S0021999118307125)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
