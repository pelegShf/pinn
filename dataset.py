"""
Dataset generation for 1D Wave Equation
Generates triplets (x, t, u(x,t)) where u satisfies the wave equation:
∂²u/∂t² = c² ∂²u/∂x²
"""
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import config
import os
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.animation import FuncAnimation, PillowWriter
from IPython.display import HTML


def analytical_solution(x, t, c=1.0):
    """
    Analytical solution to the 1D wave equation with initial condition u(x,0) = 0.

    Solution: u(x,t) = sin(k*x) * sin(ω*t)

    This creates a standing wave pattern that starts from zero displacement.

    Initial conditions:
        u(x, 0) = 0              (zero initial displacement)
        ∂u/∂t(x, 0) = ω*sin(k*x) (non-zero initial velocity)

    Args:
        x: spatial coordinate (tensor or array)
        t: temporal coordinate (tensor or array)
        c: wave speed (default: 1.0)

    Returns:
        u: solution value at (x,t)
    """
    # Wave parameters
    k = 2 * np.pi / config.WAVELENGTH  # wave number
    omega = k * c  # angular frequency (ω = kc)

    # Standing wave solution with u(x,0) = 0
    # This satisfies: ∂²u/∂t² = c² ∂²u/∂x²
    u = np.sin(k * x) * np.sin(omega * t)

    # Alternative solutions (commented out):
    # Traveling wave: u = np.sin(k * x - omega * t)
    # Standing wave with u(x,0) ≠ 0: u = np.sin(k * x) * np.cos(omega * t)

    return u


class WaveDataset(Dataset):
    """
    Dataset for 1D Wave Equation
    Generates (x, t, u(x,t)) triplets
    """

    def __init__(self, n_points, x_range=None, t_range=None, device=None):
        """
        Initialize dataset

        Args:
            n_points: number of collocation points
            x_range: tuple (x_min, x_max), defaults to config values
            t_range: tuple (t_min, t_max), defaults to config values
            device: torch device
        """
        if x_range is None:
            x_range = (config.X_MIN, config.X_MAX)
        if t_range is None:
            t_range = (config.T_MIN, config.T_MAX)
        if device is None:
            device = config.DEVICE

        self.device = device
        self.x_range = x_range
        self.t_range = t_range

        # Generate collocation points uniformly in the domain
        x = np.random.uniform(x_range[0], x_range[1], n_points)
        t = np.random.uniform(t_range[0], t_range[1], n_points)

        # Compute analytical solution
        u = analytical_solution(x, t, c=config.WAVE_SPEED)

        # Convert to tensors
        self.x = torch.FloatTensor(x).reshape(-1, 1).to(device)
        self.t = torch.FloatTensor(t).reshape(-1, 1).to(device)
        self.u = torch.FloatTensor(u).reshape(-1, 1).to(device)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.t[idx], self.u[idx]

    def save(self, filepath):
        """
        Save dataset to disk

        Args:
            filepath: path to save the dataset
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        data = {
            'x': self.x.cpu(),
            't': self.t.cpu(),
            'u': self.u.cpu(),
            'x_range': self.x_range,
            't_range': self.t_range,
            'n_points': len(self.x),
            'wave_speed': config.WAVE_SPEED,
            'wavelength': config.WAVELENGTH,
        }
        torch.save(data, filepath)
        print(f"Dataset saved to {filepath}")

    @classmethod
    def load(cls, filepath, device=None):
        """
        Load dataset from disk

        Args:
            filepath: path to the saved dataset
            device: device to load tensors to

        Returns:
            WaveDataset instance
        """
        if device is None:
            device = config.DEVICE

        data = torch.load(filepath, weights_only=False)

        # Create empty dataset
        dataset = cls.__new__(cls)
        dataset.device = device
        dataset.x_range = data['x_range']
        dataset.t_range = data['t_range']

        # Load tensors
        dataset.x = data['x'].to(device)
        dataset.t = data['t'].to(device)
        dataset.u = data['u'].to(device)

        print(f"Dataset loaded from {filepath}")
        print(f"  Points: {len(dataset)}")
        print(f"  x range: [{dataset.x_range[0]}, {dataset.x_range[1]}]")
        print(f"  t range: [{dataset.t_range[0]}, {dataset.t_range[1]}]")

        return dataset


class WaveDatasetGrid(Dataset):
    """
    Dataset on a regular grid for visualization
    """

    def __init__(self, nx=100, nt=100, x_range=None, t_range=None, device=None):
        """
        Initialize grid dataset

        Args:
            nx: number of spatial points
            nt: number of temporal points
            x_range: tuple (x_min, x_max)
            t_range: tuple (t_min, t_max)
            device: torch device
        """
        if x_range is None:
            x_range = (config.X_MIN, config.X_MAX)
        if t_range is None:
            t_range = (config.T_MIN, config.T_MAX)
        if device is None:
            device = config.DEVICE

        self.device = device

        # Create meshgrid
        x_vals = np.linspace(x_range[0], x_range[1], nx)
        t_vals = np.linspace(t_range[0], t_range[1], nt)
        X, T = np.meshgrid(x_vals, t_vals)

        # Flatten
        x = X.flatten()
        t = T.flatten()

        # Compute analytical solution
        u = analytical_solution(x, t, c=config.WAVE_SPEED)

        # Convert to tensors
        self.x = torch.FloatTensor(x).reshape(-1, 1).to(device)
        self.t = torch.FloatTensor(t).reshape(-1, 1).to(device)
        self.u = torch.FloatTensor(u).reshape(-1, 1).to(device)

        # Store grid dimensions for reshaping
        self.nx = nx
        self.nt = nt
        self.X = X
        self.T = T

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.t[idx], self.u[idx]

    def save(self, filepath):
        """Save grid dataset to disk"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        data = {
            'x': self.x.cpu(),
            't': self.t.cpu(),
            'u': self.u.cpu(),
            'nx': self.nx,
            'nt': self.nt,
            'X': self.X,
            'T': self.T,
        }
        torch.save(data, filepath)
        print(f"Grid dataset saved to {filepath}")

    @classmethod
    def load(cls, filepath, device=None):
        """Load grid dataset from disk"""
        if device is None:
            device = config.DEVICE

        data = torch.load(filepath, weights_only=False)

        dataset = cls.__new__(cls)
        dataset.device = device
        dataset.x = data['x'].to(device)
        dataset.t = data['t'].to(device)
        dataset.u = data['u'].to(device)
        dataset.nx = data['nx']
        dataset.nt = data['nt']
        dataset.X = data['X']
        dataset.T = data['T']

        print(f"Grid dataset loaded from {filepath} ({dataset.nx}x{dataset.nt} grid)")
        return dataset


def get_dataloaders(batch_size=None, load_from_disk=False, save_to_disk=False, data_dir='./data'):
    """
    Create train and test dataloaders with option to save/load from disk

    Args:
        batch_size: batch size for dataloaders
        load_from_disk: if True, load datasets from disk
        save_to_disk: if True, save generated datasets to disk
        data_dir: directory to save/load datasets

    Returns:
        train_loader, test_loader, test_grid_dataset
    """
    if batch_size is None:
        batch_size = config.MODEL1_CONFIG['batch_size']

    data_path = Path(data_dir)
    data_path.mkdir(exist_ok=True)

    train_path = data_path / 'train_dataset.pt'
    test_path = data_path / 'test_dataset.pt'
    grid_path = data_path / 'grid_dataset.pt'

    # Load or create datasets
    if load_from_disk and train_path.exists() and test_path.exists() and grid_path.exists():
        print("Loading datasets from disk...")
        train_dataset = WaveDataset.load(train_path)
        test_dataset = WaveDataset.load(test_path)
        test_grid_dataset = WaveDatasetGrid.load(grid_path)
    else:
        print("Generating datasets...")
        train_dataset = WaveDataset(config.N_TRAIN_POINTS)
        test_dataset = WaveDataset(config.N_TEST_POINTS)
        test_grid_dataset = WaveDatasetGrid(nx=200, nt=100)

        # Save if requested
        if save_to_disk:
            print("Saving datasets to disk...")
            train_dataset.save(train_path)
            test_dataset.save(test_path)
            test_grid_dataset.save(grid_path)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, test_grid_dataset


def visualize_dataset(dataset, title="Dataset Visualization", save_path=None):
    """
    Visualize the dataset

    Args:
        dataset: WaveDataset or WaveDatasetGrid instance
        title: plot title
        save_path: path to save the figure
    """
    # Convert to numpy
    x_np = dataset.x.cpu().numpy().flatten()
    t_np = dataset.t.cpu().numpy().flatten()
    u_np = dataset.u.cpu().numpy().flatten()

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(title, fontsize=16)

    # 3D scatter plot
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    scatter = ax1.scatter(x_np, t_np, u_np, c=u_np, cmap='viridis', s=1, alpha=0.6)
    ax1.set_xlabel('x')
    ax1.set_ylabel('t')
    ax1.set_zlabel('u(x,t)')
    ax1.set_title('3D Scatter Plot')
    fig.colorbar(scatter, ax=ax1, shrink=0.5)

    # If it's a grid dataset, show 2D heatmap
    if isinstance(dataset, WaveDatasetGrid):
        U = u_np.reshape(dataset.nt, dataset.nx)
        X = dataset.X
        T = dataset.T

        # 2D heatmap
        ax2 = fig.add_subplot(2, 3, 2)
        im = ax2.contourf(X, T, U, levels=50, cmap='viridis')
        ax2.set_xlabel('x')
        ax2.set_ylabel('t')
        ax2.set_title('2D Heatmap')
        fig.colorbar(im, ax=ax2)

        # 3D surface
        ax3 = fig.add_subplot(2, 3, 3, projection='3d')
        surf = ax3.plot_surface(X, T, U, cmap='viridis', alpha=0.8)
        ax3.set_xlabel('x')
        ax3.set_ylabel('t')
        ax3.set_zlabel('u(x,t)')
        ax3.set_title('3D Surface')
        fig.colorbar(surf, ax=ax3, shrink=0.5)

        # Snapshots at different times
        ax4 = fig.add_subplot(2, 3, 4)
        time_indices = [0, dataset.nt // 3, 2 * dataset.nt // 3, dataset.nt - 1]
        for idx in time_indices:
            t_val = T[idx, 0]
            ax4.plot(X[idx, :], U[idx, :], label=f't={t_val:.3f}', linewidth=2)
        ax4.set_xlabel('x')
        ax4.set_ylabel('u(x,t)')
        ax4.set_title('Snapshots at Different Times')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # Snapshots at different positions
        ax5 = fig.add_subplot(2, 3, 5)
        pos_indices = [0, dataset.nx // 3, 2 * dataset.nx // 3, dataset.nx - 1]
        for idx in pos_indices:
            x_val = X[0, idx]
            ax5.plot(T[:, idx], U[:, idx], label=f'x={x_val:.3f}', linewidth=2)
        ax5.set_xlabel('t')
        ax5.set_ylabel('u(x,t)')
        ax5.set_title('Time Evolution at Different Positions')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
    else:
        # For random collocation points, show 2D scatter plots
        ax2 = fig.add_subplot(2, 3, 2)
        scatter2 = ax2.scatter(x_np, t_np, c=u_np, cmap='viridis', s=1, alpha=0.5)
        ax2.set_xlabel('x')
        ax2.set_ylabel('t')
        ax2.set_title('Collocation Points (x-t plane)')
        fig.colorbar(scatter2, ax=ax2)

        ax3 = fig.add_subplot(2, 3, 3)
        scatter3 = ax3.scatter(x_np, u_np, c=t_np, cmap='plasma', s=1, alpha=0.5)
        ax3.set_xlabel('x')
        ax3.set_ylabel('u(x,t)')
        ax3.set_title('Solution vs x (colored by t)')
        fig.colorbar(scatter3, ax=ax3, label='t')

        ax4 = fig.add_subplot(2, 3, 4)
        scatter4 = ax4.scatter(t_np, u_np, c=x_np, cmap='plasma', s=1, alpha=0.5)
        ax4.set_xlabel('t')
        ax4.set_ylabel('u(x,t)')
        ax4.set_title('Solution vs t (colored by x)')
        fig.colorbar(scatter4, ax=ax4, label='x')

    # Distribution plots
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.hist(u_np, bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax6.set_xlabel('u(x,t)')
    ax6.set_ylabel('Frequency')
    ax6.set_title('Solution Distribution')
    ax6.grid(True, alpha=0.3)

    # Add statistics
    stats_text = f"Statistics:\n"
    stats_text += f"Points: {len(dataset)}\n"
    stats_text += f"u min: {u_np.min():.4f}\n"
    stats_text += f"u max: {u_np.max():.4f}\n"
    stats_text += f"u mean: {u_np.mean():.4f}\n"
    stats_text += f"u std: {u_np.std():.4f}"

    ax6.text(0.02, 0.98, stats_text, transform=ax6.transAxes,
             fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")

    plt.show()


def create_heatmap_visualization(grid_dataset, save_path=None):
    """
    Create detailed heatmap visualization for data quality verification

    Args:
        grid_dataset: WaveDatasetGrid instance
        save_path: path to save the figure
    """
    if not isinstance(grid_dataset, WaveDatasetGrid):
        print("Error: This function requires a WaveDatasetGrid instance")
        return

    # Convert to numpy
    x_np = grid_dataset.x.cpu().numpy().flatten()
    t_np = grid_dataset.t.cpu().numpy().flatten()
    u_np = grid_dataset.u.cpu().numpy().flatten()

    # Reshape to grid
    U = u_np.reshape(grid_dataset.nt, grid_dataset.nx)
    X = grid_dataset.X
    T = grid_dataset.T

    # Create figure with multiple heatmaps
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle('Wave Dataset Quality Check - Heatmap Analysis', fontsize=16, fontweight='bold')

    # Main heatmap with contours
    ax1 = plt.subplot(2, 3, 1)
    im1 = ax1.contourf(X, T, U, levels=100, cmap='RdBu_r')
    contours = ax1.contour(X, T, U, levels=10, colors='black', alpha=0.3, linewidths=0.5)
    ax1.clabel(contours, inline=True, fontsize=8)
    ax1.set_xlabel('Position (x)', fontsize=12)
    ax1.set_ylabel('Time (t)', fontsize=12)
    ax1.set_title('Wave Field u(x,t)', fontsize=14, fontweight='bold')
    cbar1 = fig.colorbar(im1, ax=ax1)
    cbar1.set_label('u(x,t)', fontsize=10)
    ax1.grid(True, alpha=0.2)

    # Heatmap without contours (cleaner view)
    ax2 = plt.subplot(2, 3, 2)
    im2 = ax2.imshow(U, aspect='auto', cmap='viridis',
                     extent=[X.min(), X.max(), T.min(), T.max()],
                     origin='lower', interpolation='bilinear')
    ax2.set_xlabel('Position (x)', fontsize=12)
    ax2.set_ylabel('Time (t)', fontsize=12)
    ax2.set_title('Wave Field (Smooth)', fontsize=14, fontweight='bold')
    cbar2 = fig.colorbar(im2, ax=ax2)
    cbar2.set_label('u(x,t)', fontsize=10)

    # Gradient magnitude (shows wave propagation)
    du_dx = np.gradient(U, axis=1)
    du_dt = np.gradient(U, axis=0)
    gradient_mag = np.sqrt(du_dx**2 + du_dt**2)

    ax3 = plt.subplot(2, 3, 3)
    im3 = ax3.imshow(gradient_mag, aspect='auto', cmap='hot',
                     extent=[X.min(), X.max(), T.min(), T.max()],
                     origin='lower', interpolation='bilinear')
    ax3.set_xlabel('Position (x)', fontsize=12)
    ax3.set_ylabel('Time (t)', fontsize=12)
    ax3.set_title('Gradient Magnitude (Wave Propagation)', fontsize=14, fontweight='bold')
    cbar3 = fig.colorbar(im3, ax=ax3)
    cbar3.set_label('|∇u|', fontsize=10)

    # Time slices
    ax4 = plt.subplot(2, 3, 4)
    n_slices = 5
    time_indices = np.linspace(0, grid_dataset.nt - 1, n_slices, dtype=int)
    colors = plt.cm.viridis(np.linspace(0, 1, n_slices))

    for idx, color in zip(time_indices, colors):
        t_val = T[idx, 0]
        ax4.plot(X[idx, :], U[idx, :], linewidth=2, color=color,
                label=f't={t_val:.3f}', alpha=0.8)
    ax4.set_xlabel('Position (x)', fontsize=12)
    ax4.set_ylabel('u(x,t)', fontsize=12)
    ax4.set_title('Wave Snapshots at Different Times', fontsize=14, fontweight='bold')
    ax4.legend(loc='upper right', fontsize=9)
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color='k', linestyle='--', alpha=0.3)

    # Spatial frequency analysis
    ax5 = plt.subplot(2, 3, 5)
    # Take FFT of a middle time slice
    mid_idx = grid_dataset.nt // 2
    u_mid = U[mid_idx, :]
    fft_vals = np.fft.fft(u_mid)
    fft_freq = np.fft.fftfreq(len(u_mid), d=(X[0, 1] - X[0, 0]))
    fft_power = np.abs(fft_vals)**2

    # Only plot positive frequencies
    pos_mask = fft_freq > 0
    ax5.semilogy(fft_freq[pos_mask], fft_power[pos_mask], linewidth=2, color='blue')
    ax5.set_xlabel('Spatial Frequency (1/x)', fontsize=12)
    ax5.set_ylabel('Power', fontsize=12)
    ax5.set_title(f'Spatial Frequency Spectrum (t={T[mid_idx, 0]:.3f})', fontsize=14, fontweight='bold')
    ax5.grid(True, alpha=0.3)

    # Statistics and data quality metrics
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')

    # Compute statistics
    u_min, u_max = U.min(), U.max()
    u_mean, u_std = U.mean(), U.std()
    u_range = u_max - u_min

    # Check for symmetry
    symmetry_x = np.mean(np.abs(U - U[:, ::-1])) / u_range  # left-right symmetry

    # Check for smoothness (via gradient)
    smoothness = gradient_mag.mean()

    # Temporal variance
    temporal_var = np.var(U, axis=0).mean()
    spatial_var = np.var(U, axis=1).mean()

    stats_text = "DATA QUALITY METRICS\n" + "=" * 40 + "\n\n"
    stats_text += f"Grid Size: {grid_dataset.nx} × {grid_dataset.nt}\n"
    stats_text += f"Total Points: {grid_dataset.nx * grid_dataset.nt:,}\n\n"

    stats_text += "SOLUTION STATISTICS:\n"
    stats_text += f"  Min value: {u_min:.6f}\n"
    stats_text += f"  Max value: {u_max:.6f}\n"
    stats_text += f"  Mean: {u_mean:.6f}\n"
    stats_text += f"  Std Dev: {u_std:.6f}\n"
    stats_text += f"  Range: {u_range:.6f}\n\n"

    stats_text += "QUALITY INDICATORS:\n"
    stats_text += f"  Smoothness: {smoothness:.6f}\n"
    stats_text += f"  Temporal Variance: {temporal_var:.6f}\n"
    stats_text += f"  Spatial Variance: {spatial_var:.6f}\n"
    stats_text += f"  Symmetry Error: {symmetry_x:.6f}\n\n"

    stats_text += "DOMAIN:\n"
    stats_text += f"  x ∈ [{X.min():.2f}, {X.max():.2f}]\n"
    stats_text += f"  t ∈ [{T.min():.2f}, {T.max():.2f}]\n\n"

    stats_text += "WAVE PARAMETERS:\n"
    stats_text += f"  Wave speed (c): {config.WAVE_SPEED}\n"
    stats_text += f"  Wavelength (λ): {config.WAVELENGTH}\n"
    stats_text += f"  Wave number (k): {2*np.pi/config.WAVELENGTH:.4f}\n"
    stats_text += f"  Angular freq (ω): {2*np.pi/config.WAVELENGTH * config.WAVE_SPEED:.4f}\n"

    ax6.text(0.05, 0.95, stats_text, transform=ax6.transAxes,
             fontsize=10, verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Heatmap visualization saved to {save_path}")

    plt.show()


def create_wave_animation(grid_dataset, save_path='wave_animation.gif', fps=20, duration=5):
    """
    Create an animated video of the wave evolution

    Args:
        grid_dataset: WaveDatasetGrid instance
        save_path: path to save the animation (supports .gif, .mp4)
        fps: frames per second
        duration: duration in seconds
    """
    if not isinstance(grid_dataset, WaveDatasetGrid):
        print("Error: This function requires a WaveDatasetGrid instance")
        return

    print(f"Creating wave animation...")
    print(f"  Duration: {duration}s at {fps} fps = {duration * fps} frames")

    # Convert to numpy
    u_np = grid_dataset.u.cpu().numpy().flatten()
    U = u_np.reshape(grid_dataset.nt, grid_dataset.nx)
    X = grid_dataset.X
    T = grid_dataset.T

    # Select frames to display
    n_frames = min(duration * fps, grid_dataset.nt)
    frame_indices = np.linspace(0, grid_dataset.nt - 1, n_frames, dtype=int)

    # Set up the figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle('1D Wave Equation Solution', fontsize=16, fontweight='bold')

    # Initialize plots
    x_line = X[0, :]
    u_min, u_max = U.min(), U.max()
    padding = (u_max - u_min) * 0.1

    # Top plot: wave at current time
    line1, = ax1.plot([], [], 'b-', linewidth=3, label='u(x,t)')
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3, linewidth=1)
    ax1.set_xlim(X.min(), X.max())
    ax1.set_ylim(u_min - padding, u_max + padding)
    ax1.set_xlabel('Position (x)', fontsize=12)
    ax1.set_ylabel('Displacement u(x,t)', fontsize=12)
    ax1.set_title('Wave Profile', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', fontsize=10)

    # Add time text
    time_text = ax1.text(0.02, 0.95, '', transform=ax1.transAxes,
                        fontsize=12, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Bottom plot: spatiotemporal heatmap with progress line
    im = ax2.imshow(U, aspect='auto', cmap='RdBu_r',
                   extent=[X.min(), X.max(), T.min(), T.max()],
                   origin='lower', interpolation='bilinear',
                   vmin=u_min, vmax=u_max)
    progress_line, = ax2.plot([], [], 'r-', linewidth=2, label='Current time')
    ax2.set_xlabel('Position (x)', fontsize=12)
    ax2.set_ylabel('Time (t)', fontsize=12)
    ax2.set_title('Spatiotemporal Evolution', fontsize=14)
    cbar = fig.colorbar(im, ax=ax2)
    cbar.set_label('u(x,t)', fontsize=10)
    ax2.legend(loc='upper right', fontsize=10)

    plt.tight_layout()

    def init():
        """Initialize animation"""
        line1.set_data([], [])
        progress_line.set_data([], [])
        time_text.set_text('')
        return line1, progress_line, time_text

    def animate(frame):
        """Animation function"""
        idx = frame_indices[frame]
        t_val = T[idx, 0]

        # Update wave profile
        line1.set_data(x_line, U[idx, :])

        # Update time text
        time_text.set_text(f't = {t_val:.4f}')

        # Update progress line on heatmap
        progress_line.set_data([X.min(), X.max()], [t_val, t_val])

        return line1, progress_line, time_text

    # Create animation
    anim = FuncAnimation(fig, animate, init_func=init,
                        frames=len(frame_indices), interval=1000/fps,
                        blit=True, repeat=True)

    # Save animation
    if save_path:
        if save_path.endswith('.gif'):
            writer = PillowWriter(fps=fps)
            anim.save(save_path, writer=writer)
            print(f"Animation saved as GIF: {save_path}")
        elif save_path.endswith('.mp4'):
            try:
                anim.save(save_path, writer='ffmpeg', fps=fps)
                print(f"Animation saved as MP4: {save_path}")
            except Exception as e:
                print(f"Could not save as MP4 (ffmpeg required): {e}")
                print("Saving as GIF instead...")
                gif_path = save_path.replace('.mp4', '.gif')
                writer = PillowWriter(fps=fps)
                anim.save(gif_path, writer=writer)
                print(f"Animation saved as GIF: {gif_path}")
        else:
            # Default to GIF
            writer = PillowWriter(fps=fps)
            anim.save(save_path, writer=writer)
            print(f"Animation saved: {save_path}")

    plt.show()
    return anim


if __name__ == "__main__":
    # Test dataset generation and visualization
    print("=" * 60)
    print("TESTING DATASET GENERATION AND VISUALIZATION")
    print("=" * 60)

    # Generate datasets and save to disk
    print("\n1. Generating datasets...")
    train_loader, test_loader, grid_dataset = get_dataloaders(save_to_disk=True)

    print(f"\nTrain dataset size: {len(train_loader.dataset)}")
    print(f"Test dataset size: {len(test_loader.dataset)}")
    print(f"Grid dataset size: {len(grid_dataset)}")

    # Sample batch
    x, t, u = next(iter(train_loader))
    print(f"\nBatch shapes:")
    print(f"  x: {x.shape}, t: {t.shape}, u: {u.shape}")
    print(f"\nSample values:")
    print(f"  x[0]: {x[0].item():.4f}")
    print(f"  t[0]: {t[0].item():.4f}")
    print(f"  u[0]: {u[0].item():.4f}")

    # Test loading from disk
    print("\n2. Testing load from disk...")
    train_loader2, test_loader2, grid_dataset2 = get_dataloaders(load_from_disk=True)

    # Visualize datasets
    print("\n3. Visualizing datasets...")
    print("Visualizing training dataset (collocation points)...")
    visualize_dataset(
        train_loader.dataset,
        title="Training Dataset (Random Collocation Points)",
        save_path="./data/train_dataset_viz.png"
    )

    print("Visualizing grid dataset...")
    visualize_dataset(
        grid_dataset,
        title="Grid Dataset (Structured Grid)",
        save_path="./data/grid_dataset_viz.png"
    )

    # Create detailed heatmap for data quality check
    print("\n4. Creating heatmap visualization for data quality...")
    create_heatmap_visualization(
        grid_dataset,
        save_path="./data/heatmap_quality_check.png"
    )

    # Create wave animation
    print("\n5. Creating wave animation...")
    create_wave_animation(
        grid_dataset,
        save_path="./data/wave_animation.gif",
        fps=20,
        duration=5
    )

    print("\n" + "=" * 60)
    print("TESTING COMPLETED!")
    print("=" * 60)
    print("\nGenerated files:")
    print("  - ./data/train_dataset.pt")
    print("  - ./data/test_dataset.pt")
    print("  - ./data/grid_dataset.pt")
    print("  - ./data/train_dataset_viz.png")
    print("  - ./data/grid_dataset_viz.png")
    print("  - ./data/heatmap_quality_check.png")
    print("  - ./data/wave_animation.gif")
