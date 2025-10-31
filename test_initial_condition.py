"""
Test script to verify initial condition u(x,0) = 0
"""
import numpy as np
import matplotlib.pyplot as plt
from dataset import analytical_solution
import config

def test_initial_condition():
    """Verify that u(x, 0) = 0 for all x"""
    print("=" * 70)
    print("TESTING INITIAL CONDITION: u(x, 0) = 0")
    print("=" * 70)

    # Create x values
    x_vals = np.linspace(config.X_MIN, config.X_MAX, 1000)

    # Evaluate at t=0
    t = 0.0
    u_at_t0 = analytical_solution(x_vals, t, c=config.WAVE_SPEED)

    # Check if all values are approximately zero
    max_error = np.max(np.abs(u_at_t0))
    print(f"\nMaximum |u(x, 0)|: {max_error:.2e}")

    if max_error < 1e-10:
        print("✓ PASSED: Initial condition u(x, 0) = 0 is satisfied!")
    else:
        print("✗ FAILED: Initial condition is NOT zero")

    # Test at several time points
    print("\n" + "-" * 70)
    print("Testing solution at different times:")
    print("-" * 70)

    time_points = [0.0, 0.25, 0.5, 0.75, 1.0]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('Wave Evolution with u(x,0) = 0', fontsize=16, fontweight='bold')
    axes = axes.flatten()

    for i, t_val in enumerate(time_points):
        u = analytical_solution(x_vals, t_val, c=config.WAVE_SPEED)

        # Compute statistics
        u_min, u_max = u.min(), u.max()
        u_mean = u.mean()

        print(f"\nt = {t_val:.2f}:")
        print(f"  u_min = {u_min:.6f}")
        print(f"  u_max = {u_max:.6f}")
        print(f"  u_mean = {u_mean:.6f}")

        # Plot
        axes[i].plot(x_vals, u, 'b-', linewidth=2, label=f't={t_val:.2f}')
        axes[i].axhline(y=0, color='k', linestyle='--', alpha=0.3)
        axes[i].set_xlabel('x', fontsize=10)
        axes[i].set_ylabel('u(x,t)', fontsize=10)
        axes[i].set_title(f'Time t = {t_val:.2f}', fontsize=12, fontweight='bold')
        axes[i].grid(True, alpha=0.3)
        axes[i].legend()

        # Highlight t=0
        if t_val == 0.0:
            axes[i].set_facecolor('#ffe6e6')
            axes[i].text(0.5, 0.95, 'INITIAL CONDITION',
                        transform=axes[i].transAxes,
                        ha='center', va='top', fontsize=10,
                        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

    # Remove extra subplot
    fig.delaxes(axes[-1])

    plt.tight_layout()
    plt.savefig('./data/initial_condition_verification.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved to: ./data/initial_condition_verification.png")
    plt.show()

    # Verify wave equation properties
    print("\n" + "=" * 70)
    print("WAVE PROPERTIES")
    print("=" * 70)
    k = 2 * np.pi / config.WAVELENGTH
    omega = k * config.WAVE_SPEED
    period = 2 * np.pi / omega

    print(f"Wave number (k): {k:.4f}")
    print(f"Angular frequency (ω): {omega:.4f}")
    print(f"Period (T): {period:.4f}")
    print(f"Wavelength (λ): {config.WAVELENGTH:.4f}")
    print(f"Wave speed (c): {config.WAVE_SPEED:.4f}")

    # Verify at t=period (should return to zero)
    u_at_period = analytical_solution(x_vals, period, c=config.WAVE_SPEED)
    max_error_period = np.max(np.abs(u_at_period))

    print(f"\nAt t = T (one period):")
    print(f"  Maximum |u(x, T)|: {max_error_period:.2e}")
    if max_error_period < 1e-10:
        print("  ✓ Wave returns to zero after one period")

    print("\n" + "=" * 70)
    print("VERIFICATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    test_initial_condition()
