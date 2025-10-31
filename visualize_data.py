"""
Quick script to visualize and verify wave dataset quality
Run this to check if your data looks correct before training
"""
import torch
from dataset import (
    get_dataloaders,
    visualize_dataset,
    create_heatmap_visualization,
    create_wave_animation,
    WaveDatasetGrid
)

def main():
    print("=" * 70)
    print("WAVE DATASET QUALITY VERIFICATION")
    print("=" * 70)

    # Load or generate datasets
    print("\nLoading datasets...")
    train_loader, test_loader, grid_dataset = get_dataloaders(
        load_from_disk=True,
        save_to_disk=True
    )

    print(f"\n✓ Loaded datasets:")
    print(f"  - Training: {len(train_loader.dataset):,} points")
    print(f"  - Testing: {len(test_loader.dataset):,} points")
    print(f"  - Grid: {grid_dataset.nx} × {grid_dataset.nt} = {len(grid_dataset):,} points")

    # Create visualizations
    print("\n" + "-" * 70)
    print("CREATING VISUALIZATIONS")
    print("-" * 70)

    # 1. Detailed heatmap with quality metrics
    print("\n1. Creating heatmap with data quality metrics...")
    print("   This shows:")
    print("   - Main wave field with contours")
    print("   - Smooth heatmap view")
    print("   - Gradient magnitude (wave propagation)")
    print("   - Time snapshots")
    print("   - Frequency analysis")
    print("   - Quality statistics")
    create_heatmap_visualization(
        grid_dataset,
        save_path="./data/heatmap_quality_check.png"
    )

    # 2. Wave animation
    print("\n2. Creating wave animation...")
    print("   This shows the wave evolution over time")
    print("   Duration: 5 seconds at 20 fps")
    create_wave_animation(
        grid_dataset,
        save_path="./data/wave_animation.gif",
        fps=20,
        duration=5
    )

    # 3. General dataset visualization
    print("\n3. Creating general dataset visualization...")
    visualize_dataset(
        grid_dataset,
        title="Grid Dataset - Complete Visualization",
        save_path="./data/grid_complete_viz.png"
    )

    # Data quality summary
    print("\n" + "=" * 70)
    print("DATA QUALITY SUMMARY")
    print("=" * 70)

    u_np = grid_dataset.u.cpu().numpy().flatten()
    U = u_np.reshape(grid_dataset.nt, grid_dataset.nx)

    print(f"\n✓ Solution statistics:")
    print(f"  - Min value: {U.min():.6f}")
    print(f"  - Max value: {U.max():.6f}")
    print(f"  - Mean: {U.mean():.6f}")
    print(f"  - Std Dev: {U.std():.6f}")

    print(f"\n✓ Domain:")
    print(f"  - x range: [{grid_dataset.X.min():.2f}, {grid_dataset.X.max():.2f}]")
    print(f"  - t range: [{grid_dataset.T.min():.2f}, {grid_dataset.T.max():.2f}]")

    print("\n" + "=" * 70)
    print("VERIFICATION COMPLETE!")
    print("=" * 70)
    print("\nGenerated files in ./data/:")
    print("  1. heatmap_quality_check.png  - Comprehensive quality analysis")
    print("  2. wave_animation.gif         - Animated wave evolution")
    print("  3. grid_complete_viz.png      - Complete dataset visualization")
    print("\nNext steps:")
    print("  1. Check the heatmap to verify the wave looks correct")
    print("  2. Watch the animation to see wave evolution")
    print("  3. If everything looks good, run: python train.py")


if __name__ == "__main__":
    main()
