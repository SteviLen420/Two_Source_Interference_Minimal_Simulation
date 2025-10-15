# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Stefan Len
#
# ===================================================================================
# Two_Source_Interference_3D_Simulation.py
# ===================================================================================
# Author: Stefan Len
# Overview:
#   Minimal 3D simulation of interference from two coherent point sources.
#   Implements spherical wave propagation (1/r decay) and compares numerical
#   and analytical results for validation. Generates plots, CSV outputs,
#   and quantitative error summary automatically.
# ===================================================================================

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from datetime import datetime
import csv
import sys 
import json 

# ===================================================================================
# --- MASTER CONTROL ---
# ===================================================================================
GRID_HALF_SIZE = 100          # 3D grid: much smaller due to memory (100^3 = 1M points)
MAIN_D = 60                   # Source separation
MAIN_WAVELENGTH = 5.0        # Wavelength
L_OBSERVATION = 150           # Observation plane distance
PEAK_PROMINENCE_THRESHOLD = 0.1
OUTPUT_BASE_FOLDER = 'Interference_3D_Sims'
CODE_VERSION = '2.0.0'
# ===================================================================================

find_peaks = None 
try:
    from scipy.signal import find_peaks as initial_find_peaks
    find_peaks = initial_find_peaks
except ImportError:
    pass

def setup_colab_environment():
    global find_peaks
    print("--- 3D Environment Setup ---")
    
    if 'google.colab' in sys.modules:
        print("Colab detected.")
        
        if find_peaks is None:
            print("Installing scipy...")
            os.system('pip install scipy > /dev/null 2>&1')
            try:
                from scipy.signal import find_peaks as re_import_find_peaks
                find_peaks = re_import_find_peaks
                print("scipy ready.")
            except:
                print("WARNING: scipy install failed.")

        try:
            from google.colab import drive
            drive.mount('/content/drive')
            print("Drive mounted.")
            return True, '/content/drive/MyDrive'
        except Exception as e:
            print(f"Drive mount failed: {e}")
            return False, None
    else:
        print("Local execution.")
        return False, '.'

def create_output_directory(base_path):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(base_path, OUTPUT_BASE_FOLDER, f'run_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output: {output_dir}")
    return output_dir

def save_metadata(save_path, grid_size, d, wavelength):
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'code_version': CODE_VERSION,
        'simulation_parameters': {
            'grid_half_size': grid_size,
            'source_separation_d': d,
            'wavelength': wavelength,
            'dimensionality': '3D'
        },
        'validation_parameters': {
            'observation_plane_distance': L_OBSERVATION,
            'peak_detection_prominence_threshold': PEAK_PROMINENCE_THRESHOLD,
            'analytical_model': '3D_spherical_wave_1_over_r'
        }
    }
    with open(os.path.join(save_path, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

def simulate_interference_3D(grid_size, d, wavelength, save_path=None):
    """
    3D spherical wave interference simulation.
    
    Sources positioned along Z-axis at (0, 0, ±d/2).
    Observation plane at X = L_OBSERVATION.
    """
    print(f"\n--- 3D Simulation: d={d}, lambda={wavelength} ---")
    print(f"   Grid: [{-grid_size}, {grid_size}]^3 = {(2*grid_size)**3:,} points")

    # Create 3D grid
    x = np.arange(-grid_size, grid_size, 1)
    y = np.arange(-grid_size, grid_size, 1)
    z = np.arange(-grid_size, grid_size, 1)
    
    # For memory efficiency, compute only the observation plane (X = L_OBSERVATION)
    # Instead of full 3D volume
    x_obs_idx = np.argmin(np.abs(x - L_OBSERVATION))
    x_obs = x[x_obs_idx]
    
    print(f"   Computing observation plane at X = {x_obs}")
    
    # Create 2D meshgrid for observation plane (Y, Z)
    yy, zz = np.meshgrid(y, z)
    xx = np.full_like(yy, x_obs)  # Constant X = L_OBSERVATION
    
    # Source positions along Z-axis
    source1_pos = np.array([0, 0, -d/2])
    source2_pos = np.array([0, 0, d/2])
    
    # Calculate 3D distances from sources to observation plane
    dist1 = np.sqrt((xx - source1_pos[0])**2 + 
                    (yy - source1_pos[1])**2 + 
                    (zz - source1_pos[2])**2)
    
    dist2 = np.sqrt((xx - source2_pos[0])**2 + 
                    (yy - source2_pos[1])**2 + 
                    (zz - source2_pos[2])**2)
    
    # Avoid singularity
    dist1 = np.where(dist1 < 1, 1, dist1)
    dist2 = np.where(dist2 < 1, 1, dist2)
    
    k = 2 * np.pi / wavelength
    
    # 3D spherical waves: amplitude ∝ 1/r
    amplitude1 = np.exp(1j * k * dist1) / dist1
    amplitude2 = np.exp(1j * k * dist2) / dist2
    
    # Time-averaged intensity
    total_field = amplitude1 + amplitude2
    intensity_plane = np.abs(total_field)**2
    
    # Normalize
    intensity_plane = intensity_plane / intensity_plane.max() * 4

    if save_path:
        # Save 2D observation plane as image
        plt.figure(figsize=(10, 10))
        plt.imshow(intensity_plane, origin='lower', 
                   extent=[y[0], y[-1], z[0], z[-1]], 
                   cmap='viridis', aspect='equal')
        plt.plot(0, -d/2, 'ro', markersize=8, label='Source 1', 
                 transform=plt.gca().transData)
        plt.plot(0, d/2, 'ro', markersize=8, label='Source 2')
        plt.title(f'3D Interference - Observation Plane (X={x_obs}, d={d}, λ={wavelength})', 
                  fontsize=14)
        plt.xlabel('Y Position (Grid Units)', fontsize=12)
        plt.ylabel('Z Position (Grid Units)', fontsize=12)
        plt.colorbar(label='Intensity')
        plt.legend()
        plt.grid(False)
        
        plt.savefig(os.path.join(save_path, f'3D_obs_plane_d{d}_l{wavelength}.png'), 
                    bbox_inches='tight', dpi=300)
        
        # Save as CSV
        np.savetxt(os.path.join(save_path, f'3D_intensity_plane_d{d}_l{wavelength}.csv'),
                   intensity_plane, delimiter=",", fmt='%1.4e')
        
        print("   3D observation plane saved.")
        plt.close()
        
        # Create cross-section plot (Z = 0 slice)
        z_center_idx = len(z) // 2
        intensity_slice = intensity_plane[z_center_idx, :]
        
        plt.figure(figsize=(10, 6))
        plt.plot(y, intensity_slice, 'b-', linewidth=2)
        plt.title(f'3D Cross-Section (Z=0, X={x_obs})', fontsize=14)
        plt.xlabel('Y Position', fontsize=12)
        plt.ylabel('Intensity', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(save_path, f'3D_cross_section_d{d}_l{wavelength}.png'), 
                    bbox_inches='tight', dpi=300)
        plt.close()

    return y, z, intensity_plane

def analytical_intensity_3D_spherical(y_pos, z_pos, d, wavelength, L):
    """
    Analytical intensity for 3D spherical wave interference.
    Sources at (0, 0, ±d/2), observation at (L, y, z).
    """
    k = 2 * np.pi / wavelength
    
    # 3D distances
    r1 = np.sqrt(L**2 + y_pos**2 + (z_pos + d/2)**2)
    r2 = np.sqrt(L**2 + y_pos**2 + (z_pos - d/2)**2)
    
    # 3D spherical waves: amplitude ∝ 1/r
    A1 = np.exp(1j * k * r1) / r1
    A2 = np.exp(1j * k * r2) / r2
    
    intensity = np.abs(A1 + A2)**2
    intensity = intensity / intensity.max() * 4
    
    return intensity

def create_validation_plot_3D(grid_size, d, wavelength, save_path):
    """
    Validate 3D simulation against analytical solution.
    Compare central Z=0 cross-section.
    """
    print("\n--- 3D Validation Plot ---")
    
    y_coords, z_coords, intensity_plane = simulate_interference_3D(
        grid_size=grid_size, d=d, wavelength=wavelength, save_path=None
    )
    
    # Extract central Z=0 cross-section
    z_center_idx = len(z_coords) // 2
    numerical = intensity_plane[z_center_idx, :]
    
    # Create 2D meshgrid for analytical (Y, Z=0)
    yy_1d = y_coords
    zz_1d = np.zeros_like(yy_1d)  # Z = 0
    
    analytical = analytical_intensity_3D_spherical(yy_1d, zz_1d, d, wavelength, L=L_OBSERVATION)
    
    numerical_norm = numerical / numerical.max() * 4
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, 
                                   gridspec_kw={'height_ratios': [3, 1]})
    
    ax1.plot(y_coords, numerical_norm, 'b-', label='Numerical (3D)', linewidth=2)
    ax1.plot(yy_1d, analytical, 'r--', label='Analytical (Spherical)', linewidth=2)
    ax1.legend()
    ax1.set_title(f'3D Validation (Z=0 slice, X={L_OBSERVATION}, d={d}, λ={wavelength})', 
                  fontsize=14)
    ax1.set_ylabel('Normalized Intensity', fontsize=12)
    
    difference = numerical_norm - analytical
    ax2.plot(y_coords, difference, 'g-', linewidth=1)
    ax2.axhline(0, color='k', linestyle='--', linewidth=0.8)
    ax2.set_xlabel('Y Position', fontsize=12)
    ax2.set_ylabel('Difference', fontsize=12)
    ax2.grid(True, linestyle=':', alpha=0.6)
    
    plt.savefig(os.path.join(save_path, '3D_validation_plot.png'), 
                bbox_inches='tight', dpi=300)
    print("   3D validation plot saved.")
    plt.close(fig)

def measure_fringe_spacing_3D(intensity_plane, grid_size):
    """
    Measure fringe spacing in 3D observation plane.
    Use central Z=0 cross-section.
    """
    if find_peaks is None:
        return np.nan
    
    # Extract Z=0 cross-section
    z_center_idx = intensity_plane.shape[0] // 2
    line = intensity_plane[z_center_idx, :]
    
    # Find peaks
    peaks, _ = find_peaks(line, prominence=line.max() * PEAK_PROMINENCE_THRESHOLD)
    
    if len(peaks) < 3:
        return np.nan
    
    # Calculate spacing in grid units
    y_coords = np.arange(-grid_size, grid_size, 1)
    peak_y_coords = y_coords[peaks]
    
    return np.mean(np.diff(peak_y_coords))

def analytical_fringe_spacing_3D(d, wavelength, L):
    """Same as 2D for small angles: Δy ≈ λL/d"""
    return wavelength * L / d

def generate_validation_table_3D(grid_size, save_path):
    print("\n--- 3D Validation Table ---")
    
    results = []
    test_params = [
        {'d': MAIN_D, 'wavelength': 5.0, 'grid_size': grid_size},
        {'d': MAIN_D, 'wavelength': 7.5, 'grid_size': grid_size},
        {'d': MAIN_D, 'wavelength': 10.0, 'grid_size': grid_size},
        {'d': MAIN_D, 'wavelength': 12.5, 'grid_size': grid_size},
    ]
    
    for params in test_params:
        _, _, intensity_plane = simulate_interference_3D(**params, save_path=None)
        
        measured = measure_fringe_spacing_3D(intensity_plane, grid_size)
        analytical = analytical_fringe_spacing_3D(params['d'], params['wavelength'], L_OBSERVATION)
        
        if not np.isnan(measured):
            error = abs(measured - analytical) / analytical * 100
        else:
            error = np.nan
        
        results.append({
            'd': params['d'],
            'lambda': params['wavelength'],
            'Measured_Spacing': f"{measured:.2f}" if not np.isnan(measured) else "N/A",
            'Analytical_Spacing': f"{analytical:.2f}",
            'Error_Pct': f"{error:.2f}%" if not np.isnan(error) else "N/A"
        })
    
    with open(os.path.join(save_path, '3D_quantitative_validation_table.csv'), 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print("   3D table saved.")
    
    errors = [float(r['Error_Pct'].strip('%')) for r in results if r['Error_Pct'] != "N/A"]
    
    with open(os.path.join(save_path, '3D_validation_summary.txt'), 'w') as f:
        f.write("=== 3D Validation Summary ===\n")
        f.write(f"Analytical Model: 3D Spherical Wave (1/r decay)\n")
        f.write(f"Observation Plane: X = {L_OBSERVATION}\n")
        f.write(f"Total Test Cases: {len(results)}\n")
        
        if errors:
            f.write(f"Mean Error: {np.mean(errors):.2f}%\n")
            f.write(f"Max Error: {np.max(errors):.2f}%\n")
            f.write(f"All errors < 5%: {all(e < 5 for e in errors)}\n")
            
            print(f"\n=== 3D Validation Summary ===")
            print(f"Mean Error: {np.mean(errors):.2f}%")
            print(f"Max Error: {np.max(errors):.2f}%")
            print(f"All < 5%: {all(e < 5 for e in errors)}")
        else:
            f.write("No valid data.\n")
    
    print("   3D summary saved.")

if __name__ == '__main__':
    is_colab, drive_path = setup_colab_environment()

    save_path = None
    if is_colab and drive_path:
        save_path = create_output_directory(drive_path)
    elif not is_colab:
        save_path = create_output_directory('.')

    if save_path:
        save_metadata(save_path, GRID_HALF_SIZE, MAIN_D, MAIN_WAVELENGTH)
        simulate_interference_3D(GRID_HALF_SIZE, MAIN_D, MAIN_WAVELENGTH, save_path=save_path)
        
        if find_peaks is not None:
            create_validation_plot_3D(GRID_HALF_SIZE, MAIN_D, MAIN_WAVELENGTH, save_path)
            generate_validation_table_3D(GRID_HALF_SIZE, save_path)
        else:
            print("\n3D Validation skipped: scipy unavailable.")
    
    print("\n--- 3D SIMULATION FINISHED ---")
    if save_path:
        print(f"Outputs: {save_path}")
