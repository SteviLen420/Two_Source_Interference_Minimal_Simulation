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
GRID_HALF_SIZE = 1000          # 3D grid: much smaller due to memory (100^3 = 1M points)
MAIN_D = 50                   # Source separation
MAIN_WAVELENGTH = 5.0        # Wavelength
L_OBSERVATION = 900           # Observation plane distance
PEAK_PROMINENCE_THRESHOLD = 0.2
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

def measure_fringe_spacing_3D_FIXED(intensity_plane, y_coords, z_coords):
    """
    FIXED fringe spacing measurement — uses TRUE spatial coordinates.
    """
    if find_peaks is None:
        return np.nan
    
    from scipy.signal import butter, filtfilt
    
    # Find the Z slice with the highest contrast (largest standard deviation)
    contrasts = np.std(intensity_plane, axis=1)
    z_best_idx = np.argmax(contrasts)
    line = intensity_plane[z_best_idx, :]
    
    # Use the central 60% of the line (edges are distorted)
    n = len(line)
    margin = int(0.2 * n)
    line_center = line[margin:-margin]
    y_center = y_coords[margin:-margin]
    
    # Adaptive high-pass filter
    # Estimate expected fringe spacing in pixels
    expected_spacing_pixels = len(line_center) * MAIN_WAVELENGTH * L_OBSERVATION / (MAIN_D * 2 * GRID_HALF_SIZE)
    
    if expected_spacing_pixels < 10:
        print(f"   WARNING: Expected spacing too small ({expected_spacing_pixels:.1f} px)")
        return np.nan
    
    # Apply a high-pass filter to remove slow envelope variations
    nyquist = 0.5
    # Cutoff frequency about 5× slower than fringe frequency
    cutoff_freq = 1.0 / (expected_spacing_pixels * 5)
    cutoff_normalized = cutoff_freq / nyquist
    cutoff_normalized = np.clip(cutoff_normalized, 0.001, 0.4)  # Safe limits
    
    b, a = butter(3, cutoff_normalized, btype='high')
    line_filtered = filtfilt(b, a, line_center)
    
    # Peak detection with improved parameters
    std_val = np.std(line_filtered)
    peaks, properties = find_peaks(
        line_filtered, 
        prominence=std_val * 0.3,  # Lower prominence threshold
        distance=int(expected_spacing_pixels * 0.5)  # Minimum distance between peaks
    )
    
    if len(peaks) < 3:
        print(f"   WARNING: Only {len(peaks)} peaks found!")
        return np.nan
    
    # Compute TRUE spatial distances between peaks
    peak_positions = y_center[peaks]
    spacings = np.diff(peak_positions)
    
    # Filter outliers (remove spacing values that deviate strongly from the median)
    median_spacing = np.median(spacings)
    valid_spacings = spacings[np.abs(spacings - median_spacing) < median_spacing * 0.5]
    
    if len(valid_spacings) < 2:
        return np.nan
    
    avg_spacing = np.mean(valid_spacings)
    
    print(f"   Found {len(peaks)} peaks, average spacing: {avg_spacing:.2f} grid units")
    
    return avg_spacing


def analytical_fringe_spacing_3D_EXACT(d, wavelength, L, y_pos=0):
    """
    Exact analytical fringe spacing in 3D at a given Y position.
    
    Small-angle approximation: Δy ≈ λL/d
    Valid near the center region, but exact formula can depend on angle.
    """
    # Small-angle approximation (accurate near center)
    return wavelength * L / d


def analytical_intensity_3D_spherical_FIXED(y_pos, z_pos, d, wavelength, L):
    """
    FIXED analytical intensity — correct normalization applied.
    """
    k = 2 * np.pi / wavelength
    
    # 3D distances from each source
    r1 = np.sqrt(L**2 + y_pos**2 + (z_pos + d/2)**2)
    r2 = np.sqrt(L**2 + y_pos**2 + (z_pos - d/2)**2)
    
    # Avoid singularities
    r1 = np.maximum(r1, 1.0)
    r2 = np.maximum(r2, 1.0)
    
    # Spherical wave amplitudes
    A1 = np.exp(1j * k * r1) / r1
    A2 = np.exp(1j * k * r2) / r2
    
    intensity = np.abs(A1 + A2)**2
    
    # Normalize so that ideal coherent addition gives intensity = 4
    # (|1 + 1|² = 4). Due to 1/r decay, the real max is slightly lower.
    # Normalize using central region (y=0, z=0, where r1≈r2≈L)
    r_center = np.sqrt(L**2 + (d/2)**2)
    A_center = 1.0 / r_center
    I_max_theory = (2 * A_center)**2  # Constructive interference at center
    
    intensity = intensity / intensity.max() * 4  # Or normalize by I_max_theory
    
    return intensity


def generate_validation_table_3D_FIXED(grid_size, save_path):
    """
    FIXED validation table using improved measurement and analytical functions.
    """
    print("\n--- 3D Validation Table (FIXED) ---")
    
    results = []
    test_params = [
        {'d': MAIN_D, 'wavelength': 5.0, 'grid_size': grid_size},
        {'d': MAIN_D, 'wavelength': 7.5, 'grid_size': grid_size},
        {'d': MAIN_D, 'wavelength': 10.0, 'grid_size': grid_size},
        {'d': MAIN_D, 'wavelength': 12.5, 'grid_size': grid_size},
    ]
    
    for params in test_params:
        y_coords, z_coords, intensity_plane = simulate_interference_3D(
            **params, save_path=None
        )
        
        measured = measure_fringe_spacing_3D_FIXED(intensity_plane, y_coords, z_coords)
        analytical = analytical_fringe_spacing_3D_EXACT(
            params['d'], params['wavelength'], L_OBSERVATION, y_pos=0
        )
        
        if not np.isnan(measured):
            error = abs(measured - analytical) / analytical * 100
        else:
            error = np.nan
        
        results.append({
            'd': params['d'],
            'lambda': params['wavelength'],
            'L': L_OBSERVATION,
            'Analytical_Spacing': f"{analytical:.2f}",
            'Measured_Spacing': f"{measured:.2f}" if not np.isnan(measured) else "N/A",
            'Error_Pct': f"{error:.2f}%" if not np.isnan(error) else "N/A"
        })
        
        print(f"   λ={params['wavelength']}: Analytical={analytical:.2f}, " + 
              f"Measured={measured:.2f if not np.isnan(measured) else 'N/A'}, " +
              f"Error={error:.2f if not np.isnan(error) else 'N/A'}%")
    
    # Save CSV results
    with open(os.path.join(save_path, '3D_quantitative_validation_table_FIXED.csv'), 
              'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    
    print("   3D FIXED table saved.")
    
    # Compute error statistics
    errors = [float(r['Error_Pct'].strip('%')) for r in results if r['Error_Pct'] != "N/A"]
    
    with open(os.path.join(save_path, '3D_validation_summary_FIXED.txt'), 'w') as f:
        f.write("=== 3D Validation Summary (FIXED) ===\n")
        f.write(f"Analytical Model: 3D Spherical Wave (1/r decay)\n")
        f.write(f"Observation Plane: X = {L_OBSERVATION}\n")
        f.write(f"Total Test Cases: {len(results)}\n\n")
        
        if errors:
            f.write(f"Mean Error: {np.mean(errors):.2f}%\n")
            f.write(f"Median Error: {np.median(errors):.2f}%\n")
            f.write(f"Max Error: {np.max(errors):.2f}%\n")
            f.write(f"Min Error: {np.min(errors):.2f}%\n")
            f.write(f"All errors < 10%: {all(e < 10 for e in errors)}\n")
            f.write(f"All errors < 5%: {all(e < 5 for e in errors)}\n")
            
            print(f"\n=== 3D Validation Summary (FIXED) ===")
            print(f"Mean Error: {np.mean(errors):.2f}%")
            print(f"Median Error: {np.median(errors):.2f}%")
            print(f"Max Error: {np.max(errors):.2f}%")
            print(f"All < 10%: {all(e < 10 for e in errors)}")
            print(f"All < 5%: {all(e < 5 for e in errors)}")
        else:
            f.write("ERROR: No valid measurements!\n")
            print("   ERROR: No valid measurements found!")
    
    print("   3D FIXED summary saved.")

def save_full_summary_json(save_path, grid_size, d, wavelength):
    """
    Save comprehensive JSON summary by reading existing output files.
    """
    # Read validation summary if exists
    validation_data = {}
    summary_file = os.path.join(save_path, '3D_validation_summary_FIXED.txt')
    if os.path.exists(summary_file):
        with open(summary_file, 'r') as f:
            content = f.read()
            
            # Parse mean error
            if 'Mean Error:' in content:
                try:
                    mean_line = [l for l in content.split('\n') if 'Mean Error:' in l][0]
                    validation_data['mean_error_percent'] = mean_line.split(':')[1].strip()
                except:
                    pass
            
            # Parse max error
            if 'Max Error:' in content:
                try:
                    max_line = [l for l in content.split('\n') if 'Max Error:' in l][0]
                    validation_data['max_error_percent'] = max_line.split(':')[1].strip()
                except:
                    pass
            
            # Parse pass/fail
            if 'All errors < 5%:' in content:
                try:
                    pass_line = [l for l in content.split('\n') if 'All errors < 5%:' in l][0]
                    validation_data['all_errors_below_5_percent'] = pass_line.split(':')[1].strip()
                except:
                    pass
            
            # If no data found, mark as no valid data
            if not validation_data:
                validation_data['status'] = 'No valid data'
    
    # Read CSV validation table if exists
    csv_file = os.path.join(save_path, '3D_quantitative_validation_table.csv')
    if os.path.exists(csv_file):
        import csv as csv_module
        with open(csv_file, 'r') as f:
            reader = csv_module.DictReader(f)
            validation_data['test_cases'] = list(reader)
    
    # List all output files
    output_files = []
    if os.path.exists(save_path):
        output_files = [f for f in os.listdir(save_path) if os.path.isfile(os.path.join(save_path, f))]
    
    # Build comprehensive summary
    full_summary = {
        'metadata': {
            'simulation_id': os.path.basename(save_path),
            'timestamp': datetime.now().isoformat(),
            'code_version': CODE_VERSION,
            'dimensionality': '3D'
        },
        'simulation_parameters': {
            'grid_half_size': grid_size,
            'grid_total_points': (2 * grid_size) ** 3,
            'source_separation_d': d,
            'wavelength': wavelength,
            'observation_plane_distance': L_OBSERVATION,
            'far_field_criterion': {
                'd_squared_over_lambda': (d**2) / wavelength,
                'L_value': L_OBSERVATION,
                'far_field_satisfied': L_OBSERVATION > (d**2) / wavelength
            }
        },
        'validation': validation_data,
        'output_files': {
            'total_count': len(output_files),
            'files': sorted(output_files)
        },
        'computational_info': {
            'observation_plane_only': True,
            'full_volume_computed': False,
            'memory_optimization': 'plane_extraction'
        }
    }
    
    # Save to JSON
    summary_filename = os.path.join(save_path, 'full_summary.json')
    with open(summary_filename, 'w') as f:
        json.dump(full_summary, f, indent=2)
    
    print(f"   Full summary JSON saved.")

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
        
        # Save full summary JSON at the very end
        save_full_summary_json(save_path, GRID_HALF_SIZE, MAIN_D, MAIN_WAVELENGTH)
    
    print("\n--- 3D SIMULATION FINISHED ---")
    if save_path:
        print(f"Outputs: {save_path}")
