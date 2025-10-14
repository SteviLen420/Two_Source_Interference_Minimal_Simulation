# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Stefan Len
#
# ===================================================================================
# Two_Source_Interference_Minimal_Simulation.py
# ===================================================================================
# Author: Stefan Len
# Overview:
#   Fully validated computational framework for two-source wave interference.
#   CRITICAL FIX: Uses correct cylindrical wave analytical formula for validation.
# ===================================================================================

# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Stefan Len

import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
import csv
import sys 
import json 

# ===================================================================================
# --- MASTER CONTROL ---
# ===================================================================================
GRID_HALF_SIZE = 600
MAIN_D = 20
MAIN_WAVELENGTH = 10.0
L_OBSERVATION = 400
PEAK_PROMINENCE_THRESHOLD = 0.3
OUTPUT_BASE_FOLDER = 'Interference_Sims'
CODE_VERSION = '1.3.0'
# ===================================================================================

find_peaks = None 
try:
    from scipy.signal import find_peaks as initial_find_peaks
    find_peaks = initial_find_peaks
except ImportError:
    pass

def setup_colab_environment():
    global find_peaks
    print("--- Environment Setup ---")
    
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
            'wavelength': wavelength
        },
        'validation_parameters': {
            'far_field_distance_L': L_OBSERVATION,
            'peak_detection_prominence_threshold': PEAK_PROMINENCE_THRESHOLD,
            'analytical_model': 'cylindrical_wave_complex_phasor'
        }
    }
    with open(os.path.join(save_path, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

def simulate_interference(grid_size, d, wavelength, save_path=None):
    print(f"\n--- Simulation: d={d}, lambda={wavelength} ---")

    x = np.arange(-grid_size, grid_size, 1)
    y = np.arange(-grid_size, grid_size, 1)
    xx, yy = np.meshgrid(x, y)
    
    source1_pos = [0, -d/2]
    source2_pos = [0, d/2]

    dist1 = np.sqrt((xx - source1_pos[0])**2 + (yy - source1_pos[1])**2)
    dist2 = np.sqrt((xx - source2_pos[0])**2 + (yy - source2_pos[1])**2)
    
    # Avoid singularity
    dist1 = np.where(dist1 < 1, 1, dist1)
    dist2 = np.where(dist2 < 1, 1, dist2)
    
    k = 2 * np.pi / wavelength
    
    # FIXED: Complex phasor for cylindrical waves
    amplitude1 = np.exp(1j * k * dist1) / np.sqrt(dist1)
    amplitude2 = np.exp(1j * k * dist2) / np.sqrt(dist2)
    
    # Time-averaged intensity
    total_field = amplitude1 + amplitude2
    intensity = np.abs(total_field)**2

    if save_path:
        plt.figure(figsize=(10, 8))
        plt.imshow(intensity, origin='lower', 
                   extent=[-grid_size, grid_size, -grid_size, grid_size], 
                   cmap='viridis')
        plt.plot(source1_pos[0], source1_pos[1], 'ro', markersize=8, label='Source 1')
        plt.plot(source2_pos[0], source2_pos[1], 'ro', markersize=8, label='Source 2')
        plt.title(f'Two-Source Interference (d={d}, λ={wavelength})', fontsize=16)
        plt.xlabel('X (Grid Units)', fontsize=12)
        plt.ylabel('Y (Grid Units)', fontsize=12)
        plt.colorbar(label='Intensity')
        plt.legend()
        plt.grid(False)
        
        plt.savefig(os.path.join(save_path, f'2D_pattern_d{d}_l{wavelength}.png'), 
                    bbox_inches='tight', dpi=300)
        np.savetxt(os.path.join(save_path, f'intensity_data_d{d}_l{wavelength}.csv'),
                   intensity, delimiter=",", fmt='%1.4e')
        print("2D saved.")
        plt.close()

    return x, y, intensity

def analytical_intensity_cylindrical(y_pos, d, wavelength, L):
    """FIXED: Complex phasor analytical formula"""
    k = 2 * np.pi / wavelength
    
    r1 = np.sqrt(L**2 + (y_pos + d/2)**2)
    r2 = np.sqrt(L**2 + (y_pos - d/2)**2)
    
    # Complex cylindrical waves
    A1 = np.exp(1j * k * r1) / np.sqrt(r1)
    A2 = np.exp(1j * k * r2) / np.sqrt(r2)
    
    intensity = np.abs(A1 + A2)**2
    
    # Normalize to max=4
    intensity = intensity / intensity.max() * 4
    
    return intensity

def create_validation_plot(grid_size, d, wavelength, save_path):
    print("\n--- Validation Plot ---")
    
    x_coords, y_coords, intensity_2d = simulate_interference(
        grid_size=grid_size, d=d, wavelength=wavelength, save_path=None
    )
    
    x_idx = np.argmin(np.abs(x_coords - L_OBSERVATION))
    actual_x = x_coords[x_idx]
    print(f"  X = {actual_x:.1f}")
    
    numerical = intensity_2d[:, x_idx]
    analytical = analytical_intensity_cylindrical(y_coords, d, wavelength, L=actual_x)
    
    numerical_norm = numerical / numerical.max() * 4
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, 
                                   gridspec_kw={'height_ratios': [3, 1]})
    
    ax1.plot(y_coords, numerical_norm, 'b-', label='Numerical', linewidth=2)
    ax1.plot(y_coords, analytical, 'r--', label='Analytical', linewidth=2)
    ax1.legend()
    ax1.set_title(f'Validation X={actual_x:.0f} (d={d}, λ={wavelength})', fontsize=14)
    ax1.set_ylabel('Normalized Intensity', fontsize=12)
    
    difference = numerical_norm - analytical
    ax2.plot(y_coords, difference, 'g-', linewidth=1)
    ax2.axhline(0, color='k', linestyle='--', linewidth=0.8)
    ax2.set_xlabel('Y Position', fontsize=12)
    ax2.set_ylabel('Difference', fontsize=12)
    ax2.grid(True, linestyle=':', alpha=0.6)
    
    plt.savefig(os.path.join(save_path, 'validation_comparison_plot.png'), 
                bbox_inches='tight', dpi=300)
    print("Validation plot saved.")
    plt.close(fig)

def measure_fringe_spacing(intensity_2d, grid_size):
    if find_peaks is None:
        return np.nan
    
    x_coords = np.arange(-grid_size, grid_size, 1)
    y_coords = np.arange(-grid_size, grid_size, 1)
    
    x_idx = np.argmin(np.abs(x_coords - L_OBSERVATION))
    if x_idx >= intensity_2d.shape[1]:
        return np.nan
    
    numerical = intensity_2d[:, x_idx]
    peaks, _ = find_peaks(numerical, prominence=numerical.max() * PEAK_PROMINENCE_THRESHOLD)
    
    if len(peaks) < 3:
        return np.nan
    
    # Convert indices to Y-coordinates
    peak_y_coords = y_coords[peaks]
    return np.mean(np.diff(peak_y_coords))

def analytical_fringe_spacing_cylindrical(d, wavelength, L):
    return wavelength * L / d

def generate_validation_table(grid_size, save_path):
    print("\n--- Validation Table ---")
    
    results = []
    test_params = [
        {'d': MAIN_D, 'wavelength': 5.0, 'grid_size': grid_size},
        {'d': MAIN_D, 'wavelength': 7.5, 'grid_size': grid_size},
        {'d': MAIN_D, 'wavelength': 10.0, 'grid_size': grid_size},
        {'d': MAIN_D, 'wavelength': 12.5, 'grid_size': grid_size},
    ]
    
    for params in test_params:
        _, _, intensity = simulate_interference(**params, save_path=None)
        
        measured = measure_fringe_spacing(intensity, grid_size)
        analytical = analytical_fringe_spacing_cylindrical(params['d'], params['wavelength'], L_OBSERVATION)
        
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
    
    with open(os.path.join(save_path, 'quantitative_validation_table.csv'), 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print("Table saved.")
    
    errors = [float(r['Error_Pct'].strip('%')) for r in results if r['Error_Pct'] != "N/A"]
    
    with open(os.path.join(save_path, 'validation_summary.txt'), 'w') as f:
        f.write("=== Validation Summary ===\n")
        f.write(f"Analytical Model: Complex Cylindrical Wave\n")
        f.write(f"Measurement Position: X = {L_OBSERVATION}\n")
        f.write(f"Total Test Cases: {len(results)}\n")
        
        if errors:
            f.write(f"Mean Error: {np.mean(errors):.2f}%\n")
            f.write(f"Max Error: {np.max(errors):.2f}%\n")
            f.write(f"All errors < 5%: {all(e < 5 for e in errors)}\n")
            
            print(f"\n=== Validation Summary ===")
            print(f"Mean Error: {np.mean(errors):.2f}%")
            print(f"Max Error: {np.max(errors):.2f}%")
            print(f"All < 5%: {all(e < 5 for e in errors)}")
        else:
            f.write("No valid data.\n")
    
    print("Summary saved.")

if __name__ == '__main__':
    is_colab, drive_path = setup_colab_environment()

    save_path = None
    if is_colab and drive_path:
        save_path = create_output_directory(drive_path)
    elif not is_colab:
        save_path = create_output_directory('.')

    if save_path:
        save_metadata(save_path, GRID_HALF_SIZE, MAIN_D, MAIN_WAVELENGTH)
        simulate_interference(GRID_HALF_SIZE, MAIN_D, MAIN_WAVELENGTH, save_path=save_path)
        
        if find_peaks is not None:
            create_validation_plot(GRID_HALF_SIZE, MAIN_D, MAIN_WAVELENGTH, save_path)
            generate_validation_table(GRID_HALF_SIZE, save_path)
        else:
            print("\nValidation skipped: scipy unavailable.")
    
    print("\n--- FINISHED ---")
    if save_path:
        print(f"Outputs: {save_path}")
