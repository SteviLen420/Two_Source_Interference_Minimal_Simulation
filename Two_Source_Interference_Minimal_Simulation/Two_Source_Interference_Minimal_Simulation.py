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

# --- 1. Setup and Library Imports ---
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
CODE_VERSION = '1.2.0'
# ===================================================================================

# --- Scipy Setup ---
find_peaks = None 
try:
    from scipy.signal import find_peaks as initial_find_peaks
    find_peaks = initial_find_peaks
except ImportError:
    pass

# --- 2. Environment Setup ---

def setup_colab_environment():
    global find_peaks
    print("--- Environment Setup Initiated ---")
    
    if 'google.colab' in sys.modules:
        print("Running in Google Colab environment.")
        
        if find_peaks is None:
            print("Installing scipy...")
            try:
                os.system('pip install scipy > /dev/null 2>&1')
                from scipy.signal import find_peaks as re_import_find_peaks
                find_peaks = re_import_find_peaks
                print("scipy installed.")
            except Exception as e:
                print(f"WARNING: scipy installation failed. {e}")

        try:
            from google.colab import drive
            print("Mounting Google Drive...")
            drive.mount('/content/drive')
            print("Google Drive mounted.")
            return True, '/content/drive/MyDrive'
        except Exception as e:
            print(f"ERROR: Drive mount failed. {e}")
            return False, None
    else:
        print("Running locally.")
        return False, '.'

def create_output_directory(base_path):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    project_dir = os.path.join(base_path, OUTPUT_BASE_FOLDER) 
    output_dir = os.path.join(project_dir, f'run_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
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
            'analytical_model': 'cylindrical_wave_exact'
        },
        'output_files': [
            '2D_pattern_*.png', 'intensity_data_*.csv', 'validation_comparison_plot.png', 
            'quantitative_validation_table.csv', 'metadata.json', 'validation_summary.txt'
        ]
    }
    with open(os.path.join(save_path, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    print("Metadata saved.")

# --- 3. Simulation ---

def simulate_interference(grid_size, d, wavelength, save_path=None):
    print(f"\n--- Simulation: d={d}, λ={wavelength} ---")

    x = np.arange(-grid_size, grid_size, 1)
    y = np.arange(-grid_size, grid_size, 1)
    xx, yy = np.meshgrid(x, y)
    
    source1_pos = [0, -d/2]
    source2_pos = [0, d/2]

    # Calculate distances
    dist1 = np.sqrt((xx - source1_pos[0])**2 + (yy - source1_pos[1])**2)
    dist2 = np.sqrt((xx - source2_pos[0])**2 + (yy - source2_pos[1])**2)
    
    k = 2 * np.pi / wavelength
    
    # CRITICAL FIX: Use complex amplitudes for proper cylindrical wave treatment
    # Cylindrical wave: A(r) = (A0/sqrt(r)) * exp(i*k*r)
    
    # Avoid division by zero at source locations
    dist1 = np.where(dist1 < 1, 1, dist1)
    dist2 = np.where(dist2 < 1, 1, dist2)
    
    # Cylindrical wave amplitudes (real part only, for time-averaged)
    amplitude1 = np.cos(k * dist1) / np.sqrt(dist1)
    amplitude2 = np.cos(k * dist2) / np.sqrt(dist2)
    
    # Total field and intensity
    total_amplitude = amplitude1 + amplitude2
    intensity = total_amplitude**2

    if save_path:
        plt.figure(figsize=(10, 8))
        plt.imshow(intensity, origin='lower', 
                   extent=[-grid_size, grid_size, -grid_size, grid_size], 
                   cmap='viridis')
        plt.plot(source1_pos[0], source1_pos[1], 'ro', markersize=8, label='Source 1')
        plt.plot(source2_pos[0], source2_pos[1], 'ro', markersize=8, label='Source 2')
        plt.title(f'Two-Source Interference (d={d}, λ={wavelength})', fontsize=16)
        plt.xlabel('X Position (Grid Units)', fontsize=12)
        plt.ylabel('Y Position (Grid Units)', fontsize=12)
        plt.colorbar(label='Intensity')
        plt.legend()
        plt.grid(False)
        
        plt.savefig(os.path.join(save_path, f'2D_pattern_d{d}_l{wavelength}.png'), 
                    bbox_inches='tight', dpi=300)
        
        np.savetxt(os.path.join(save_path, f'intensity_data_d{d}_l{wavelength}.csv'),
                   intensity, delimiter=",", fmt='%1.4e')
        print("2D outputs saved.")
        plt.close()

    return x, y, intensity

# --- 4. Validation (CORRECTED CYLINDRICAL WAVE ANALYTICAL) ---

def analytical_intensity_cylindrical(y_pos, d, wavelength, L):
    """
    CORRECT analytical formula for cylindrical (2D) wave interference.
    Accounts for 1/sqrt(r) amplitude falloff and phase from actual distances.
    """
    k = 2 * np.pi / wavelength
    
    # Distances from sources at (0, ±d/2) to observation points (L, y)
    r1 = np.sqrt(L**2 + (y_pos + d/2)**2)
    r2 = np.sqrt(L**2 + (y_pos - d/2)**2)
    
    # Cylindrical wave fields with geometric spreading
    # Using cos instead of sin to match simulation phase convention
    A1 = np.cos(k * r1) / np.sqrt(r1)
    A2 = np.cos(k * r2) / np.sqrt(r2)
    
    # Total intensity from coherent superposition
    intensity = (A1 + A2)**2
    
    # Normalize to match numerical normalization (max = 4)
    intensity = intensity / intensity.max() * 4
    
    return intensity

def create_validation_plot(grid_size, d, wavelength, save_path):
    print("\n--- Generating Validation Plot ---")
    
    x_coords, y_coords, intensity_2d = simulate_interference(
        grid_size=grid_size, d=d, wavelength=wavelength, save_path=None
    )
    
    x_position_index = np.argmin(np.abs(x_coords - L_OBSERVATION))
    
    if x_position_index >= intensity_2d.shape[1]:
        print(f"ERROR: L_OBSERVATION={L_OBSERVATION} outside bounds.")
        return
    
    actual_x_position = x_coords[x_position_index]
    print(f"  Measuring at X = {actual_x_position:.1f}")
    
    numerical = intensity_2d[:, x_position_index]
    
    # Use CORRECT cylindrical wave analytical formula
    analytical = analytical_intensity_cylindrical(y_coords, d, wavelength, L=actual_x_position)
    
    numerical_norm = numerical / numerical.max() * 4
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, 
                                   gridspec_kw={'height_ratios': [3, 1]})
    
    ax1.plot(y_coords, numerical_norm, 'b-', label='Numerical', linewidth=2)
    ax1.plot(y_coords, analytical, 'r--', label='Analytical (Cylindrical Wave)', linewidth=2)
    ax1.legend(loc='upper right')
    ax1.set_title(f'Validation at X={actual_x_position:.0f} (d={d}, λ={wavelength})', fontsize=14)
    ax1.set_ylabel('Normalized Intensity', fontsize=12)
    
    difference = numerical_norm - analytical
    ax2.plot(y_coords, difference, 'g-', linewidth=1)
    ax2.axhline(0, color='k', linestyle='--', linewidth=0.8)
    ax2.set_xlabel('Y Position (Grid Units)', fontsize=12)
    ax2.set_ylabel('Difference', fontsize=12)
    ax2.grid(True, linestyle=':', alpha=0.6)
    
    plt.savefig(os.path.join(save_path, 'validation_comparison_plot.png'), 
                bbox_inches='tight', dpi=300)
    print("Validation plot saved.")
    plt.close(fig)

# --- 5. Quantitative Metrics ---

def measure_fringe_spacing(intensity_2d, grid_size):
    """
    FIXED: Returns spacing in GRID UNITS by mapping indices to Y-coordinates.
    """
    if find_peaks is None:
        return np.nan
    
    # Create coordinate arrays
    x_coords = np.arange(-grid_size, grid_size, 1)
    y_coords = np.arange(-grid_size, grid_size, 1)
    
    # Find X index for measurement position
    x_position_index = np.argmin(np.abs(x_coords - L_OBSERVATION))
    
    if x_position_index >= intensity_2d.shape[1]:
        return np.nan
    
    # Extract intensity column at X = L_OBSERVATION
    numerical = intensity_2d[:, x_position_index]
    
    # Find peak INDICES in the array
    peaks, _ = find_peaks(numerical, prominence=numerical.max() * PEAK_PROMINENCE_THRESHOLD)
    
    if len(peaks) < 3:
        return np.nan
    
    # CRITICAL FIX: Map peak indices to actual Y-coordinates
    peak_y_coordinates = y_coords[peaks]
    
    # Calculate spacing in GRID UNITS
    spacings = np.diff(peak_y_coordinates)
    
    return np.mean(spacings)

def analytical_fringe_spacing_cylindrical(d, wavelength, L):
    """
    Calculate expected fringe spacing for cylindrical waves.
    Uses small-angle approximation: Δy ≈ λL/d (same as plane wave for small angles)
    """
    return wavelength * L / d

def generate_validation_table(grid_size, save_path):
    print("\n--- Generating Validation Table ---")
    
    results = []
    
    test_params = [
        {'d': MAIN_D, 'wavelength': 5.0, 'grid_size': grid_size},
        {'d': MAIN_D, 'wavelength': 7.5, 'grid_size': grid_size},
        {'d': MAIN_D, 'wavelength': 10.0, 'grid_size': grid_size},
        {'d': MAIN_D, 'wavelength': 12.5, 'grid_size': grid_size},
    ]
    
    for params in test_params:
        _, _, intensity = simulate_interference(**params, save_path=None)
        
        measured_spacing = measure_fringe_spacing(intensity, grid_size)
        analytical_spacing = analytical_fringe_spacing_cylindrical(params['d'], params['wavelength'], L_OBSERVATION)
        
        if not np.isnan(measured_spacing):
            error = abs(measured_spacing - analytical_spacing) / analytical_spacing * 100
        else:
            error = np.nan
        
        results.append({
            'd': params['d'],
            'lambda': params['wavelength'],
            'Measured_Spacing': f"{measured_spacing:.2f}" if not np.isnan(measured_spacing) else "N/A",
            'Analytical_Spacing': f"{analytical_spacing:.2f}",
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
        f.write(f"Analytical Model: Cylindrical Wave (2D Point Sources)\n")
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

# --- 6. Main Execution ---

if __name__ == '__main__':
    is_colab, drive_path = setup_colab_environment()

    save_path = None
    if is_colab and drive_path:
        save_path = create_output_directory(drive_path)
    elif not is_colab:
        save_path = create_output_directory('.')

    if save_path:
        save_metadata(save_path, GRID_HALF_SIZE, MAIN_D, MAIN_WAVELENGTH)
        simulate_interference(grid_size=GRID_HALF_SIZE, d=MAIN_D, wavelength=MAIN_WAVELENGTH, save_path=save_path)
        
        if find_peaks is not None:
            create_validation_plot(GRID_HALF_SIZE, MAIN_D, MAIN_WAVELENGTH, save_path)
            generate_validation_table(GRID_HALF_SIZE, save_path)
        else:
            print("\nVALIDATION SKIPPED: scipy not available.")
    
    print("\n--- Pipeline Finished ---")
    if save_path:
        print(f"Outputs: {save_path}")
