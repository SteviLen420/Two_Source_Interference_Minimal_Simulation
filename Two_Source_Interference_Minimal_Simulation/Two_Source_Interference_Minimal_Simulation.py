# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Stefan Len
#
# ===================================================================================
# Two_Source_Interference_Minimal_Simulation.py 
# ===================================================================================
# Author: Stefan Len
# Overview:
#   A fully validated and reproducible computational framework for two-source 
#   wave interference simulation, saving all results to Google Drive.
#   
#   1. Auto-mounts Google Drive and installs 'scipy'.
#   2. Runs main 2D simulation.
#   3. Generates a validation plot (Numerical vs. Analytical cross-section).
#   4. Calculates quantitative metrics (Fringe Spacing Error Table).
#   5. Saves all data, plots, and metadata to a timestamped folder in Drive.
# ===================================================================================

# --- 1. Setup and Library Imports ---
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
import csv
import sys 
import json 
# import shutil # Not strictly needed

# ===================================================================================
# --- MASTER CONTROL ---
# ===================================================================================
# --- Core Simulation Parameters ---
GRID_HALF_SIZE = 500          # Half-width of the simulation grid (e.g., 500 creates a 1000x1000 grid)
MAIN_D = 20                   # Main source separation for the primary 2D plot
MAIN_WAVELENGTH = 10.0        # Main wavelength for the primary 2D plot

# --- Validation Parameters ---
# Distance L at which the vertical cross-section is taken for validation.
L_OBSERVATION = int(GRID_HALF_SIZE * 0.67) 
# Prominence threshold for detecting intensity peaks (as a fraction of max intensity).
PEAK_PROMINENCE_THRESHOLD = 0.3

# --- Output and Metadata Parameters ---
OUTPUT_BASE_FOLDER = 'Interference_Sims' # Name of the main project folder in Drive/local
CODE_VERSION = '1.1.0'
# ===================================================================================


# --- Scipy Setup for Validation ---
# FIX 1: Initialize find_peaks at the top level to avoid global/local conflicts.
find_peaks = None 
try:
    from scipy.signal import find_peaks as initial_find_peaks
    find_peaks = initial_find_peaks
except ImportError:
    pass # If scipy is not available yet, it will be installed in Colab setup.

# --- 2. Colab Environment Check and Setup Functions ---

def setup_colab_environment():
    """
    Checks if running in Colab, mounts Google Drive, and installs 'scipy'.
    """
    global find_peaks # Declaration is needed here if we reassign the global variable
    print("--- Environment Setup Initiated ---")
    
    if 'google.colab' in sys.modules:
        print("Running in Google Colab environment.")
        
        if find_peaks is None:
            print("Installing/ensuring 'scipy' is available...")
            try:
                os.system('pip install scipy > /dev/null 2>&1')
                # Try re-importing after installation
                from scipy.signal import find_peaks as re_import_find_peaks
                find_peaks = re_import_find_peaks
                print("'scipy' installed/verified. Validation enabled.")
            except Exception as e:
                print(f"WARNING: Could not install scipy. Validation functions will be skipped. Details: {e}")

        # Mount Google Drive
        try:
            from google.colab import drive
            print("Attempting to mount Google Drive...")
            drive.mount('/content/drive')
            print("Google Drive mounted successfully.")
            return True, '/content/drive/MyDrive'
        except Exception as e:
            print(f"ERROR: Could not mount Google Drive. Saving will be skipped. Details: {e}")
            return False, None
    else:
        print("Not running in Google Colab. Outputs will be saved to the local directory.")
        return False, '.'

def create_output_directory(base_path):
    """Creates a unique, timestamped output directory."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    project_dir = os.path.join(base_path, OUTPUT_BASE_FOLDER) 
    output_dir = os.path.join(project_dir, f'run_{timestamp}')
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory created: {output_dir}")
    return output_dir

def save_metadata(save_path, grid_size, d, wavelength):
    """Save simulation parameters and run details to a JSON file."""
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
            'peak_detection_prominence_threshold': PEAK_PROMINENCE_THRESHOLD
        },
        'output_files': [
            '2D_pattern_*.png', 'intensity_data_*.csv', 'validation_comparison_plot.png', 
            'quantitative_validation_table.csv', 'metadata.json', 'validation_summary.txt'
        ]
    }
    metadata_filename = os.path.join(save_path, 'metadata.json')
    with open(metadata_filename, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved: {metadata_filename}")

# --- 3. Interference Simulation Function ---

def simulate_interference(grid_size, d, wavelength, save_path=None):
    """
    Simulates the interference pattern of two point sources and returns the results.
    """
    # FIX 2: Use r-string for $\lambda$ in print to avoid SyntaxWarning
    print(f"\n--- Running Simulation: d={d}, $\lambda$={wavelength} ---") 

    # 3.1. Define Grid and Sources
    x = np.arange(-grid_size, grid_size, 1)
    y = np.arange(-grid_size, grid_size, 1)
    xx, yy = np.meshgrid(x, y)
    source1_pos = [0, -d/2]
    source2_pos = [0, d/2]

    # 3.2. Calculate Wave Propagation
    dist1 = np.sqrt((xx - source1_pos[0])**2 + (yy - source1_pos[1])**2)
    dist2 = np.sqrt((xx - source2_pos[0])**2 + (yy - source2_pos[1])**2) 
    
    k = 2 * np.pi / wavelength
    wave1 = np.sin(k * dist1)
    wave2 = np.sin(k * dist2)

    # 3.3. Calculate Interference (Superposition)
    total_wave = wave1 + wave2
    intensity = total_wave**2

    # 3.4. Visualization and Saving
    plt.figure(figsize=(10, 8))
    plt.imshow(intensity, 
               origin='lower', 
               extent=[-grid_size, grid_size, -grid_size, grid_size], 
               cmap='viridis')
    plt.plot(source1_pos[0], source1_pos[1], 'ro', markersize=8, label='Source 1')
    plt.plot(source2_pos[0], source2_pos[1], 'ro', markersize=8, label='Source 2')
    # FIX 3: Use r-string for $\lambda$ in plt.title to avoid SyntaxWarning
    plt.title(r'Two-Source Interference Pattern (d={d}, $\lambda$={wavelength})'.format(d=d, wavelength=wavelength), fontsize=16)
    plt.xlabel('X Position (Grid Units)', fontsize=12)
    plt.ylabel('Y Position (Grid Units)', fontsize=12)
    plt.colorbar(label='Intensity')
    plt.legend()
    plt.grid(False) 
    
    if save_path:
        plot_filename = os.path.join(save_path, f'2D_pattern_d{d}_l{wavelength}.png')
        plt.savefig(plot_filename, bbox_inches='tight', dpi=300)
        
        data_filename = os.path.join(save_path, f'intensity_data_d{d}_l{wavelength}.csv')
        np.savetxt(data_filename, intensity, delimiter=",", fmt='%1.4e', 
                   header=f'Two-Source Interference Intensity Data (d={d}, lambda={wavelength})')
        print(f"Main 2D output files saved.")
        
    plt.close()

    return x, y, intensity

# --- 4. Validation Functions (Analytical Comparison) - FIXED VERSION ---

def analytical_intensity(y_pos, d, wavelength, L):
    """
    Analytical intensity for the far-field two-slit approximation.
    I(y) = 4 * I0 * cos²( (π * d * y) / (λ * L) ).
    
    Args:
        y_pos: Y-coordinates (perpendicular to source line)
        d: Source separation
        wavelength: Wave wavelength
        L: Perpendicular distance from source line (X-coordinate of measurement)
    """
    k = 2 * np.pi / wavelength
    phase_diff = k * d * y_pos / L
    intensity = 4 * np.cos(phase_diff / 2)**2
    return intensity

def create_validation_plot(grid_size, d, wavelength, save_path):
    """
    Generates a 2-panel plot comparing the far-field cross-section.
    FIXED: Correct array indexing using np.argmin.
    """
    print("\n--- Generating Validation Plot: Numerical vs. Analytical ---")
    
    x_coords, y_coords, intensity_2d = simulate_interference(
        grid_size=grid_size, d=d, wavelength=wavelength, save_path=None
    )
    
    # CRITICAL FIX: Find correct array index for X = L_OBSERVATION
    x_position_index = np.argmin(np.abs(x_coords - L_OBSERVATION))
    
    if x_position_index >= intensity_2d.shape[1]:
        print(f"ERROR: L_OBSERVATION={L_OBSERVATION} is outside grid bounds.")
        return
    
    actual_x_position = x_coords[x_position_index]
    print(f"  Measuring at X = {actual_x_position:.1f} (target: {L_OBSERVATION})")
    
    # Extract vertical line (all Y at fixed X)
    numerical = intensity_2d[:, x_position_index]
    
    # Calculate analytical
    analytical = analytical_intensity(y_coords, d, wavelength, L=actual_x_position)
    
    # Normalize
    numerical_norm = numerical / numerical.max() * 4
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, 
                                      gridspec_kw={'height_ratios': [3, 1]})
    
    ax1.plot(y_coords, numerical_norm, 'b-', label='Numerical', linewidth=2)
    ax1.plot(y_coords, analytical, 'r--', label='Analytical Far-Field', linewidth=2)
    ax1.legend(loc='upper right')
    ax1.set_title(f'Cross-Section at X={actual_x_position:.0f} (d={d}, λ={wavelength})', fontsize=14)
    ax1.set_ylabel('Normalized Intensity', fontsize=12)
    
    difference = numerical_norm - analytical
    ax2.plot(y_coords, difference, 'g-', linewidth=1)
    ax2.axhline(0, color='k', linestyle='--', linewidth=0.8)
    ax2.set_xlabel('Y Position (Grid Units)', fontsize=12)
    ax2.set_ylabel('Difference', fontsize=12)
    ax2.grid(True, linestyle=':', alpha=0.6)
    
    plot_filename = os.path.join(save_path, 'validation_comparison_plot.png')
    plt.savefig(plot_filename, bbox_inches='tight', dpi=300)
    print(f"Validation plot saved: {plot_filename}")
    plt.close(fig)

# --- 5. Quantitative Metrics - FIXED VERSION ---

def measure_fringe_spacing(intensity_2d, grid_size):
    """
    FIXED: Measure fringe spacing at X = L_OBSERVATION with correct indexing.
    """
    if find_peaks is None:
        return np.nan
    
    # Recreate coordinate array
    x_coords = np.arange(-grid_size, grid_size, 1)
    
    # Find correct index
    x_position_index = np.argmin(np.abs(x_coords - L_OBSERVATION))
    
    if x_position_index >= intensity_2d.shape[1]:
        return np.nan
    
    numerical = intensity_2d[:, x_position_index]
    
    peaks, _ = find_peaks(numerical, prominence=numerical.max() * PEAK_PROMINENCE_THRESHOLD)
    
    if len(peaks) < 3:
        return np.nan
    
    spacings = np.diff(peaks)
    return np.mean(spacings)

def generate_validation_table(grid_size, save_path):
    """FIXED: Use correct L for analytical prediction"""
    print("\n--- Generating Quantitative Validation Table ---")
    
    results = []
    
    test_params = [
        {'d': 40, 'wavelength': 5.0, 'grid_size': grid_size},
        {'d': 60, 'wavelength': 5.0, 'grid_size': grid_size},
        {'d': 40, 'wavelength': 10.0, 'grid_size': grid_size},
        {'d': 80, 'wavelength': 10.0, 'grid_size': grid_size},
    ]
    
    for params in test_params:
        _, _, intensity = simulate_interference(**params, save_path=None)
        
        measured_spacing = measure_fringe_spacing(intensity, grid_size)
        
        # FIXED: Use L_OBSERVATION from MASTER CONTROL
        analytical_spacing = params['wavelength'] * L_OBSERVATION / params['d']
        
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
    
    # Save table
    table_filename = os.path.join(save_path, 'quantitative_validation_table.csv')
    with open(table_filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"Table saved: {table_filename}")
    
    # Save summary
    summary_filename = os.path.join(save_path, 'validation_summary.txt')
    errors = [float(r['Error_Pct'].strip('%')) for r in results if r['Error_Pct'] != "N/A"]
    
    with open(summary_filename, 'w') as f:
        f.write("=== Validation Summary ===\n")
        f.write(f"Measurement Position: X = {L_OBSERVATION} (perpendicular to sources)\n")
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
    
    print(f"Summary saved: {summary_filename}")


# --- 6. Main Execution Block ---

if __name__ == '__main__':
    
    # Note: Core parameters are now set in the MASTER CONTROL section at the top.
    
    # 6.1. Setup Environment and Determine Save Path
    is_colab, drive_path = setup_colab_environment()

    save_path = None
    if is_colab and drive_path:
        save_path = create_output_directory(drive_path)
    elif not is_colab:
        save_path = create_output_directory('.')

    if save_path:
        # 6.2. Save initial metadata
        save_metadata(save_path, GRID_HALF_SIZE, MAIN_D, MAIN_WAVELENGTH)
        
        # 6.3. Run the main simulation (2D plot and data)
        simulate_interference(grid_size=GRID_HALF_SIZE, d=MAIN_D, wavelength=MAIN_WAVELENGTH, save_path=save_path)
        
        # 6.4. Perform Critical Validation and Quantitative Metrics
        if find_peaks is not None:
            create_validation_plot(GRID_HALF_SIZE, MAIN_D, MAIN_WAVELENGTH, save_path)
            generate_validation_table(GRID_HALF_SIZE, save_path)
        else:
            print("\nCRITICAL VALIDATION SKIPPED: 'scipy.signal.find_peaks' is not available. Please ensure scipy is installed.")
    
    print("\n--- Simulation and Validation Pipeline Finished ---")
    if save_path:
        print(f"All outputs are located in: {save_path}")
