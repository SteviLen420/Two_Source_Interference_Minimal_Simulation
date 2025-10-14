# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Stefan Len
#
# ===================================================================================
# Two_Source_Interference_Minimal_Simulation.py (Version 1.1.0)
# ===================================================================================
# Author: Stefan Len
# Overview:
#   A fully validated and reproducible computational framework for two-source 
#   wave interference simulation, saving all results to Google Drive.
#   
#   1. Auto-mounts Google Drive and installs 'scipy'.
#   2. Runs main 2D simulation.
#   3. Generates a validation plot (Numerical vs. Analytical cross-section).
#   4. Calculates quantitative metrics (Fringe Spacing Error Table).
#   5. Saves all data, plots, and metadata to a timestamped folder in Drive.
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

# Global default parameters
GRID_HALF_SIZE = 300

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
    project_dir = os.path.join(base_path, 'Interference_Sims') 
    output_dir = os.path.join(project_dir, f'run_{timestamp}')
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory created: {output_dir}")
    return output_dir

def save_metadata(save_path, grid_size, d, wavelength):
    """Save simulation parameters and run details to a JSON file."""
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'code_version': '1.1.0', 
        'simulation_parameters': {
            'grid_half_size': grid_size,
            'source_separation_d': d,
            'wavelength': wavelength
        },
        'validation_parameters': {
            'far_field_distance_L': grid_size, # L is set equal to the grid half-size
            'peak_detection_prominence_threshold': 0.3
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

# --- 4. Validation Functions (Analytical Comparison) ---

def analytical_intensity(y_pos, d, wavelength, L):
    """
    Analytical intensity for the far-field two-slit approximation.
    $$I(y) = 4I_0 \cos^2\left(\frac{\pi d y}{\lambda L}\right)$$
    """
    k = 2 * np.pi / wavelength
    phase_diff = k * d * y_pos / L
    # Normalize to max 4 (assuming I0=1)
    intensity = 4 * np.cos(phase_diff / 2)**2 
    return intensity

def create_validation_plot(grid_size, d, wavelength, save_path):
    """
    Generates a 2-panel plot comparing the numerical central cross-section to the analytical solution.
    """
    print("\n--- Generating Validation Plot: Numerical vs. Analytical ---")
    
    x_coords, _, intensity_2d = simulate_interference(grid_size=grid_size, d=d, wavelength=wavelength, save_path=None)
    
    center_idx = intensity_2d.shape[0] // 2
    numerical = intensity_2d[center_idx, :]
    
    analytical = analytical_intensity(x_coords, d, wavelength, L=grid_size)
    
    # Normalize numerical to match analytical max (I_max = 4 for unit amplitude waves)
    numerical_norm = numerical / numerical.max() * 4
    
    # Plot Comparison
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, 
                                   gridspec_kw={'height_ratios': [3, 1]})
    
    # FIX 4: Use r-string for $\cos^2$ in label to avoid SyntaxWarning
    ax1.plot(x_coords, numerical_norm, 'b-', label='Numerical Simulation (Normalized)', linewidth=2)
    ax1.plot(x_coords, analytical, 'r--', label=r'Analytical Far-Field ($I \propto \cos^2$)', linewidth=2) 
    ax1.legend(loc='upper right')
    # FIX 5: Use r-string for $\lambda$ in plt.title to avoid SyntaxWarning
    ax1.set_title(r'Numerical vs. Analytical Intensity Cross-Section (d={d}, $\lambda$={wavelength})'.format(d=d, wavelength=wavelength), fontsize=14) 
    ax1.set_ylabel('Normalized Intensity', fontsize=12)
    
    difference = numerical_norm - analytical
    ax2.plot(x_coords, difference, 'g-', linewidth=1)
    ax2.axhline(0, color='k', linestyle='--', linewidth=0.8)
    ax2.set_xlabel('Position along Measurement Line (Grid Units)', fontsize=12)
    ax2.set_ylabel('Difference (Num - Anal)', fontsize=12)
    ax2.grid(True, linestyle=':', alpha=0.6)
    
    plot_filename = os.path.join(save_path, 'validation_comparison_plot.png')
    plt.savefig(plot_filename, bbox_inches='tight', dpi=300)
    print(f"Validation comparison plot saved: {plot_filename}")
    plt.close(fig)

# --- 5. Quantitative Metrics (Table Generation) ---

def measure_fringe_spacing(intensity_2d, prominence_threshold=0.3):
    """
    Measures the average fringe spacing (distance between two adjacent maxima) 
    from the central cross-section using peak detection (SciPy).
    """
    if find_peaks is None:
        return np.nan
        
    center_idx = intensity_2d.shape[0] // 2
    numerical = intensity_2d[center_idx, :]
    
    peaks, _ = find_peaks(numerical, prominence=numerical.max() * prominence_threshold)
    
    if len(peaks) < 3:
        return np.nan
    
    spacings = np.diff(peaks)
    
    return np.mean(spacings)

def generate_validation_table(grid_size, save_path):
    """
    Runs simulations with different parameters to quantitatively compare 
    measured fringe spacing to the analytical prediction (λL/d).
    """
    print("\n--- Generating Quantitative Validation Table Data ---")
    
    results = []
    
    test_params = [
        {'d': 40, 'wavelength': 5.0, 'grid_size': grid_size},
        {'d': 60, 'wavelength': 5.0, 'grid_size': grid_size},
        {'d': 40, 'wavelength': 10.0, 'grid_size': grid_size},
        {'d': 80, 'wavelength': 10.0, 'grid_size': grid_size},
    ]
    
    for params in test_params:
        _, _, intensity = simulate_interference(**params, save_path=None)
        
        measured_spacing = measure_fringe_spacing(intensity)
        
        # Analytical prediction (Δy = λ * L / d). L = grid_size.
        analytical_spacing = params['wavelength'] * params['grid_size'] / params['d']
        
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
        
    # Save the results as a CSV table
    table_filename = os.path.join(save_path, 'quantitative_validation_table.csv')
    
    if results:
        with open(table_filename, 'w', newline='') as csvfile:
            fieldnames = list(results[0].keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"Quantitative validation table saved: {table_filename}")
        
    # Summary statistics
    summary_filename = os.path.join(save_path, 'validation_summary.txt')
    errors = [float(r['Error_Pct'].strip('%')) for r in results if r['Error_Pct'] != "N/A"]
    
    with open(summary_filename, 'w') as f:
        f.write("=== Validation Summary ===\n")
        f.write(f"Total Test Cases: {len(results)}\n")
        
        if errors:
            mean_error = np.mean(errors)
            max_error = np.max(errors)
            all_errors_low = all(e < 5 for e in errors)
            f.write(f"Mean Error in Fringe Spacing: {mean_error:.2f}%\n")
            f.write(f"Maximum Observed Error: {max_error:.2f}%\n")
            f.write(f"All errors < 5% threshold: {all_errors_low}\n")
            
            print("\n=== Validation Summary ===")
            print(f"Mean Error: {mean_error:.2f}%")
            print(f"Max Error: {max_error:.2f}%")
            print(f"All errors < 5%: {all_errors_low}")
        else:
            f.write("No valid error data available (scipy/peak detection failed).\n")
            print("\nValidation Summary: No valid error data available.")
        
    print(f"Validation summary statistics saved: {summary_filename}")


# --- 6. Main Execution Block ---

if __name__ == '__main__':
    
    # Core simulation parameters (Customization Point)
    MAIN_D = 60
    MAIN_WAVELENGTH = 10.0
    
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
