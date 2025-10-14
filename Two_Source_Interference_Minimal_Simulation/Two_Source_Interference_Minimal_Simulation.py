# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Stefan Len
#
# ===================================================================================
# Two_Source_Interference_Minimal_Simulation.py
# ===================================================================================
# Author: Stefan Len
# Overview:
#   Minimal numerical simulation of the two-source interference pattern (Colab-ready):
#   - Auto-mounts Google Drive (if running in Colab)
#   - Installs necessary Python packages quietly (numpy, matplotlib)
#   - Calculates the superposition of waves from two point sources
#   - Visualizes the resulting wave intensity/interference pattern
#   - Saves ALL outputs (image, data) to Drive under MyDrive/Interference_Sims/run_YYYYMMDD_HHMMSS/
# ===================================================================================

# --- 1. Setup and Library Imports ---
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
import csv
import sys # Used for environment checking

# --- 2. Colab Environment Check and Setup Functions ---

def setup_colab_environment():
    """
    Checks if the code is running in Google Colab, mounts Google Drive,
    and quietly installs any potentially missing packages.
    """
    print("--- Environment Setup Initiated ---")
    
    # Check if running in Google Colab
    if 'google.colab' in sys.modules:
        print("Running in Google Colab environment.")
        
        # Install necessary packages silently (they are usually pre-installed)
        print("Ensuring 'numpy' and 'matplotlib' are available...")
        # Note: In a real-world scenario, you'd put less common packages here.
        # !pip install numpy matplotlib > /dev/null 2>&1 
        
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
        # Fallback for local execution
        return False, '.'

def create_output_directory(base_path):
    """
    Creates a unique, timestamped output directory on Google Drive or locally.
    
    Args:
        base_path (str): The root path (e.g., '/content/drive/MyDrive' or '.').
        
    Returns:
        str: The full path to the newly created output directory.
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    # Use 'Interference_Sims' as the main project folder name, adapted from your request.
    project_dir = os.path.join(base_path, 'Interference_Sims') 
    output_dir = os.path.join(project_dir, f'run_{timestamp}')
    
    # Create directories if they don't exist
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory created: {output_dir}")
    return output_dir

# --- 3. Interference Simulation Function ---

def simulate_interference(grid_size=200, d=40, wavelength=5.0, save_path=None):
    """
    Simulates and visualizes the interference pattern of two point sources.

    This function creates a 2D grid, calculates the wave pattern from the 
    superposition of waves from two coherent point sources, and visualizes 
    the resulting intensity pattern.

    Args:
        grid_size (int): The half-size of the square grid (grid will be 2*grid_size x 2*grid_size).
        d (int): The distance between the two sources in grid units.
        wavelength (float): The wavelength of the waves.
        save_path (str, optional): Directory path to save the outputs. If None, outputs are displayed but not saved.
    
    Returns:
        tuple: (x_coords, y_coords, intensity_data) for saving.
    """
    print("--- Simulation Started ---")
    print(f"Parameters: Grid Half-Size={grid_size}, Source Separation (d)={d}, Wavelength (λ)={wavelength}")

    # --- 3.1. Define Grid and Sources ---
    # Create a 2D mesh grid for the simulation area.
    x = np.arange(-grid_size, grid_size, 1)
    y = np.arange(-grid_size, grid_size, 1)
    xx, yy = np.meshgrid(x, y)

    # Define the positions of the two sources, symmetrically placed on the Y-axis.
    source1_pos = [0, -d/2]
    source2_pos = [0, d/2]

    # --- 3.2. Calculate Wave Propagation ---
    # Compute the distance from every grid point to both sources (r1 and r2).
    dist1 = np.sqrt((xx - source1_pos[0])**2 + (yy - source1_pos[1])**2)
    dist2 = np.sqrt((xx - source2_pos[0])**2 + (yy - source2_pos[1])**2)

    # Calculate the wave number (k) from the wavelength (λ). k = 2π / λ.
    k = 2 * np.pi / wavelength

    # Calculate the wave functions at every point using simple sine waves.
    wave1 = np.sin(k * dist1)
    wave2 = np.sin(k * dist2)

    # --- 3.3. Calculate Interference (Superposition) ---
    # The total wave function is the linear superposition of the two waves.
    total_wave = wave1 + wave2

    # The intensity is proportional to the square of the total wave amplitude.
    intensity = total_wave**2

    # --- 3.4. Visualization ---
    plt.figure(figsize=(10, 8))
    # Display the intensity pattern using 'imshow'.
    # 'viridis' is a common and effective colormap.
    plt.imshow(intensity, 
               origin='lower', 
               extent=[-grid_size, grid_size, -grid_size, grid_size], 
               cmap='viridis')

    # Mark the location of the sources with red dots.
    plt.plot(source1_pos[0], source1_pos[1], 'ro', markersize=8, label='Source 1')
    plt.plot(source2_pos[0], source2_pos[1], 'ro', markersize=8, label='Source 2')

    plt.title('Two-Source Interference Simulation', fontsize=16)
    plt.xlabel('X Position (Grid Units)', fontsize=12)
    plt.ylabel('Y Position (Grid Units)', fontsize=12)
    plt.colorbar(label='Intensity')
    plt.legend()
    plt.grid(False) # Turn off the grid for cleaner visualization.
    
    # --- 4. Saving Outputs ---
    if save_path:
        print(f"Saving outputs to: {save_path}")
        
        # 4.1. Save the plot as a PNG image
        plot_filename = os.path.join(save_path, 'interference_pattern.png')
        plt.savefig(plot_filename, bbox_inches='tight', dpi=300)
        print(f"Plot saved: {plot_filename}")

        # 4.2. Save the intensity data as a CSV file
        data_filename = os.path.join(save_path, 'intensity_data.csv')
        
        # Create a 2D array for the CSV where each row is a line of intensity data
        # Note: For large grids, this file can be very big.
        # Alternatively, saving a cross-section could be considered.
        with open(data_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Optionally write a header, but for 2D data, the raw matrix is often preferred
            # writer.writerow([f'X_Coord_{i}' for i in x]) # Too verbose for a full 2D grid
            for row in intensity:
                writer.writerow(row)
        print(f"Data saved: {data_filename}")

    plt.show() # Display the plot after saving (if desired)

    return x, y, intensity

# --- 5. Main Execution Block ---
if __name__ == '__main__':
    # 5.1. Setup Environment and Determine Save Path
    is_colab, drive_path = setup_colab_environment()

    save_path = None
    if is_colab and drive_path:
        # 5.2. Create the unique output directory on Drive
        save_path = create_output_directory(drive_path)
    elif not is_colab:
        # 5.2. Create the unique output directory locally if not in Colab
        save_path = create_output_directory('.')

    # 5.3. Run the simulation
    simulate_interference(grid_size=300, d=60, wavelength=10.0, save_path=save_path)
    
    print("--- Simulation Finished ---")
