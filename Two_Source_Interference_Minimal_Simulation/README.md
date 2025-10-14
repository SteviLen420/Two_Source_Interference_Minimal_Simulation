# Minimal Simulation of Two-Source Wave Interference

**Author:** Stefan Len

**License:** MIT License

**Repository:** https://github.com/SteviLen420/Two_Source_Interference_Minimal_Simulation

## Overview: Numerical Demonstration of Superposition

This repository contains a concise **Python source code** designed for a minimal numerical simulation of the classic **two-source wave interference pattern**.

The primary purpose of this script is to serve as a **reproducible** and **pedagogical tool** to visually and quantitatively demonstrate the principle of **wave superposition**. It calculates the resulting intensity field when two coherent point sources radiate waves across a two-dimensional grid.

## Key Features

The simulation pipeline is engineered for efficiency and robust output management:

* **Minimalist & Focused Physics:** The implementation focuses strictly on the core physics of wave superposition ($A_{total} = A_1 + A_2$) and intensity calculation ($I \propto A_{total}^2$), ensuring clarity without unnecessary computational complexity.
* **Google Colab Ready:** The script includes an automated routine to detect the Google Colab environment. When executed in Colab, it automatically mounts **Google Drive** and ensures all required dependencies are met.
* **Automated Output Management:** For maximal **reproducibility**, all output files (plots and raw data) are saved into a unique, timestamped directory to prevent data corruption or overwriting. The structure is: `MyDrive/Interference_Sims/run_YYYYMMDD_HHMMSS/`.
* **Comprehensive Data Export:** The simulation saves both the visual result (PNG) and the underlying numerical data (CSV), facilitating independent validation and further post-processing.

## Requirements

The simulation relies on standard scientific Python libraries:

* **Python 3.x**
* **NumPy** (for array and mathematical operations)
* **Matplotlib** (for visualization)

These dependencies are typically pre-installed in Google Colab, making it the most seamless execution environment.

## Usage Instructions

The code is configured for two main execution methods, prioritizing ease of use through Google Colab.

### **Option 1: Running in Google Colab (Recommended)**

This method is highly recommended for straightforward execution and perfect reproducibility, as the environment setup is fully automated.

1.  **Open in Colab:** Navigate to the `Two_Source_Interference_Minimal_Simulation.py` file and open it directly in a Google Colab notebook.
2.  **Execute the Script:** Run the entire notebook (via the "Runtime" menu $\to$ "Run all").
3.  **Authorize Google Drive:** You will be prompted once to authorize Colab to access your Google Drive. This is mandatory for the automated saving of the output files.
4.  **Retrieve Results:** Upon completion, all generated plots and data files will be organized and saved in your Google Drive at the following path:
    `MyDrive/Interference_Sims/run_YYYYMMDD_HHMMSS/`

### **Option 2: Running Locally**

To execute the simulation on a local machine (requires the Python dependencies listed above):

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/SteviLen420/Two_Source_Interference_Minimal_Simulation.git](https://github.com/SteviLen420/Two_Source_Interference_Minimal_Simulation.git)
    ```
2.  **Navigate and Run:**
    ```bash
    cd Two_Source_Interference_Minimal_Simulation
    python Two_Source_Interference_Minimal_Simulation.py
    ```
3.  **Find the Results:** The outputs will be saved to a similar timestamped directory structure in your local project folder: `./Interference_Sims/run_YYYYMMDD_HHMMSS/`

## Output Files

Each successful simulation generates two critical files within the run directory:

| Filename | Description | Content |
| :--- | :--- | :--- |
| **interference\_pattern.png** | **High-Resolution Plot** | A $300 \text{ DPI}$ image visualizing the calculated wave intensity. This figure clearly illustrates the constructive and destructive interference fringes. |
| **intensity\_data.csv** | **Raw Numerical Data** | A Comma-Separated Values file containing the full $N \times N$ numerical matrix of the intensity values across the simulation grid. This allows for independent analysis and data manipulation. |

## Customizing Physical Parameters

The core physical setup can be modified directly within the `if __name__ == '__main__':` block at the end of the script, allowing for direct exploration of the interference phenomenon:

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `grid_size` | *int* | Defines the half-size of the simulation area. A larger value increases both the domain size and computational resolution. |
| `d` | *int* | The distance (in grid units) separating the two point wave sources. |
| `wavelength` | *float* | The wavelength ($\lambda$) of the coherent waves. |

By changing these input variables, one can observe the direct impact of source separation and wavelength on the resulting interference fringe spacing and pattern geometry.

## How to Cite

If you utilize this code or its generated results in any publication, teaching materials, or research project, proper citation is requested.

Please cite the GitHub repository directly:

```bibtex
@misc{Len_Interference_Sim_2025,
  author = {Stefan Len},
  title = {A Minimal Python Simulation of Two-Source Wave Interference},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{[https://github.com/SteviLen420/Two_Source_Interference_Minimal_Simulation](https://github.com/SteviLen420/Two_Source_Interference_Minimal_Simulation)}}
}
