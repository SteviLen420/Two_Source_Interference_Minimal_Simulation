Minimal Simulation of Two-Source Wave Interference
Author: Stefan Len

License: MIT License

Overview
This repository contains the Python source code for a minimal numerical simulation of the classic two-source wave interference pattern. I developed this script as a reproducible and pedagogical tool to visually demonstrate the principle of superposition.

The code is designed to be lightweight, easy to run, and serves as the supplementary material for my corresponding submission to the physics.gen-ph category on arXiv. The pipeline is fully automated, handling environment setup (for Google Colab) and organizing all outputs into timestamped directories for clarity and reproducibility.

Features
Minimalist & Focused: The simulation focuses solely on the core physics of wave superposition without unnecessary complexity.

Google Colab Ready: The script automatically detects a Colab environment, mounts Google Drive, and handles dependencies. This is the recommended way to ensure perfect reproducibility.

Automated Output Management: All generated files (plots and raw data) are saved into a unique, timestamped directory (run_YYYYMMDD_HHMMSS) to prevent overwriting results from different runs.

Data Export: The simulation saves not only the visual interference pattern as a high-quality PNG image but also the underlying raw intensity data as a CSV file for further analysis.

Requirements
To run this simulation, you will need:

Python 3.x

NumPy

Matplotlib

If you use the Google Colab environment, these dependencies are pre-installed, and the script will handle everything for you.

Usage Instructions
I have designed the code to be executable in two primary ways: via Google Colab (recommended for ease of use) or on a local machine.

Option 1: Running in Google Colab (Recommended)
This is the simplest method to replicate my results.

Open in Colab: Navigate to the Two_Source_Interference_Minimal_Simulation.py file in this repository and open it directly in Google Colab.

Run the Script: Run the entire script (from the "Runtime" menu, select "Run all").

Authorize Google Drive: The first time you run it, you will be prompted to authorize Colab to access your Google Drive. Please follow the on-screen instructions. This is necessary for saving the output files.

Find the Results: After the simulation completes, all outputs will be saved in a new folder within your Google Drive at the following path: MyDrive/Interference_Sims/run_YYYYMMDD_HHMMSS/.

Option 2: Running Locally
If you prefer to run the simulation on your own machine:

Clone the Repository: Open a terminal and clone this repository to your local machine using git:

Bash

git clone https://github.com/SteviLen420/Two_Source_Interference_Minimal_Simulation.git
Navigate to the Directory:

Bash

cd Two_Source_Interference_Minimal_Simulation
Run the Python Script: Execute the script from your terminal:

Bash

python Two_Source_Interference_Minimal_Simulation.py
Find the Results: The script will create a new directory named Interference_Sims in the same folder. Inside, you will find a timestamped sub-directory (run_...) containing the output files.

Output Files
For each simulation run, two files are generated:

interference_pattern.png: A high-resolution (300 DPI) image visualizing the calculated wave intensity. This plot clearly shows the constructive and destructive interference fringes.


Shutterstock
intensity_data.csv: A CSV file containing the raw numerical matrix of the intensity values for every point on the simulation grid. This allows for independent analysis and replotting of the data.

Customizing the Simulation
You can easily modify the physical parameters of the simulation by editing the following variables in the if __name__ == '__main__': block at the end of the script:

grid_size: An integer that defines the half-size of the simulation area. A larger value results in a larger area and higher resolution.

d: The distance (in grid units) separating the two wave sources.

wavelength: The wavelength (λ) of the waves.

Changing these parameters will directly affect the resulting interference pattern, allowing for interactive exploration of the phenomenon.

How to Cite
If you use this code or the results in your research, I would appreciate a citation. Please cite the GitHub repository directly. Once the corresponding arXiv paper is published, I will update this section with the formal citation information.

@misc{Len_Interference_Sim_2025,
  author = {Stefan Len},
  title = {A Minimal Python Simulation of Two-Source Wave Interference},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/SteviLen420/Two_Source_Interference_Minimal_Simulation}}
}
