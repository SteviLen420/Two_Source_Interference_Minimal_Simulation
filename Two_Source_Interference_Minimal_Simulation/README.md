# Numerical Validation of Two-Source Wave Interference: A Reproducible Computational Framework

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![arXiv](https://img.shields.io/badge/arXiv-physics.comp--ph-b31b1b.svg)](https://arxiv.org/)

**Author:** Stefan Len  
**Version:** 1.1.0  
**License:** MIT License  
**Repository:** https://github.com/SteviLen420/Two_Source_Interference_Minimal_Simulation

---

## Overview

This repository provides a **validated numerical framework** for simulating classical two-source wave interference patterns with **quantitative comparison to analytical solutions**.

The code implements a grid-based wave superposition calculation and includes comprehensive validation against the far-field approximation, demonstrating numerical accuracy suitable for both **educational applications** and as a **methodological foundation** for more complex wave propagation studies.

### Scientific Objectives

1. **Numerical Implementation:** Discrete simulation of coherent wave superposition from two point sources
2. **Quantitative Validation:** Rigorous comparison of numerical results with analytical far-field predictions
3. **Reproducibility:** Complete automation of simulation pipeline with timestamped outputs and metadata logging
4. **Accessibility:** Cloud-ready execution environment (Google Colab) with automated dependency management

---

## Key Features

### **1. Physics Implementation**
- **Wave Superposition:** Direct calculation of total amplitude $A_{\text{total}} = A_1 + A_2$ on a 2D Cartesian grid
- **Intensity Field:** Computation of observable intensity $I \propto |A_{\text{total}}|^2$
- **Coherent Sources:** Two monochromatic point sources with adjustable separation and wavelength

### **2. Validation Framework**
- **Analytical Comparison:** Quantitative overlay with far-field approximation $I(y) = 4I_0 \cos^2\left(\frac{\pi d y}{\lambda L}\right)$
- **Fringe Spacing Analysis:** Automated peak detection and measurement of interference pattern periodicity
- **Error Quantification:** Statistical analysis across multiple parameter combinations (source separation $d$, wavelength $\lambda$)
- **Validation Visualization:** Two-panel comparison plots showing numerical vs. analytical solutions and residual differences

### **3. Reproducibility Infrastructure**
- **Timestamped Runs:** Unique output directories (`run_YYYYMMDD_HHMMSS/`) prevent data overwriting
- **Metadata Logging:** JSON-formatted parameter tracking with code version control
- **Complete Data Export:** High-resolution figures (300 DPI PNG), raw numerical data (CSV), and validation metrics
- **Summary Statistics:** Automated generation of validation summary reports

### **4. Cloud Integration**
- **Google Colab Ready:** Automatic environment detection and Google Drive mounting
- **Dependency Management:** Automated installation of scientific Python stack (NumPy, Matplotlib, SciPy)
- **Cross-Platform:** Seamless execution in cloud (Colab) or local environments

---

## Requirements

### **Core Dependencies**
- **Python 3.x** (tested on 3.8+)
- **NumPy** ≥ 1.19 (array operations, mathematical functions)
- **Matplotlib** ≥ 3.3 (visualization)
- **SciPy** ≥ 1.5 (signal processing for peak detection in validation)

### **Optional (Auto-installed in Colab)**
- Google Colab environment for cloud execution
- Google Drive API access for automated output storage

All dependencies are standard scientific Python libraries, pre-installed in Google Colab and available via `pip` for local installations.

---

## Installation and Usage

### **Method 1: Google Colab Execution (Recommended for Reproducibility)**

This method provides the most streamlined workflow with zero local setup requirements.

1. **Open in Colab:**
   - Navigate to the repository and open `Two_Source_Interference_Minimal_Simulation.py` in Google Colab
   - Or use: [Open in Colab](https://colab.research.google.com/) → Upload file

2. **Execute Complete Pipeline:**
```python
   # Run all cells via: Runtime → Run all
```

3. **Authorize Drive Access:**
   - First execution will prompt for Google Drive authorization
   - Required for automated output saving to persistent storage

4. **Retrieve Results:**
   - All outputs automatically saved to: `MyDrive/Interference_Sims/run_YYYYMMDD_HHMMSS/`
   - Includes: validation plots, data tables, metadata, summary statistics

### **Method 2: Local Execution**

For users preferring local computational control:

1. **Clone Repository:**
```bash
   git clone https://github.com/SteviLen420/Two_Source_Interference_Minimal_Simulation.git
   cd Two_Source_Interference_Minimal_Simulation
```

2. **Install Dependencies:**
```bash
   pip install numpy matplotlib scipy
```

3. **Run Simulation:**
```bash
   python Two_Source_Interference_Minimal_Simulation.py
```

4. **Access Results:**
   - Local output directory: `./Interference_Sims/run_YYYYMMDD_HHMMSS/`

---

## Output Structure

Each simulation run generates a comprehensive set of outputs for analysis and validation:
```
Interference_Sims/
└── run_YYYYMMDD_HHMMSS/
    ├── metadata.json                          # Simulation parameters and configuration
    ├── 2D_pattern_d60_l10.0.png              # Main interference pattern visualization
    ├── intensity_data_d60_l10.0.csv          # Raw 2D intensity field (NxN matrix)
    ├── validation_comparison_plot.png         # Numerical vs. analytical comparison
    ├── quantitative_validation_table.csv      # Fringe spacing measurements
    └── validation_summary.txt                 # Statistical error analysis
```

### **File Descriptions**

| Filename | Type | Purpose | Details |
|:---------|:-----|:--------|:--------|
| `metadata.json` | Metadata | Parameter documentation | Code version, simulation parameters, validation settings |
| `2D_pattern_*.png` | Figure | Interference visualization | 300 DPI, false-color intensity map with source positions |
| `intensity_data_*.csv` | Raw Data | Numerical results | Full 2D intensity field for independent analysis |
| `validation_comparison_plot.png` | Figure | Validation analysis | 2-panel: (1) Numerical vs. analytical overlay (2) Residual differences |
| `quantitative_validation_table.csv` | Data Table | Metrics across parameters | Measured vs. predicted fringe spacing, percentage errors |
| `validation_summary.txt` | Report | Statistical summary | Mean error, maximum error, validation pass/fail criteria |

---

## Physical Parameters and Customization

Modify simulation parameters in the main execution block:
```python
if __name__ == '__main__':
    # Core simulation parameters
    GRID_HALF_SIZE = 300      # Simulation domain: [-300, 300] × [-300, 300]
    MAIN_D = 60               # Source separation (grid units)
    MAIN_WAVELENGTH = 10.0    # Wavelength λ (grid units)
```

### **Parameter Guide**

| Parameter | Symbol | Type | Physical Meaning | Typical Range |
|:----------|:-------|:-----|:-----------------|:--------------|
| `GRID_HALF_SIZE` | — | `int` | Half-width of simulation domain | 100–500 |
| `MAIN_D` | $d$ | `float` | Distance between coherent sources | 20–100 |
| `MAIN_WAVELENGTH` | $\lambda$ | `float` | Wave wavelength | 5.0–20.0 |

### **Physical Scaling Considerations**

- **Fringe Spacing:** Approximately $\Delta y \approx \frac{\lambda L}{d}$ where $L$ = `GRID_HALF_SIZE`
- **Number of Fringes:** Scales as $N_{\text{fringes}} \sim \frac{2d}{\lambda}$ for full domain
- **Resolution Requirements:** Ensure $\lambda \geq 3$ grid units for adequate sampling (Nyquist criterion)

---

## Validation Methodology

### **Analytical Benchmark**

The numerical results are validated against the standard far-field two-slit interference formula:

$$I(y) = 4I_0 \cos^2\left(\frac{\pi d y}{\lambda L}\right)$$

where:
- $I_0$ = intensity from single source (normalized to unity)
- $d$ = source separation
- $y$ = position along observation line
- $\lambda$ = wavelength
- $L$ = far-field distance (set equal to `GRID_HALF_SIZE`)

### **Quantitative Metrics**

1. **Fringe Spacing Measurement:**
   - Peak detection via SciPy's `find_peaks` algorithm
   - Average spacing between consecutive intensity maxima
   - Comparison with theoretical prediction $\Delta y_{\text{theory}} = \frac{\lambda L}{d}$

2. **Error Analysis:**
   - Percentage error: $\epsilon = \frac{|\Delta y_{\text{measured}} - \Delta y_{\text{theory}}|}{|\Delta y_{\text{theory}}|} \times 100\%$
   - Statistical aggregation across multiple test cases
   - Validation threshold: errors should be < 5% for acceptable accuracy

3. **Test Suite:**
   - Four parameter combinations spanning different $d$ and $\lambda$ values
   - Ensures robustness across physical regimes

---

## Validation Results Summary

Expected performance metrics (typical output from `validation_summary.txt`):
```
=== Validation Summary ===
Total Test Cases: 4
Mean Error in Fringe Spacing: ~1-3%
Maximum Observed Error: ~3-5%
All errors < 5% threshold: True
```

**Interpretation:** The numerical implementation reproduces analytical predictions within acceptable computational accuracy, validating the framework for quantitative applications.

---

## Use Cases

### **1. Educational Applications**
- Interactive demonstration of wave interference principles
- Visualization of constructive/destructive interference
- Quantitative comparison of numerical and analytical methods
- Laboratory supplement for wave optics courses

### **2. Research Validation**
- Baseline verification for more complex wave propagation codes
- Testing of numerical discretization schemes
- Foundation for extensions: 3D geometries, time-dependent sources, incoherent sources

### **3. Methodological Development**
- Template for reproducible computational physics workflows
- Example of quantitative validation practices
- Demonstration of open-source scientific computing

---

## Limitations and Assumptions

### **Physical Approximations**
1. **2D Simplification:** Simulation uses 2D scalar waves; real optical systems are 3D vectorial
2. **Coherent Sources:** Assumes perfect phase coherence (monochromatic, fixed phase relationship)
3. **Far-Field Regime:** Analytical comparison valid for observation distances $L \gg d^2/\lambda$ (Fraunhofer approximation)
4. **Scalar Wave Equation:** Neglects polarization effects

### **Numerical Considerations**
1. **Finite Domain:** Periodic boundary effects may appear near grid edges
2. **Discretization:** Grid spacing must satisfy sampling criterion ($\Delta x < \lambda/3$)
3. **Computational Cost:** Memory scales as $O(N^2)$ for $N \times N$ grid

---

## Extending the Framework

### **Suggested Extensions**
- **3D Simulation:** Extend to volumetric wave propagation
- **Time Evolution:** Add temporal dynamics with phase evolution
- **Partial Coherence:** Include finite coherence length effects
- **Multiple Sources:** Generalize to $N$-source interference patterns
- **Non-linear Media:** Add intensity-dependent refractive index

### **Code Modification Points**
All extension-friendly functions are clearly documented with modular structure. Key modification targets:
- `simulate_interference()`: Core physics implementation
- `analytical_intensity()`: Theoretical comparison function
- Main execution block: Parameter scanning and batch processing

---

## Citation

If you use this framework in academic work, teaching, or research, please cite:
```bibtex
@software{Len_Interference_Validation_2025,
  author = {Stefan Len},
  title = {Numerical Validation of Two-Source Wave Interference: 
           A Reproducible Computational Framework},
  year = {2025},
  version = {1.1.0},
  publisher = {GitHub},
  url = {https://github.com/SteviLen420/Two_Source_Interference_Minimal_Simulation},
  license = {MIT}
}
```

---

## License

MIT License - See `LICENSE` file for full terms.

Copyright (c) 2025 Stefan Len

**Summary:** Free for academic, educational, and commercial use with attribution.

---

## Contact and Contributions

- **Issues:** Report bugs or request features via [GitHub Issues](https://github.com/SteviLen420/Two_Source_Interference_Minimal_Simulation/issues)
- **Pull Requests:** Contributions welcome for extensions or improvements
- **Questions:** Open a discussion thread in the repository

---

## Acknowledgments

This work implements standard wave interference physics as described in classical optics textbooks. The validation methodology follows best practices in computational physics for verification and validation of numerical codes.

---

**Version History:**
- **v1.1.0** (2025): Added quantitative validation framework, metadata logging, statistical analysis
- **v1.0.0** (2025): Initial release with basic simulation and visualization
