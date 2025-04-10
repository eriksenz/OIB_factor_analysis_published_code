# OIB_factor_analysis_published_code
This repository contains all code and data used to reproduce the analysis and results from:

**Eriksen et al. (2025)** – _Disentangling partial melting and crustal recycling signatures in ocean island basalts with multivariate statistics_, currently under review at *Geochemistry, Geophysics, Geosystems (G-Cubed)*.

---

## Overview

This project applies compositional data analysis and multivariate statistics—including principal factor analysis (PFA) and bootstrapped partial least squares (PLS)—to fractional crystallization-corrected ocean island basalt (OIB) compositions. The aim is to isolate and quantify the relative effects of partial melting and source heterogeneity on OIB compositions, and to elucidate the origin of enriched mantle signatures (e.g., EM-1, EM-2, HIMU).

---

## Repository Structure

### 1. 📁 **Determining ol+cpx fractionation ratios**

- `determination_functions.py` – Functions used for calculating crystallizing assemblage ratios.
- `ol_cpx_ratio_determination.ipynb` – Jupyter notebook that determines the appropriate olivine–clinopyroxene (ol+cpx) ratio for each OIB island by matching slopes in CaO/Al₂O₃ vs. MgO space.
- Input: `OIB_majors_filtered.csv`
- Output: Island-dependent ol-cpx ratios are summarized in Table S2 of the manuscript.

### 2. 📁 **Correcting for ol+cpx fractionation**

- `correction_functions.py` – Functions to correct major and trace element concentrations for ol or ol+cpx crystallization.
- `Ol+cpx_fractionation_correction.ipynb` – Jupyter notebook that executes these corrections using the ratios from step 1.
- Input: `OIB_major+trace_filtered.csv`  
- Output: `Ol+cpx_fractionation_correction_output.csv` – Fractional crystallization-corrected dataset.

### 3. **Multivariate Analysis of OIB**

- 📁 `OIB_GeoRoc_files` - GeoRoc excel files of compositional data for each archipelago (Query date: May 06 2024).
- `bootstrap.py` – Self-written bootstrap resampling functions.
- `processing.py` – Functions to post-process PFA results.
- `compositional_analysis.py` – Functions for compositional data transformations (e.g., log-ratio transforms).
- `visualization.py` – Functions for visualizing PFA and PLS results.
- `bootstrap_pls.py` – Bootstrapped PLS implementation, including a custom VIP function.

#### Analysis Notebooks:

- `alkaline_OIB_PFA.ipynb` – **Part 1** of the statistical analysis. Applies log-ratio transformations and performs PFA on the full, corrected dataset (`Ol+cpx_fractionation_correction_output.csv`).
- `alkaline_OIB_PFA_F1_trimmed.ipynb` – **Part 2** of the analysis. Repeats PFA on a **trimmed** dataset (`OIB_data_F1_trimmed.csv`), focusing on OIB formed from similar degrees of partial melting (e.g., similar F1 scores from the first analysis).

---

## How to Reproduce the Analysis

1. Clone this repository and install any required dependencies (see below).
2. Run the notebooks in the following order:
   1. `ol_cpx_ratio_determination.ipynb`
   2. `Ol+cpx_fractionation_correction.ipynb`
   3. `alkaline_OIB_PFA.ipynb`
   4. `alkaline_OIB_PFA_F1_trimmed.ipynb`

---

## Requirements

This project uses Python and Jupyter Notebooks. The following libraries are required:

- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`
- `scipy`
- `composition-stats` https://composition-stats.readthedocs.io/en/latest/
- `factor_analyzer` https://factor-analyzer.readthedocs.io/en/latest/
- `joblib`
- `seaborn`
- `plotly`
- `cartopy`
