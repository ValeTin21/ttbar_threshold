# Data Preprocessing Pipeline

This branch contains the complete data preprocessing pipeline for converting raw ROOT files into ML-ready datasets for top quark pair physics analysis.

## 📁 Files Overview

- **`DataPreProcessing.ipynb`** - Main preprocessing notebook with complete analysis pipeline
- **`DataHandling.py`** - Physics utilities for 4-vector operations and kinematic calculations  
- **`Plotting.py`** - Visualization functions for physics distributions and analysis plots

## 🔄 Processing Workflow

1. **ROOT File Loading** - Import particle collision data from simulation files
2. **4-Vector Creation** - Construct physics objects (top quarks, leptons, jets, neutrinos)
3. **Quality Cuts** - Filter unphysical events (mass cuts, kinematic constraints)
4. **Feature calculation** - Calculate 13 physics variables for ML training
5. **Production Classification** - Categorize events by collision type (gg, gq, qq)
6. **Data Export** - Save processed datasets for neural network training

## 🎯 Key Outputs

- **Physics Variables**: β, invariant masses, angular separations (ΔR, Δη), helicity angles
- **Event Classification**: Signal (gg→ttbar) vs Background (gq/qq→ttbar)
- **Monte Carlo Weights**: Statistical importance weighting for realistic simulations
- **Quality Assurance**: Event filtering and distribution validation plots
