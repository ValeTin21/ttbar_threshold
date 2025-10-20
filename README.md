# Top Quark Pair Physics Analysis with Deep Neural Networks

A comprehensive physics analysis pipeline for studying top quark pair (ttbar) production using deep neural networks to classify signal vs background events in particle physics data.

## 🔬 Project Overview

This repository contains a complete analysis workflow for studying top quark pair production in particle physics, specifically focusing on **signal vs background classification** using Monte Carlo simulated data. The project implements advanced data preprocessing, feature engineering, and deep learning techniques to identify physics signatures in high-energy particle collisions.

## 📊 Physics Context

- **Dataset**: Simulated particle collision events from ROOT files
- **Physics Process**: Top quark pair (ttbar) production analysis
- **Classification Task**: Signal (gg → ttbar) vs Background (gq/qq → ttbar) 
- **Key Physics Variables**: 13 kinematic and angular features including invariant masses, angular separations (ΔR, Δη), helicity angles, and momentum distributions
- **Event Reconstruction**: Includes hadronic and leptonic top quark reconstruction with neutrino momentum estimation

## 🚀 Repository Structure
- main
  - DataProcessing #contaning file used to create the input for DNN
  - Plots #Containing plots
  
## 🛠️ Key Features

### Data Processing Pipeline
- **ROOT File Handling**: Efficient loading and processing of particle physics data
- **4-Vector Mathematics**: Advanced kinematics calculations using the `vector` library
- **Feature Engineering**: Physics-motivated variable construction (invariant masses, angular correlations)
- **Monte Carlo Weights**: Proper statistical weighting for realistic physics simulations
- **Data Quality**: Automatic filtering of unphysical events and outliers

### Deep Learning Implementation
- **Architecture**: Binary classification DNN (128→64→1 neurons)
- **Framework**: TensorFlow/Keras
- **Training Strategy**: Weighted training with Monte Carlo statistical importance
- **Model Interpretability**: Multiple feature importance analysis methods

### Physics Analysis Tools
- **Production Classification**: Automatic categorization of collision types (gg, gq, qq)
- **Angular Variables**: Helicity angles, spin correlations (cos θ*, D-variable)
- **Kinematic Distributions**: Mass spectra, momentum transfers, jet multiplicities
- **Statistical Analysis**: Proper treatment of Monte Carlo uncertainties
