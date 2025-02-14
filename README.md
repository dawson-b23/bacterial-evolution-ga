# Bacterial Evolution Simulator 🔬

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A genetic algorithm simulation studying trade-offs between bacterial growth rates and lag phases under fluctuating resource conditions. Inspired by Lenski's long-term evolution experiment.

## 📌 Overview

This project implements:
- Consumer-resource ODE models with alternating substrates
- Tournament selection genetic algorithm
- Mutation rate comparison across 5 parameters
- Experimental replication with statistical analysis
- Automated visualization of evolutionary trajectories

## 🧬 Key Features
- **Biologically Realistic Modeling**
  - Logistic growth with lag phase adaptation
  - Resource-dependent fitness landscapes
  - Trade-off constrained parameter space
- **Evolutionary Framework**
  - Configurable mutation rates (0.01-0.1)
  - Tournament selection with size control
  - Index-based mutation preserving trait correlations
- **Analysis Tools**
  - Multi-experiment averaging
  - Generation-by-generation parameter tracking
  - Comparative fitness visualization

## ⚙️ Installation

```bash
git clone https://github.com/dawson-b23/bacterial-evolution-ga.git
cd bacterial-evolution-ga
pip install -r requirements.txt
