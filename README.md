# Bacterial Evolution Simulator 🔬

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A genetic algorithm simulation studying trade-offs between bacterial growth rates and lag phases under fluctuating resource conditions. Inspired by Lenski's long-term evolution experiment.

Created for a class at the University of Idaho

My partners for this project: Matthew Kinahan, Mohammad Abbaspour

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
```

## 🧪 Usage

Run 3 experimental replicates:
```
python3 finalcode_project1.py
```

### Expected Output:
```text
experiment_1/
├── mutation_0.1/
│   ├── fitness_progression.png
│   ├── results_table.png
├── mutation_0.075/
│   ├── ...
...
average_fitness_across_experiments.png
```

## 📂 File Structure
```text
bacterial-evolution-ga/
├── experiments/                   # Auto-generated results
│   ├── experiment_1/              # First experimental run
│   │   ├── mutation_0.1/          # Mutation rate subdirectory
│   │   │   ├── fitness_progression.png  # Generation plot
│   │   │   └── results_table.png        # Parameter table
│   │   ├── mutation_0.075/
│   │   └── ... 
│   ├── experiment_2/              # Second replicate
│   └── experiment_3/              # Third replicate
├── finalcode_project1.py          # Main simulation code
├── average_fitness_across_experiments.png  # Combined results
├── README.md
└── requirements.txt
```

## 📊 Key Results

Experimental Findings:

1. Optimal mutation rate: 0.075 (balances exploration/exploitation)

2. 122% growth rate increase over 25 generations

3. 31% lag phase reduction despite enforced trade-off

4. High mutation (>0.05) populations show 2.1× faster early adaptation

## 🤝 Contributing
1. Fork the repository

2. Create feature branch (git checkout -b feature/yourfeature)

3. Commit changes (git commit -am 'Add some feature')

4. Push to branch (git push origin feature/yourfeature)

5. Open Pull Request

## 📜 License

MIT License - see LICENSE for details

## 📚 References

- Lenski, R. E., & Travisano, M. (1994). Dynamics of adaptation and diversification. PNAS, 91(15), 6808-6814.

- Monod, J. (1949). The growth of bacterial cultures. Annual Reviews of Microbiology, 3(1), 371-394.
