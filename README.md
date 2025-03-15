# Physics 129AL - Section Worksheet Week 7A

This repository contains the implementation of the tasks assigned in the Physics 129AL Section Worksheet for Week 7A.

## Project Structure

The project is organized into the following directories:

```
.
├── Task1/
│   ├── bayesian/           # Task 1a: Bayesian Inference
│   ├── stirling/           # Task 1b: Stirling's Approximation
│   └── bootstrapping/      # Task 1c: Bootstrapping
├── Task2/
│   ├── vacuum/             # Task 2a: Vacuum Decay Analysis
│   └── cavity/             # Task 2b: Cavity Decay Analysis
├── output/                 # Generated output files and plots
├── dataset_3.json          # Coin flip dataset
├── Vacuum_decay_dataset.json  # Vacuum decay dataset
├── Section_worksheet_Week7A.pdf  # Assignment PDF
├── run_all.py              # Script to run all tasks
└── README.md               # This file
```

## Requirements

- Python 3.6+
- NumPy
- SciPy
- Matplotlib

## How to Run

You can run all tasks at once using the `run_all.py` script:

```bash
python run_all.py
```

Or you can run individual tasks:

```bash
# Task 1a: Bayesian Inference
python Task1/bayesian/bayesian_inference.py

# Task 1b: Stirling's Approximation
python Task1/stirling/stirling_approximation.py

# Task 1c: Bootstrapping
python Task1/bootstrapping/bootstrapping.py

# Task 2a: Vacuum Decay Analysis
python Task2/vacuum/vacuum_decay_analysis.py

# Task 2b: Cavity Decay Analysis
python Task2/cavity/cavity_decay_analysis.py
```

## Task Descriptions

### Task 1: Statistical Inference on Biased Coins

This task involves analyzing three datasets of coin flips to investigate potential biases.

#### Task 1a: Bayesian Inference

Applies Bayesian inference to calculate likelihood functions and posterior distributions for the coin flip datasets.

#### Task 1b: Stirling's Approximation

Numerically checks Stirling's approximation for factorial calculations.

#### Task 1c: Bootstrapping

Applies bootstrapping to the datasets with various sample sizes to analyze the statistical properties.

### Task 2: Particle Decay

This task involves analyzing particle decay data from two environments: vacuum and optical cavity.

#### Task 2a: Vacuum Decay Analysis

Analyzes the vacuum decay dataset to determine the decay constant using maximum likelihood estimation.

#### Task 2b: Cavity Decay Analysis

Analyzes the optical cavity decay dataset, which contains a mixture of two particle types with different decay properties.

## Results

All results, including plots and numerical data, are saved in the `output/` directory, organized by task. 
