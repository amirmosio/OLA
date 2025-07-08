# Online Learning Applications - Pricing Project

Complete implementation of all 5 project requirements for the Online Learning Applications course.

## Project Overview

This project implements **dynamic pricing algorithms** with **production constraints** using online learning techniques. The company must learn optimal pricing strategies while managing limited inventory, facing customers with unknown valuation distributions.

### Core Problem
- **T** rounds of pricing decisions
- **N** types of products  
- **P** discrete set of possible prices
- **B** total production capacity (budget constraint)
- Customers buy ALL products priced below their valuation

## Requirements Implemented

### Requirement 1: Single Product + Stochastic Environment
- **Algorithms**: UCB1 (ignoring constraint) vs UCB with Inventory Constraint
- **Environment**: Stochastic valuations from normal distribution
- **Key Innovation**: UCB-like approach from the provided images with confidence bounds

### Requirement 2: Multiple Products + Stochastic Environment  
- **Algorithm**: Combinatorial UCB with inventory constraint
- **Environment**: Correlated valuations across products
- **Challenge**: Combinatorial optimization with budget constraints

### Requirement 3: Best-of-Both-Worlds Single Product
- **Algorithms**: UCB with Inventory vs Primal-Dual
- **Environments**: Stochastic + Highly Non-Stationary
- **Goal**: Good performance in both stochastic and adversarial settings

### Requirement 4: Best-of-Both-Worlds Multiple Products
- **Algorithms**: Combinatorial UCB vs Primal-Dual Multi-Product
- **Environments**: Stochastic + Non-Stationary multi-product
- **Innovation**: Decomposed primal-dual approach per product

### Requirement 5: Slightly Non-Stationary + Sliding Window
- **Algorithms**: Sliding Window Combinatorial UCB vs Primal-Dual
- **Environment**: Intervals with different but fixed distributions
- **Comparison**: Adaptive vs robust approaches

## Algorithm Details

### UCB with Inventory Constraint
Based on the UCB-like approach from the provided images:
```
f̄ₜᵁᶜᴮ(b) = f̂ₜ(b) + √(2log(T)/Nₜ₋₁(b))
c̄ₜᴸᶜᴮ(b) = ĉₜ(b) - √(2log(T)/Nₜ₋₁(b))

OPTₜ = { sup f̄ₜᵁᶜᴮ(γ) | γ∈ΔB, s.t. c̄ₜᴸᶜᴮ(γ) ≤ ρ }
```

### Primal-Dual Methods
- Exponential weights with dual variables for budget constraints
- Lagrangian approach: reward - λₜ × consumption
- Best-of-both-worlds guarantees

### Sliding Window UCB
- Maintains estimates using only recent observations
- Adapts to non-stationary environments
- Configurable window size

## Project Structure

```
OLA3/
├── algorithms/            # Algorithm implementations
│   ├── __init__.py
│   ├── base.py           # Base algorithm class
│   ├── ucb_algorithms.py # UCB-based algorithms
│   └── primal_dual_algorithms.py # Primal-dual methods
├── environments/         # Environment implementations
│   ├── __init__.py
│   ├── base.py          # Base environment class
│   ├── stochastic_environments.py
│   └── non_stationary_environments.py
├── simulation/          # Simulation framework
│   ├── __init__.py
│   ├── simulation.py    # Main simulation class
│   └── experiments.py   # Experiment configurations
├── main.py             # Entry point and CLI
├── requirements.txt    # Python dependencies
├── README.md          # This documentation
└── results/           # Structured output folder
    ├── requirement_1/
    │   ├── plots/     # Visualization plots
    │   └── data/      # CSV files with results
    ├── requirement_2/
    │   ├── plots/
    │   └── data/
    ├── requirement_3/
    │   ├── plots/
    │   └── data/
    ├── requirement_4/
    │   ├── plots/
    │   └── data/
    └── requirement_5/
        ├── plots/
        └── data/
```

## Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Run All Requirements
```bash
python main.py --requirement all --runs 10 --save-plots --save-csv
```

### Run Specific Requirement
```bash
python main.py --requirement 1 --runs 5
python main.py --requirement 3 --save-plots --save-csv
```

### Command Line Options
```bash
python main.py --help

Options:
  -r, --requirement {1,2,3,4,5,all}  Which requirement to run (default: all)
  -n, --runs INT                     Number of runs (default: 5)
  -s, --save-plots                   Save plots as PNG files
  -c, --save-csv                     Save numerical results to CSV files
  --seed INT                         Random seed (default: 42)
```

## Expected Output

The program will:
1. **Run experiments** with progress bars showing remaining time
2. **Display results** with mean ± std revenue for each algorithm
3. **Generate plots** showing:
   - Revenue comparison across environments
   - Cumulative revenue over time
   - Budget usage patterns
   - Performance heatmap
4. **Save outputs** to structured `results/` folder:
   - **Visualizations** (if `--save-plots` specified) in `results/requirement_X/plots/`
   - **CSV files** (if `--save-csv` specified) in `results/requirement_X/data/`:
     - Summary CSV with aggregated statistics
     - Detailed CSV with individual run results

### Results Directory Structure
After running experiments, your results folder will be organized as:
```
results/
├── requirement_1/
│   ├── plots/requirement_1_results.png
│   └── data/
│       ├── requirement_1_summary_TIMESTAMP.csv
│       └── requirement_1_detailed_TIMESTAMP.csv
├── requirement_2/
│   ├── plots/requirement_2_results.png
│   └── data/...
└── ...
```

## Key Results & Insights

### Performance Expectations
- **UCB with Inventory** should outperform naive UCB1 when budget is constraining
- **Primal-Dual** methods should be robust across stochastic and adversarial settings
- **Sliding Window** should adapt better to non-stationary environments
- **Combinatorial UCB** should handle multi-product correlations effectively

### Theoretical Guarantees
- UCB algorithms: O(√T log T) regret in stochastic settings
- Primal-dual: Best-of-both-worlds bounds
- All algorithms respect budget constraints

## Implementation Notes

### Key Features
- **Accurate UCB-like approach** following provided algorithm images
- **Proper confidence bounds** for revenue (upper) and consumption (lower)
- **Budget constraint handling** throughout all algorithms
- **Comprehensive evaluation** with statistical significance
- **Professional visualizations** with seaborn styling
- **Structured output organization** for easy analysis

### Technical Details
- Numpy for efficient computation
- Matplotlib/Seaborn for visualization
- Proper statistical analysis with confidence intervals
- CSV export for further analysis in external tools
- Modular design for easy extension
- Organized file structure for reproducible research

### Validation
- Implementation matches theoretical algorithm descriptions
- Results show expected behavior (UCB exploring initially, then exploiting)
- Budget constraints properly enforced across all algorithms
- Performance differences align with theoretical expectations
