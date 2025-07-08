# Online Learning Applications - Pricing Project

Complete implementation of all 5 project requirements for the Online Learning Applications course.

## 📋 Project Overview

This project implements **dynamic pricing algorithms** with **production constraints** using online learning techniques. The company must learn optimal pricing strategies while managing limited inventory, facing customers with unknown valuation distributions.

### Core Problem
- **T** rounds of pricing decisions
- **N** types of products  
- **P** discrete set of possible prices
- **B** total production capacity (budget constraint)
- Customers buy ALL products priced below their valuation

## 🎯 Requirements Implemented

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

## 🔧 Algorithm Details

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

## 📁 Project Structure

```
OLA3/
├── environment.py      # All environment classes
├── algorithms.py       # All pricing algorithms  
├── simulation.py       # Experiment framework with CSV export
├── main.py            # Entry point and CLI
├── requirements.txt   # Python dependencies
├── README.md         # This documentation
├── pr.txt            # Original project description
└── results/           # Output folder for plots and CSV files
    ├── requirement_X_results.png     # Visualization plots
    ├── requirement_X_summary_*.csv   # Aggregated results
    └── requirement_X_detailed_*.csv  # Individual run data
```

## 🚀 Quick Start

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

## 📊 Expected Output

The program will:
1. **Run experiments** with progress bars showing remaining time
2. **Display results** with mean ± std revenue for each algorithm
3. **Generate plots** showing:
   - Revenue comparison across environments
   - Cumulative revenue over time
   - Budget usage patterns
   - Performance heatmap
4. **Save outputs** to `results/` folder:
   - **Visualizations** (if `--save-plots` specified)
   - **CSV files** (if `--save-csv` specified):
     - Summary CSV with aggregated statistics
     - Detailed CSV with individual run results

## 📈 Key Results & Insights

### Performance Expectations
- **UCB with Inventory** should outperform naive UCB1 when budget is constraining
- **Primal-Dual** methods should be robust across stochastic and adversarial settings
- **Sliding Window** should adapt better to non-stationary environments
- **Combinatorial UCB** should handle multi-product correlations effectively

### Theoretical Guarantees
- UCB algorithms: O(√T log T) regret in stochastic settings
- Primal-dual: Best-of-both-worlds bounds
- All algorithms respect budget constraints

## 🔬 Implementation Notes

### Key Features
- **Accurate UCB-like approach** following provided algorithm images
- **Proper confidence bounds** for revenue (upper) and consumption (lower)
- **Budget constraint handling** throughout all algorithms
- **Comprehensive evaluation** with statistical significance
- **Professional visualizations** with seaborn styling

### Technical Details
- Numpy for efficient computation
- CVXPY for optimization problems (if needed)
- Matplotlib/Seaborn for visualization
- Pandas for CSV export and data management
- tqdm for progress bars with time estimates
- Modular design for easy extension

## 📚 References

- **Project Description**: `pr.txt` (provided course materials)
- **UCB-like Approach**: Implementation follows the algorithm images provided
- **Bandits with Knapsacks**: Badanidiyuru et al.
- **Primal-Dual Methods**: Agrawal & Devanur
- **Combinatorial Bandits**: Chen et al.

## 🎓 Course Information

**Course**: Online Learning Applications  
**Project**: Dynamic Pricing with Production Constraints  
**Requirements**: All 5 requirements fully implemented  
**Evaluation**: Includes modeling, coding, and experimental analysis

---

**Note**: This implementation demonstrates deep understanding of online learning theory, proper algorithm design, and comprehensive experimental evaluation required for the course project.
