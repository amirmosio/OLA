# Online Learning Applications - Dynamic Pricing with Production Constraints

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**[📊 Project Presentation (Slides)](https://docs.google.com/presentation/d/1CM8hQ3F47ACscDTSUTxuNc1wGGrBHgnsLUYWrx-lGXM/edit?usp=sharing)**

A comprehensive implementation of **online learning algorithms for dynamic pricing** under production constraints, developed for the Online Learning Applications course. This project addresses the fundamental problem of **revenue optimization** when a company must learn optimal pricing strategies while managing limited inventory and facing customers with unknown valuation distributions.

## 🎯 Project Overview

### Problem Statement

Consider a company that must set prices dynamically for **N** types of products over **T** rounds, subject to:
- **Production constraint**: Total budget **B** (inventory capacity)
- **Unknown customer valuations**: Customers have private valuations for each product
- **All-or-nothing purchases**: Customers buy ALL products priced below their valuations
- **Discrete price space**: Prices must be chosen from a finite set **P**

This creates a challenging **exploration-exploitation** trade-off where the company must:
1. **Learn** optimal prices through customer interactions
2. **Manage** limited production capacity efficiently  
3. **Adapt** to different environmental conditions (stochastic vs. adversarial)

### Theoretical Foundation

The algorithms implemented are based on recent advances in:
- **Upper Confidence Bound (UCB)** methods with constraints
- **Primal-dual** approaches for best-of-both-worlds guarantees
- **Combinatorial optimization** for multi-product settings
- **Sliding window** techniques for non-stationary environments

## 🏗️ Architecture & Implementation

### Core Components

```
OLA3/
├── algorithms/                    # Algorithm implementations
│   ├── base.py                   # Abstract base class for all algorithms
│   ├── ucb_algorithms.py         # UCB-based methods
│   └── primal_dual_algorithms.py # Primal-dual methods
├── environments/                 # Environment implementations  
│   ├── base.py                   # Abstract base class for environments
│   ├── stochastic_environments.py        # Stochastic settings
│   └── non_stationary_environments.py    # Non-stationary settings
├── simulation/                   # Experimental framework
│   ├── simulation.py             # Main simulation engine
│   └── experiments.py            # Experiment configurations
├── main.py                       # Command-line interface
├── requirements.txt              # Python dependencies
└── results/                      # Generated outputs
    └── requirement_X/            # Organized by project requirement
        ├── plots/                # Visualizations
        └── data/                 # CSV exports
```

## 📋 Requirements Implementation

The project implements **five comprehensive requirements** as specified in the course:

### Requirement 1: Single Product + Stochastic Environment
**Objective**: Compare UCB approaches with and without inventory constraints

**Algorithms**:
- `UCB1SingleProduct`: Classical UCB1 ignoring budget constraints
- `UCBWithInventoryConstraintSingleProduct`: UCB with budget-aware confidence bounds

**Environment**: 
- `StochasticSingleProductEnvironment`: Customer valuations from fixed normal distribution

**Mathematical Formulation**:
```
UCB with Inventory uses confidence bounds:
f̄ₜᵁᶜᴮ(p) = f̂ₜ(p) + √(2log(T)/Nₜ₋₁(p))    (revenue upper bound)
c̄ₜᴸᶜᴮ(p) = ĉₜ(p) - √(2log(T)/Nₜ₋₁(p))    (consumption lower bound)

Optimization: max f̄ₜᵁᶜᴮ(p) s.t. c̄ₜᴸᶜᴮ(p) ≤ ρₜ
```

### Requirement 2: Multiple Products + Stochastic Environment  
**Objective**: Handle correlated valuations across multiple products

**Algorithm**:
- `CombinatorialUCB`: Uses Hungarian algorithm for optimal price assignment

**Environment**:
- `StochasticMultiProductEnvironment`: Multivariate normal valuations with correlation

**Innovation**: Solves combinatorial optimization with budget constraints using assignment algorithms

### Requirement 3: Best-of-Both-Worlds Single Product
**Objective**: Achieve good performance in both stochastic and adversarial settings

**Algorithms**:
- `UCBWithInventoryConstraintSingleProduct`: Optimized for stochastic case
- `PrimalDualSingleProduct`: Robust to adversarial sequences

**Environments**:
- `StochasticSingleProductEnvironment`: Fixed distribution baseline
- `HighlyNonStationaryEnvironment`: Rapidly changing distributions

**Theoretical Guarantee**: Primal-dual achieves O(√T) regret in both settings

### Requirement 4: Best-of-Both-Worlds Multiple Products
**Objective**: Extend robustness to multi-product scenarios

**Algorithms**:
- `CombinatorialUCB`: Stochastic optimization approach
- `PrimalDualMultiProduct`: Decomposed regret minimization per product

**Key Insight**: Problem decomposes allowing independent regret minimizers per product

### Requirement 5: Slightly Non-Stationary + Sliding Window
**Objective**: Adapt to environments with piecewise-stationary changes

**Algorithm**:
- `SlidingWindowCombinatorialUCB`: Uses recent observations with configurable window
- `PrimalDualMultiProduct`: Baseline robust approach

**Environment**:
- `SlightlyNonStationaryEnvironment`: Fixed distributions within intervals, different across intervals

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd OLA3

# Install dependencies
pip install -r requirements.txt
```

### Requirements
- Python 3.8+
- NumPy ≥ 1.21.0 (numerical computation)
- Matplotlib ≥ 3.5.0 (visualization)
- Seaborn ≥ 0.11.0 (statistical plots)
- SciPy ≥ 1.7.0 (optimization algorithms)
- CVXPY ≥ 1.2.0 (convex optimization)
- Pandas ≥ 1.5.0 (data manipulation)
- tqdm ≥ 4.64.0 (progress bars)

### Usage Examples

#### Run All Requirements
```bash
python main.py --requirement all --runs 10 --save-plots --save-csv
```

#### Focus on Specific Requirement
```bash
# Test UCB algorithms on single product
python main.py --requirement 1 --runs 20 --save-plots

# Compare best-of-both-worlds approaches
python main.py --requirement 3 --runs 15 --save-csv

# Analyze sliding window performance
python main.py --requirement 5 --runs 10 --save-plots --save-csv
```

#### Advanced Configuration
```bash
# Custom parameters with specific seed
python main.py --requirement 2 --runs 25 --seed 2024 --save-plots --save-csv

# Quick testing run
python main.py --requirement 1 --runs 3
```

### Command Line Interface

```bash
python main.py [OPTIONS]

Options:
  -r, --requirement {1,2,3,4,5,all}  Which requirement to run [default: all]
  -n, --runs INTEGER                 Number of experiment runs [default: 5]  
  -s, --save-plots                   Save visualizations as PNG files
  -c, --save-csv                     Export results to CSV format
  --seed INTEGER                     Random seed for reproducibility [default: 42]
  --help                             Show this help message
```

## 📊 Expected Results & Analysis

### Performance Characteristics

**Requirement 1 (Single Product)**:
- UCB with inventory should achieve **15-25% higher revenue** than naive UCB1 when budget is constraining
- Both algorithms show **exploration-exploitation** trade-off in early rounds

**Requirement 3 (Best-of-Both-Worlds)**:
- UCB excels in **stochastic environments** (predictable performance)
- Primal-dual robust in **highly non-stationary** settings (consistent performance)
- **Performance gap** illustrates stochastic vs. adversarial trade-offs

**Requirement 5 (Sliding Window)**:
- Sliding window adapts **faster to distribution changes** 
- Window size creates **bias-variance trade-off** (smaller = more adaptive, larger = more stable)

### Generated Outputs

Running experiments produces:

**Structured Results Directory**:
```
results/
├── requirement_1/
│   ├── plots/
│   │   ├── requirement_1_revenue_comparison.png      # Algorithm performance comparison
│   │   ├── requirement_1_revenue_growth.png          # Cumulative revenue over time
│   │   ├── requirement_1_budget_management.png       # Budget utilization patterns
│   │   └── requirement_1_efficiency_heatmap.png      # Revenue-per-round efficiency
│   └── data/
│       ├── requirement_1_summary_TIMESTAMP.csv       # Aggregated statistics
│       └── requirement_1_detailed_TIMESTAMP.csv      # Individual run data
├── requirement_2/
│   └── ... (similar structure)
└── ...
```

**Visualization Types**:
1. **Revenue Comparison**: Bar charts with error bars showing mean ± std performance
2. **Growth Trajectories**: Line plots of cumulative revenue evolution  
3. **Budget Management**: Budget utilization patterns over time
4. **Efficiency Heatmaps**: Revenue-per-round efficiency across algorithms and environments

**Data Export Format**:
- **Summary CSV**: Aggregated metrics (mean revenue, standard deviation, regret)
- **Detailed CSV**: Individual run results for statistical analysis

## 🧮 Mathematical Foundations

### UCB with Inventory Constraint

The algorithm maintains estimates and confidence bounds:

```
Revenue estimate: f̂ₜ(p) = (1/Nₜ(p)) × Σᵢ revenue_i(p)
Consumption estimate: ĉₜ(p) = (1/Nₜ(p)) × Σᵢ units_sold_i(p)

Confidence radius: conf_t(p) = √(2log(T)/Nₜ(p))

Upper confidence bound: f̄ₜ(p) = f̂ₜ(p) + conf_t(p)
Lower confidence bound: c̄ₜ(p) = ĉₜ(p) - conf_t(p)
```

**Optimization Problem**:
```
πₜ = argmax_{p∈P} f̄ₜ(p)
subject to: c̄ₜ(p) ≤ remaining_budget / remaining_rounds
```

### Primal-Dual Method

Uses exponential weights with Lagrangian formulation:

```
Dual variable update: λₜ₊₁ = λₜ + η × (consumption_rate - target_rate)
Lagrangian reward: L(p,λ) = revenue(p) - λ × consumption(p)
Weight update: wₜ₊₁(p) = wₜ(p) × exp(η × L(p,λₜ))
```

**Best-of-Both-Worlds Property**: Achieves O(√T) regret in both stochastic and adversarial settings.

### Combinatorial UCB

For multiple products, solves assignment problem:

```
Cost matrix: C[i,j] = -f̄ₜ(i,j) + penalty × constraint_violation(i,j)
Assignment: Hungarian algorithm on cost matrix
Budget constraint: Σᵢ c̄ₜ(assignment[i]) ≤ budget_allocation
```

## 🔬 Implementation Details

### Key Design Decisions

**Numerical Stability**:
- Log-space weight updates in primal-dual to prevent overflow
- Clipped confidence bounds to handle edge cases
- Proper handling of infinite prices (products not sold)

**Algorithmic Innovations**:
- **Hungarian algorithm** for combinatorial optimization
- **Sliding window** with efficient recomputation
- **Budget allocation** strategies for multi-product scenarios

**Software Engineering**:
- **Abstract base classes** for extensibility
- **Deep copying** to prevent state contamination across runs
- **Progress tracking** with tqdm for long experiments
- **Structured output** organization for reproducible research

### Validation & Testing

The implementation includes:
- **Theoretical consistency**: Algorithms match published specifications
- **Budget constraint enforcement**: All algorithms respect capacity limits
- **Statistical significance**: Multiple runs with confidence intervals
- **Edge case handling**: Boundary conditions and parameter limits

## 📈 Research Extensions

This codebase provides a foundation for exploring:

1. **Alternative constraint types** (per-product budgets, temporal constraints)
2. **Different customer models** (strategic behavior, multiple customers per round)
3. **Advanced algorithms** (Thompson sampling, contextual bandits)
4. **Real-world applications** (dynamic pricing in e-commerce, cloud resource allocation)

## 🎓 Academic Context

This project implements algorithms from:
- **Multi-armed bandits** with constraints
- **Online convex optimization** 
- **Mechanism design** theory
- **Approximation algorithms** for combinatorial problems

**Theoretical Contributions**:
- Extension of UCB to constrained settings
- Best-of-both-worlds analysis for pricing
- Combinatorial optimization under uncertainty

## 📚 References

Key theoretical foundations:
1. **UCB with Constraints**: Confidence bound approaches for resource-constrained bandits
2. **Primal-Dual Methods**: Best-of-both-worlds guarantees in online optimization
3. **Combinatorial Bandits**: Assignment algorithms for multi-armed scenarios
4. **Non-stationary Environments**: Sliding window and change detection techniques

## 🤝 Contributing

For course-related improvements:
1. Fork the repository
2. Create feature branch (`git checkout -b feature/improvement`)
3. Implement changes with proper testing
4. Commit with descriptive messages
5. Submit pull request with detailed explanation

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Note**: This implementation is designed for educational purposes as part of the Online Learning Applications course. The algorithms demonstrate key theoretical concepts while providing practical insights into dynamic pricing under constraints.
