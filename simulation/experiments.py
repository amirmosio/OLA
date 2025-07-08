import numpy as np
from typing import Dict, Tuple

# Import from the new package structure
from environments import (
    StochasticSingleProductEnvironment,
    StochasticMultiProductEnvironment,
    NonStationaryEnvironment,
    SlightlyNonStationaryEnvironment
)
from algorithms import (
    UCB1SingleProduct,
    UCBWithInventoryConstraintSingleProduct,
    CombinatorialUCB,
    PrimalDualSingleProduct,
    PrimalDualMultiProduct,
    SlidingWindowCombinatorialUCB
)
from .simulation import PricingSimulation

def create_requirement_1_experiment():
    """Create experiment for Requirement 1: Single product stochastic"""
    
    # Define common parameters
    prices = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    T = 1000
    B = 500
    
    # Create environment
    env = StochasticSingleProductEnvironment(
        prices=prices, T=T, B=B, 
        valuation_mean=5.0, valuation_std=2.0
    )
    
    # Create algorithms
    algorithms = {
        'UCB1 (No Constraint)': UCB1SingleProduct(prices, T, B),
        'UCB with Inventory': UCBWithInventoryConstraintSingleProduct(prices, T, B)
    }
    
    return {'Stochastic Single Product': env}, algorithms

def create_requirement_2_experiment():
    """Create experiment for Requirement 2: Multiple products stochastic"""
    
    prices = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    T = 1000
    B = 500
    n_products = 3
    
    # Create environment
    mean_vals = np.array([4.0, 5.0, 6.0])
    cov_matrix = np.array([[2.0, 0.5, 0.3],
                          [0.5, 2.0, 0.4],
                          [0.3, 0.4, 2.0]])
    
    env = StochasticMultiProductEnvironment(
        n_products=n_products, prices=prices, T=T, B=B,
        mean_valuations=mean_vals, correlation_matrix=cov_matrix
    )
    
    # Create algorithms
    algorithms = {
        'Combinatorial UCB': CombinatorialUCB(prices, n_products, T, B)
    }
    
    return {'Stochastic Multi Product': env}, algorithms

def create_requirement_3_experiment():
    """Create experiment for Requirement 3: Best-of-both-worlds single product"""
    
    prices = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    T = 1000
    B = 500
    
    # Create environments
    stochastic_env = StochasticSingleProductEnvironment(
        prices=prices, T=T, B=B, valuation_mean=5.0, valuation_std=2.0
    )
    
    # Create highly non-stationary environment
    change_points = [200, 400, 600, 800]
    distributions = [
        {'mean': 3.0, 'std': 1.0},
        {'mean': 7.0, 'std': 1.5},
        {'mean': 4.0, 'std': 2.0},
        {'mean': 6.0, 'std': 1.2},
        {'mean': 5.0, 'std': 1.8}
    ]
    
    non_stationary_env = NonStationaryEnvironment(
        n_products=1, prices=prices, T=T, B=B,
        change_points=change_points, distributions=distributions
    )
    
    # Create algorithms
    algorithms = {
        'UCB with Inventory': UCBWithInventoryConstraintSingleProduct(prices, T, B),
        'Primal-Dual': PrimalDualSingleProduct(prices, T, B)
    }
    
    return {
        'Stochastic': stochastic_env,
        'Non-Stationary': non_stationary_env
    }, algorithms

def create_requirement_4_experiment():
    """Create experiment for Requirement 4: Best-of-both-worlds multiple products"""
    
    prices = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    T = 1000
    B = 500
    n_products = 3
    
    # Create stochastic environment
    mean_vals = np.array([4.0, 5.0, 6.0])
    cov_matrix = np.array([[2.0, 0.5, 0.3],
                          [0.5, 2.0, 0.4],
                          [0.3, 0.4, 2.0]])
    
    stochastic_env = StochasticMultiProductEnvironment(
        n_products=n_products, prices=prices, T=T, B=B,
        mean_valuations=mean_vals, correlation_matrix=cov_matrix
    )
    
    # Create non-stationary environment
    change_points = [250, 500, 750]
    distributions = [
        {'mean': np.array([3.0, 4.0, 5.0]), 
         'cov': np.array([[1.5, 0.2, 0.1], [0.2, 1.5, 0.3], [0.1, 0.3, 1.5]])},
        {'mean': np.array([6.0, 7.0, 8.0]), 
         'cov': np.array([[2.0, 0.4, 0.2], [0.4, 2.0, 0.5], [0.2, 0.5, 2.0]])},
        {'mean': np.array([4.5, 5.5, 6.5]), 
         'cov': np.array([[1.8, 0.3, 0.15], [0.3, 1.8, 0.4], [0.15, 0.4, 1.8]])},
        {'mean': np.array([5.0, 6.0, 7.0]), 
         'cov': np.array([[2.2, 0.5, 0.25], [0.5, 2.2, 0.6], [0.25, 0.6, 2.2]])}
    ]
    
    non_stationary_env = NonStationaryEnvironment(
        n_products=n_products, prices=prices, T=T, B=B,
        change_points=change_points, distributions=distributions
    )
    
    # Create algorithms
    algorithms = {
        'Combinatorial UCB': CombinatorialUCB(prices, n_products, T, B),
        'Primal-Dual Multi': PrimalDualMultiProduct(prices, n_products, T, B)
    }
    
    return {
        'Stochastic Multi': stochastic_env,
        'Non-Stationary Multi': non_stationary_env
    }, algorithms

def create_requirement_5_experiment():
    """Create experiment for Requirement 5: Slightly non-stationary with sliding window"""
    
    prices = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    T = 1000
    B = 500
    n_products = 3
    
    # Create slightly non-stationary environment
    interval_length = 200
    n_intervals = 5
    
    env = SlightlyNonStationaryEnvironment(
        n_products=n_products, prices=prices, T=T, B=B,
        interval_length=interval_length, n_intervals=n_intervals
    )
    
    # Create algorithms
    algorithms = {
        'Combinatorial UCB (Sliding Window)': SlidingWindowCombinatorialUCB(
            prices, n_products, T, B, window_size=150
        ),
        'Primal-Dual Multi': PrimalDualMultiProduct(prices, n_products, T, B)
    }
    
    return {'Slightly Non-Stationary': env}, algorithms

def run_all_requirements():
    """Run all project requirements"""
    
    simulation = PricingSimulation()
    
    print("="*60)
    print("ONLINE LEARNING PRICING PROJECT - COMPLETE EVALUATION")
    print("="*60)
    
    # Requirement 1
    print("\n" + "="*40)
    print("REQUIREMENT 1: Single Product Stochastic")
    print("="*40)
    envs1, algs1 = create_requirement_1_experiment()
    results1 = simulation.run_comparison(envs1, algs1, n_runs=5)
    simulation.plot_results(results1, 1)
    
    # Requirement 2
    print("\n" + "="*40)
    print("REQUIREMENT 2: Multiple Products Stochastic")
    print("="*40)
    envs2, algs2 = create_requirement_2_experiment()
    results2 = simulation.run_comparison(envs2, algs2, n_runs=5)
    simulation.plot_results(results2, 2)
    
    # Requirement 3
    print("\n" + "="*40)
    print("REQUIREMENT 3: Best-of-Both-Worlds Single Product")
    print("="*40)
    envs3, algs3 = create_requirement_3_experiment()
    results3 = simulation.run_comparison(envs3, algs3, n_runs=5)
    simulation.plot_results(results3, 3)
    
    # Requirement 4
    print("\n" + "="*40)
    print("REQUIREMENT 4: Best-of-Both-Worlds Multiple Products")
    print("="*40)
    envs4, algs4 = create_requirement_4_experiment()
    results4 = simulation.run_comparison(envs4, algs4, n_runs=5)
    simulation.plot_results(results4, 4)
    
    # Requirement 5
    print("\n" + "="*40)
    print("REQUIREMENT 5: Slightly Non-Stationary with Sliding Window")
    print("="*40)
    envs5, algs5 = create_requirement_5_experiment()
    results5 = simulation.run_comparison(envs5, algs5, n_runs=5)
    simulation.plot_results(results5, 5)
    
    return {
        'req1': results1, 'req2': results2, 'req3': results3,
        'req4': results4, 'req5': results5
    } 