import numpy as np
from typing import Dict, Tuple

# Import from the new package structure
from environments import (
    StochasticSingleProductEnvironment,
    StochasticMultiProductEnvironment,
    HighlyNonStationaryEnvironment,
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

def create_requirement_1_experiment(env_seed: int = 1, alg_seed: int = 100):
    """Create experiment for Requirement 1: Single product stochastic"""
    
    # Define common parameters
    prices = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    T = 1000
    B = 500
    
    # Create environment with seed
    env = StochasticSingleProductEnvironment(
        prices=prices, T=T, B=B, 
        valuation_mean=5.0, valuation_std=2.0, seed=env_seed
    )
    
    # Create algorithms with seeds
    algorithms = {
        'UCB1 (No Constraint)': UCB1SingleProduct(prices, T, B, seed=alg_seed),
        'UCB with Inventory': UCBWithInventoryConstraintSingleProduct(prices, T, B, seed=alg_seed + 1)
    }
    
    return {'Stochastic Single Product': env}, algorithms

def create_requirement_2_experiment(env_seed: int = 2, alg_seed: int = 200):
    """Create experiment for Requirement 2: Multiple products stochastic"""
    
    prices = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    T = 1000
    B = 500
    n_products = 3
    
    # Create environment with seed
    mean_vals = np.array([4.0, 5.0, 6.0])
    cov_matrix = np.array([[2.0, 0.5, 0.3],
                          [0.5, 2.0, 0.4],
                          [0.3, 0.4, 2.0]])
    
    env = StochasticMultiProductEnvironment(
        n_products=n_products, prices=prices, T=T, B=B,
        mean_valuations=mean_vals, correlation_matrix=cov_matrix, seed=env_seed
    )
    
    # Create algorithms with seeds
    algorithms = {
        'Combinatorial UCB': CombinatorialUCB(prices, n_products, T, B, seed=alg_seed)
    }
    
    return {'Stochastic Multi Product': env}, algorithms

def create_requirement_3_experiment(env_seed: int = 3, alg_seed: int = 300):
    """Create experiment for Requirement 3: Best-of-both-worlds single product"""
    
    prices = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    T = 1000
    B = 500
    
    # Create environments with seeds
    stochastic_env = StochasticSingleProductEnvironment(
        prices=prices, T=T, B=B, valuation_mean=5.0, valuation_std=2.0, seed=env_seed
    )
    
    # Create highly non-stationary environment with quickly changing distributions
    non_stationary_env = HighlyNonStationaryEnvironment(
        n_products=1, prices=prices, T=T, B=B,
        base_mean=5.0, base_std=2.0, change_rate=0.1, noise_scale=1.5, seed=env_seed + 1
    )
    
    # Create algorithms with seeds
    algorithms = {
        'UCB with Inventory': UCBWithInventoryConstraintSingleProduct(prices, T, B, seed=alg_seed),
        'Primal-Dual': PrimalDualSingleProduct(prices, T, B, seed=alg_seed + 1)
    }
    
    return {
        'Stochastic': stochastic_env,
        'Non-Stationary': non_stationary_env
    }, algorithms

def create_requirement_4_experiment(env_seed: int = 4, alg_seed: int = 400):
    """Create experiment for Requirement 4: Best-of-both-worlds multiple products"""
    
    prices = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    T = 1000
    B = 500
    n_products = 3
    
    # Create stochastic environment with seed
    mean_vals = np.array([4.0, 5.0, 6.0])
    cov_matrix = np.array([[2.0, 0.5, 0.3],
                          [0.5, 2.0, 0.4],
                          [0.3, 0.4, 2.0]])
    
    stochastic_env = StochasticMultiProductEnvironment(
        n_products=n_products, prices=prices, T=T, B=B,
        mean_valuations=mean_vals, correlation_matrix=cov_matrix, seed=env_seed
    )
    
    # Create highly non-stationary environment with quickly changing distributions
    non_stationary_env = HighlyNonStationaryEnvironment(
        n_products=n_products, prices=prices, T=T, B=B,
        base_mean=5.0, base_std=2.0, change_rate=0.08, noise_scale=1.2, seed=env_seed + 1
    )
    
    # Create algorithms with seeds
    algorithms = {
        'Combinatorial UCB': CombinatorialUCB(prices, n_products, T, B, seed=alg_seed),
        'Primal-Dual Multi': PrimalDualMultiProduct(prices, n_products, T, B, seed=alg_seed + 1)
    }
    
    return {
        'Stochastic Multi': stochastic_env,
        'Non-Stationary Multi': non_stationary_env
    }, algorithms

def create_requirement_5_experiment(env_seed: int = 5, alg_seed: int = 500):
    """Create experiment for Requirement 5: Slightly non-stationary with sliding window"""
    
    prices = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    T = 1000
    B = 500
    n_products = 3
    
    # Create slightly non-stationary environment with seed
    interval_length = 200
    n_intervals = 5
    
    env = SlightlyNonStationaryEnvironment(
        n_products=n_products, prices=prices, T=T, B=B,
        interval_length=interval_length, n_intervals=n_intervals, seed=env_seed
    )
    
    # Create algorithms with seeds
    algorithms = {
        'Combinatorial UCB (Sliding Window)': SlidingWindowCombinatorialUCB(
            prices, n_products, T, B, window_size=150, seed=alg_seed
        ),
        'Primal-Dual Multi': PrimalDualMultiProduct(prices, n_products, T, B, seed=alg_seed + 1)
    }
    
    return {'Slightly Non-Stationary': env}, algorithms

def run_all_requirements(base_seed: int = 42):
    """Run all project requirements with seed-based reproducibility"""
    
    simulation = PricingSimulation(seed=base_seed)
    
    print("="*60)
    print("ONLINE LEARNING PRICING PROJECT - COMPLETE EVALUATION")
    print("="*60)
    
    # Requirement 1
    print("\n" + "="*40)
    print("REQUIREMENT 1: Single Product Stochastic")
    print("="*40)
    envs1, algs1 = create_requirement_1_experiment(env_seed=base_seed + 1, alg_seed=base_seed + 100)
    results1 = simulation.run_comparison(envs1, algs1, n_runs=5)
    simulation.plot_results(results1, 1)
    
    # Requirement 2
    print("\n" + "="*40)
    print("REQUIREMENT 2: Multiple Products Stochastic")
    print("="*40)
    envs2, algs2 = create_requirement_2_experiment(env_seed=base_seed + 2, alg_seed=base_seed + 200)
    results2 = simulation.run_comparison(envs2, algs2, n_runs=5)
    simulation.plot_results(results2, 2)
    
    # Requirement 3
    print("\n" + "="*40)
    print("REQUIREMENT 3: Best-of-Both-Worlds Single Product")
    print("="*40)
    envs3, algs3 = create_requirement_3_experiment(env_seed=base_seed + 3, alg_seed=base_seed + 300)
    results3 = simulation.run_comparison(envs3, algs3, n_runs=5)
    simulation.plot_results(results3, 3)
    
    # Requirement 4
    print("\n" + "="*40)
    print("REQUIREMENT 4: Best-of-Both-Worlds Multiple Products")
    print("="*40)
    envs4, algs4 = create_requirement_4_experiment(env_seed=base_seed + 4, alg_seed=base_seed + 400)
    results4 = simulation.run_comparison(envs4, algs4, n_runs=5)
    simulation.plot_results(results4, 4)
    
    # Requirement 5
    print("\n" + "="*40)
    print("REQUIREMENT 5: Slightly Non-Stationary with Sliding Window")
    print("="*40)
    envs5, algs5 = create_requirement_5_experiment(env_seed=base_seed + 5, alg_seed=base_seed + 500)
    results5 = simulation.run_comparison(envs5, algs5, n_runs=5)
    simulation.plot_results(results5, 5)
    
    return {
        'req1': results1, 'req2': results2, 'req3': results3,
        'req4': results4, 'req5': results5
    } 