import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
import seaborn as sns
import pandas as pd
from tqdm import tqdm
import csv
import os
from datetime import datetime
from environment import *
from algorithms import *

class PricingSimulation:
    """Main simulation class for pricing experiments"""
    
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        self.results = {}
        
    def run_single_experiment(self, environment: PricingEnvironment, 
                            algorithm: PricingAlgorithm, 
                            algorithm_name: str) -> Dict:
        """Run a single experiment with given environment and algorithm"""
        # Create fresh copies to avoid state contamination
        import copy
        environment = copy.deepcopy(environment)
        algorithm = copy.deepcopy(algorithm)
        environment.reset()
        algorithm.reset()
        
        total_revenue = 0
        revenues = []
        regrets = []
        budget_usage = []
        
        for t in range(environment.T):
            if environment.remaining_budget <= 0:
                break
                
            # Algorithm selects prices
            prices_set = algorithm.select_prices(environment.remaining_budget)
            
            # Environment responds
            result = environment.step(prices_set)
            
            # Update algorithm
            algorithm.update(result['prices_set'], result['valuations'], 
                           result['purchases'], result['revenue'])
            
            # Track metrics
            total_revenue += result['revenue']
            revenues.append(result['revenue'])
            budget_usage.append(environment.initial_B - environment.remaining_budget)
            
            # Compute regret (simplified - against oracle that knows all future valuations)
            # For demonstration, we'll compute pseudo-regret
            regrets.append(0)  # Placeholder - would need oracle computation
        
        return {
            'algorithm': algorithm_name,
            'total_revenue': total_revenue,
            'revenues': revenues,
            'regrets': regrets,
            'budget_usage': budget_usage,
            'rounds_completed': len(revenues),
            'final_budget': environment.remaining_budget
        }
    
    def run_comparison(self, environments: Dict[str, PricingEnvironment],
                      algorithms: Dict[str, PricingAlgorithm],
                      n_runs: int = 10) -> Dict:
        """Run comparison between multiple algorithms on multiple environments"""
        
        results = {}
        
        # Calculate total experiments for progress bar
        total_experiments = len(environments) * len(algorithms) * n_runs
        
        with tqdm(total=total_experiments, desc="🧪 Running Experiments", 
                 bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]") as pbar:
            
            for env_name, env in environments.items():
                print(f"\n🌍 Running experiments on {env_name}...")
                results[env_name] = {}
                
                for alg_name, alg in algorithms.items():
                    print(f"  🤖 Running {alg_name}...")
                    
                    run_results = []
                    for run in range(n_runs):
                        pbar.set_description(f"🧪 {env_name[:15]} | {alg_name[:20]} | Run {run+1}/{n_runs}")
                        result = self.run_single_experiment(env, alg, alg_name)
                        run_results.append(result)
                        pbar.update(1)
                    
                    # Aggregate results
                    avg_revenue = np.mean([r['total_revenue'] for r in run_results])
                    std_revenue = np.std([r['total_revenue'] for r in run_results])
                    avg_regret = np.mean([np.sum(r['regrets']) for r in run_results])
                    avg_rounds = np.mean([r['rounds_completed'] for r in run_results])
                    avg_budget_used = np.mean([r['final_budget'] for r in run_results])
                    
                    results[env_name][alg_name] = {
                        'avg_revenue': avg_revenue,
                        'std_revenue': std_revenue,
                        'avg_regret': avg_regret,
                        'avg_rounds': avg_rounds,
                        'avg_budget_used': avg_budget_used,
                        'all_runs': run_results
                    }
        
        print(f"\n✅ Completed {total_experiments} experiments!")
        return results
    
    def save_results_to_csv(self, results: Dict, requirement_name: str = "experiment"):
        """Save experimental results to CSV files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Ensure results directory exists
        os.makedirs('results', exist_ok=True)
        
        # 1. Save Summary Results
        summary_data = []
        for env_name, env_results in results.items():
            for alg_name, alg_results in env_results.items():
                summary_data.append({
                    'requirement': requirement_name,
                    'environment': env_name,
                    'algorithm': alg_name,
                    'avg_revenue': alg_results['avg_revenue'],
                    'std_revenue': alg_results['std_revenue'],
                    'avg_regret': alg_results['avg_regret'],
                    'avg_rounds': alg_results['avg_rounds'],
                    'avg_budget_used': alg_results['avg_budget_used'],
                    'timestamp': timestamp
                })
        
        summary_df = pd.DataFrame(summary_data)
        summary_filename = f"results/{requirement_name}_summary_{timestamp}.csv"
        summary_df.to_csv(summary_filename, index=False)
        print(f"📊 Summary results saved to: {summary_filename}")
        
        # 2. Save Detailed Run Results
        detailed_data = []
        for env_name, env_results in results.items():
            for alg_name, alg_results in env_results.items():
                for run_idx, run_result in enumerate(alg_results['all_runs']):
                    detailed_data.append({
                        'requirement': requirement_name,
                        'environment': env_name,
                        'algorithm': alg_name,
                        'run_number': run_idx + 1,
                        'total_revenue': run_result['total_revenue'],
                        'rounds_completed': run_result['rounds_completed'],
                        'final_budget': run_result['final_budget'],
                        'budget_used': run_result.get('initial_budget', 500) - run_result['final_budget'],
                        'avg_revenue_per_round': run_result['total_revenue'] / max(run_result['rounds_completed'], 1),
                        'timestamp': timestamp
                    })
        
        detailed_df = pd.DataFrame(detailed_data)
        detailed_filename = f"results/{requirement_name}_detailed_{timestamp}.csv"
        detailed_df.to_csv(detailed_filename, index=False)
        print(f"📋 Detailed results saved to: {detailed_filename}")
        
        return {
            'summary_file': summary_filename,
            'detailed_file': detailed_filename
        }
    
    def plot_results(self, results: Dict, save_path: Optional[str] = None):
        """Plot comparison results"""
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Average Revenue by Algorithm and Environment
        env_names = list(results.keys())
        alg_names = list(results[env_names[0]].keys())
        
        x = np.arange(len(env_names))
        width = 0.8 / len(alg_names)
        
        for i, alg_name in enumerate(alg_names):
            revenues = [results[env][alg_name]['avg_revenue'] for env in env_names]
            errors = [results[env][alg_name]['std_revenue'] for env in env_names]
            
            axes[0, 0].bar(x + i * width, revenues, width, 
                          label=alg_name, alpha=0.8, yerr=errors, capsize=5)
        
        axes[0, 0].set_xlabel('Environment')
        axes[0, 0].set_ylabel('Average Revenue')
        axes[0, 0].set_title('Revenue Comparison Across Environments')
        axes[0, 0].set_xticks(x + width * (len(alg_names) - 1) / 2)
        axes[0, 0].set_xticklabels(env_names, rotation=45)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Revenue over time for first environment
        first_env = env_names[0]
        for alg_name in alg_names:
            sample_run = results[first_env][alg_name]['all_runs'][0]
            cumulative_revenue = np.cumsum(sample_run['revenues'])
            axes[0, 1].plot(cumulative_revenue, label=alg_name, alpha=0.8)
        
        axes[0, 1].set_xlabel('Round')
        axes[0, 1].set_ylabel('Cumulative Revenue')
        axes[0, 1].set_title(f'Revenue Over Time - {first_env}')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Budget usage over time
        for alg_name in alg_names:
            sample_run = results[first_env][alg_name]['all_runs'][0]
            axes[1, 0].plot(sample_run['budget_usage'], label=alg_name, alpha=0.8)
        
        axes[1, 0].set_xlabel('Round')
        axes[1, 0].set_ylabel('Budget Used')
        axes[1, 0].set_title(f'Budget Usage Over Time - {first_env}')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Algorithm performance heatmap
        performance_matrix = np.zeros((len(alg_names), len(env_names)))
        for i, alg_name in enumerate(alg_names):
            for j, env_name in enumerate(env_names):
                performance_matrix[i, j] = results[env_name][alg_name]['avg_revenue']
        
        im = axes[1, 1].imshow(performance_matrix, cmap='viridis', aspect='auto')
        axes[1, 1].set_xticks(range(len(env_names)))
        axes[1, 1].set_yticks(range(len(alg_names)))
        axes[1, 1].set_xticklabels(env_names, rotation=45)
        axes[1, 1].set_yticklabels(alg_names)
        axes[1, 1].set_title('Performance Heatmap (Revenue)')
        
        # Add colorbar
        plt.colorbar(im, ax=axes[1, 1])
        
        # Add values to heatmap
        for i in range(len(alg_names)):
            for j in range(len(env_names)):
                axes[1, 1].text(j, i, f'{performance_matrix[i, j]:.1f}',
                               ha='center', va='center', color='white', fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            # Ensure results directory exists
            os.makedirs('results', exist_ok=True)
            # Save to results folder
            results_path = f"results/{save_path}"
            plt.savefig(results_path, dpi=300, bbox_inches='tight')
        
        plt.show()

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
    
    simulation = PricingSimulation(seed=42)
    
    print("="*60)
    print("ONLINE LEARNING PRICING PROJECT - COMPLETE EVALUATION")
    print("="*60)
    
    # Requirement 1
    print("\n" + "="*40)
    print("REQUIREMENT 1: Single Product Stochastic")
    print("="*40)
    envs1, algs1 = create_requirement_1_experiment()
    results1 = simulation.run_comparison(envs1, algs1, n_runs=5)
    simulation.plot_results(results1, 'requirement_1_results.png')
    
    # Requirement 2
    print("\n" + "="*40)
    print("REQUIREMENT 2: Multiple Products Stochastic")
    print("="*40)
    envs2, algs2 = create_requirement_2_experiment()
    results2 = simulation.run_comparison(envs2, algs2, n_runs=5)
    simulation.plot_results(results2, 'requirement_2_results.png')
    
    # Requirement 3
    print("\n" + "="*40)
    print("REQUIREMENT 3: Best-of-Both-Worlds Single Product")
    print("="*40)
    envs3, algs3 = create_requirement_3_experiment()
    results3 = simulation.run_comparison(envs3, algs3, n_runs=5)
    simulation.plot_results(results3, 'requirement_3_results.png')
    
    # Requirement 4
    print("\n" + "="*40)
    print("REQUIREMENT 4: Best-of-Both-Worlds Multiple Products")
    print("="*40)
    envs4, algs4 = create_requirement_4_experiment()
    results4 = simulation.run_comparison(envs4, algs4, n_runs=5)
    simulation.plot_results(results4, 'requirement_4_results.png')
    
    # Requirement 5
    print("\n" + "="*40)
    print("REQUIREMENT 5: Slightly Non-Stationary with Sliding Window")
    print("="*40)
    envs5, algs5 = create_requirement_5_experiment()
    results5 = simulation.run_comparison(envs5, algs5, n_runs=5)
    simulation.plot_results(results5, 'requirement_5_results.png')
    
    return {
        'req1': results1, 'req2': results2, 'req3': results3,
        'req4': results4, 'req5': results5
    }
