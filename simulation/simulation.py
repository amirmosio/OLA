import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for better performance
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
import seaborn as sns
import pandas as pd
from tqdm import tqdm
import csv
import os
from datetime import datetime
import copy

# Import from the new package structure
from environments import PricingEnvironment
from algorithms import PricingAlgorithm

class PricingSimulation:
    """Main simulation class for pricing experiments"""
    
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        self.results = {}
        
    def create_requirement_directories(self, requirement_num: int):
        """Create structured directories for a specific requirement"""
        base_dir = f"results/requirement_{requirement_num}"
        plots_dir = os.path.join(base_dir, "plots")
        data_dir = os.path.join(base_dir, "data")
        
        # Create directories if they don't exist
        os.makedirs(plots_dir, exist_ok=True)
        os.makedirs(data_dir, exist_ok=True)
        
        return {
            'base_dir': base_dir,
            'plots_dir': plots_dir,
            'data_dir': data_dir
        }

    def run_single_experiment(self, environment: PricingEnvironment, 
                            algorithm: PricingAlgorithm, 
                            algorithm_name: str) -> Dict:
        """Run a single experiment with given environment and algorithm"""
        # Create fresh copies to avoid state contamination
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
        
        with tqdm(total=total_experiments, desc="Running Experiments",
                 bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]") as pbar:
            
            for env_name, env in environments.items():
                print(f"\nRunning experiments on {env_name}...")
                results[env_name] = {}
                
                for alg_name, alg in algorithms.items():
                    print(f"  Running {alg_name}...")
                    
                    run_results = []
                    for run in range(n_runs):
                        pbar.set_description(f"{env_name[:15]} | {alg_name[:20]} | Run {run+1}/{n_runs}")
                        
                        # Create fresh copies with run-specific seeds for reproducibility
                        run_env_seed = hash((env_name, run)) % 2**31  # Ensure positive 32-bit int
                        run_alg_seed = hash((alg_name, run)) % 2**31  # Ensure positive 32-bit int
                        
                        # Create new instances with run-specific seeds
                        env_copy = copy.deepcopy(env)
                        alg_copy = copy.deepcopy(alg)
                        
                        # Set new seeds for this run
                        if hasattr(env_copy, 'rng'):
                            env_copy.rng = np.random.RandomState(run_env_seed)
                        if hasattr(alg_copy, 'rng'):
                            alg_copy.rng = np.random.RandomState(run_alg_seed)
                        
                        result = self.run_single_experiment(env_copy, alg_copy, alg_name)
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
        
        print(f"\nCompleted {total_experiments} experiments!")
        return results
    
    def save_results_to_csv(self, results: Dict, requirement_num: int):
        """Save experimental results to CSV files in structured directories"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create structured directories
        dirs = self.create_requirement_directories(requirement_num)
        
        # Save summary results
        summary_filename = os.path.join(dirs['data_dir'], f"requirement_{requirement_num}_summary_{timestamp}.csv")
        with open(summary_filename, 'w', newline='') as csvfile:
            fieldnames = ['Environment', 'Algorithm', 'Avg_Revenue', 'Std_Revenue', 'Avg_Regret', 'Avg_Rounds', 'Avg_Budget_Used']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for env_name, env_results in results.items():
                for alg_name, alg_results in env_results.items():
                    writer.writerow({
                        'Environment': env_name,
                        'Algorithm': alg_name,
                        'Avg_Revenue': f"{alg_results['avg_revenue']:.2f}",
                        'Std_Revenue': f"{alg_results['std_revenue']:.2f}",
                        'Avg_Regret': f"{alg_results['avg_regret']:.2f}",
                        'Avg_Rounds': f"{alg_results['avg_rounds']:.1f}",
                        'Avg_Budget_Used': f"{alg_results['avg_budget_used']:.1f}"
                    })
        
        print(f"Summary results saved to: {summary_filename}")
        
        # Save detailed results (individual runs)
        detailed_filename = os.path.join(dirs['data_dir'], f"requirement_{requirement_num}_detailed_{timestamp}.csv")
        with open(detailed_filename, 'w', newline='') as csvfile:
            fieldnames = ['Environment', 'Algorithm', 'Run', 'Total_Revenue', 'Rounds_Completed', 'Final_Budget']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for env_name, env_results in results.items():
                for alg_name, alg_results in env_results.items():
                    for i, run_result in enumerate(alg_results['all_runs']):
                        writer.writerow({
                            'Environment': env_name,
                            'Algorithm': alg_name,
                            'Run': i + 1,
                            'Total_Revenue': f"{run_result['total_revenue']:.2f}",
                            'Rounds_Completed': run_result['rounds_completed'],
                            'Final_Budget': run_result['final_budget']
                        })
        
        print(f"Detailed results saved to: {detailed_filename}")
        
        return {
            'summary_file': summary_filename,
            'detailed_file': detailed_filename
        }
    
    def plot_results(self, results: Dict, requirement_num: int):
        """Plot comparison results and save to structured directories as separate files"""
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        
        # Create structured directory
        dirs = self.create_requirement_directories(requirement_num)
        
        # Extract common data
        env_names = list(results.keys())
        alg_names = list(results[env_names[0]].keys())
        plot_files = []
        
        # Plot 1: Average Revenue by Algorithm and Environment
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(env_names))
        width = 0.8 / len(alg_names)
        
        for i, alg_name in enumerate(alg_names):
            revenues = [results[env][alg_name]['avg_revenue'] for env in env_names]
            errors = [results[env][alg_name]['std_revenue'] for env in env_names]
            
            ax1.bar(x + i * width, revenues, width, 
                   label=alg_name, alpha=0.8, yerr=errors, capsize=5)
        
        ax1.set_xlabel('Environment')
        ax1.set_ylabel('Average Revenue')
        ax1.set_title('Revenue Comparison Across Environments')
        ax1.set_xticks(x + width * (len(alg_names) - 1) / 2)
        ax1.set_xticklabels(env_names, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot1_filename = os.path.join(dirs['plots_dir'], f"requirement_{requirement_num}_revenue_comparison.png")
        plt.savefig(plot1_filename, dpi=300, bbox_inches='tight')
        plt.close()
        plot_files.append(plot1_filename)
        
        # Plot 2: Revenue over time for first environment
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        
        first_env = env_names[0]
        for alg_name in alg_names:
            # Get first run's revenue trajectory
            first_run = results[first_env][alg_name]['all_runs'][0]
            revenues = first_run['revenues']
            cumulative = np.cumsum(revenues)
            ax2.plot(cumulative, label=alg_name, alpha=0.8)
        
        ax2.set_xlabel('Round')
        ax2.set_ylabel('Cumulative Revenue')
        ax2.set_title(f'Revenue Growth Over Time - {first_env}')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot2_filename = os.path.join(dirs['plots_dir'], f"requirement_{requirement_num}_revenue_growth.png")
        plt.savefig(plot2_filename, dpi=300, bbox_inches='tight')
        plt.close()
        plot_files.append(plot2_filename)
        
        # Plot 3: Remaining budget through rounds
        if len(env_names) == 1:
            # Single environment - show in one plot
            fig3, ax3 = plt.subplots(figsize=(12, 6))
            
            env_name = env_names[0]
            for alg_name in alg_names:
                # Get first run's budget usage trajectory
                first_run = results[env_name][alg_name]['all_runs'][0]
                budget_usage = first_run['budget_usage']
                
                # Convert budget usage to remaining budget
                # budget_usage tracks cumulative budget used, initial budget is 500
                initial_budget = 500
                remaining_budget = [initial_budget - used for used in budget_usage]
                
                rounds = range(len(remaining_budget))
                ax3.plot(rounds, remaining_budget, label=alg_name, alpha=0.8, linewidth=2)
            
            ax3.set_xlabel('Round')
            ax3.set_ylabel('Remaining Budget')
            ax3.set_title(f'Remaining Budget Through Rounds - {env_name}')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            ax3.set_ylim(bottom=0)  # Ensure y-axis starts from 0
        else:
            # Multiple environments - show in subplots
            fig3, axes = plt.subplots(1, len(env_names), figsize=(6*len(env_names), 6))
            if len(env_names) == 2:
                axes = [axes[0], axes[1]]  # Ensure it's always a list
            
            for env_idx, env_name in enumerate(env_names):
                ax = axes[env_idx]
                
                for alg_name in alg_names:
                    # Get first run's budget usage trajectory
                    first_run = results[env_name][alg_name]['all_runs'][0]
                    budget_usage = first_run['budget_usage']
                    
                    # Convert budget usage to remaining budget
                    initial_budget = 500
                    remaining_budget = [initial_budget - used for used in budget_usage]
                    
                    rounds = range(len(remaining_budget))
                    ax.plot(rounds, remaining_budget, label=alg_name, alpha=0.8, linewidth=2)
                
                ax.set_xlabel('Round')
                ax.set_ylabel('Remaining Budget')
                ax.set_title(f'Remaining Budget - {env_name}')
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.set_ylim(bottom=0)
        
        plt.tight_layout()
        plot3_filename = os.path.join(dirs['plots_dir'], f"requirement_{requirement_num}_budget_management.png")
        plt.savefig(plot3_filename, dpi=300, bbox_inches='tight')
        plt.close()
        plot_files.append(plot3_filename)
        
        # Plot 4: Revenue per round efficiency
        fig4, ax4 = plt.subplots(figsize=(10, 6))
        
        efficiency_data = []
        for env_name in env_names:
            for alg_name in alg_names:
                avg_revenue = results[env_name][alg_name]['avg_revenue']
                avg_rounds = results[env_name][alg_name]['avg_rounds']
                efficiency = avg_revenue / max(avg_rounds, 1)
                efficiency_data.append({
                    'Environment': env_name,
                    'Algorithm': alg_name,
                    'Revenue_per_Round': efficiency
                })
        
        efficiency_df = pd.DataFrame(efficiency_data)
        pivot_df = efficiency_df.pivot(index='Environment', columns='Algorithm', values='Revenue_per_Round')
        
        # Create heatmap
        sns.heatmap(pivot_df, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax4)
        ax4.set_title('Revenue per Round Efficiency')
        
        plt.tight_layout()
        plot4_filename = os.path.join(dirs['plots_dir'], f"requirement_{requirement_num}_efficiency_heatmap.png")
        plt.savefig(plot4_filename, dpi=300, bbox_inches='tight')
        plt.close()
        plot_files.append(plot4_filename)
        
        print(f"Plots saved to separate files:")
        for plot_file in plot_files:
            print(f"  - {plot_file}")
        
        return plot_files 