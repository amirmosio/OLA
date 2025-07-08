#!/usr/bin/env python3
"""
Online Learning Applications - Pricing Project
===============================================

Complete implementation of all 5 project requirements:
1. Single product stochastic environment with UCB algorithms
2. Multiple products stochastic environment with Combinatorial UCB
3. Best-of-both-worlds single product with primal-dual methods
4. Best-of-both-worlds multiple products with primal-dual methods  
5. Slightly non-stationary environments with sliding window techniques

Usage:
    python main.py --requirement [1,2,3,4,5,all] --runs [num_runs] --save-plots
"""

import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
from simulation import *

def print_project_header():
    """Print project information header"""
    print("="*80)
    print("ONLINE LEARNING APPLICATIONS - PRICING PROJECT")
    print("="*80)
    print("Complete implementation addressing all 5 requirements:")
    print("1. Single Product + Stochastic: UCB1 vs UCB with Inventory Constraint")
    print("2. Multiple Products + Stochastic: Combinatorial UCB")
    print("3. Best-of-Both-Worlds Single Product: Primal-Dual Methods")
    print("4. Best-of-Both-Worlds Multiple Products: Primal-Dual Methods")
    print("5. Slightly Non-Stationary: Sliding Window vs Primal-Dual")
    print("="*80)

def run_requirement_1(n_runs=5, save_plots=False, save_csv=False):
    """Run Requirement 1: Single product stochastic"""
    print("\nREQUIREMENT 1: Single Product Stochastic Environment")
    print("-" * 60)
    
    simulation = PricingSimulation()
    envs, algs = create_requirement_1_experiment()
    
    print("Algorithms being tested:")
    for alg_name in algs.keys():
        print(f"   • {alg_name}")
    
    print(f"Running {n_runs} experiments...")
    results = simulation.run_comparison(envs, algs, n_runs=n_runs)
    
    # Print summary
    print("\nRESULTS SUMMARY:")
    for env_name, env_results in results.items():
        print(f"\n{env_name}:")
        for alg_name, alg_results in env_results.items():
            print(f"  {alg_name:25}: Revenue = {alg_results['avg_revenue']:.1f} ± {alg_results['std_revenue']:.1f}")
    
    if save_plots:
        simulation.plot_results(results, 1)
        print("Plots saved to 'results/requirement_1/plots/'")
    
    if save_csv:
        simulation.save_results_to_csv(results, 1)
        print("Data saved to 'results/requirement_1/data/'")
    
    return results

def run_requirement_2(n_runs=5, save_plots=False, save_csv=False):
    """Run Requirement 2: Multiple products stochastic"""
    print("\nREQUIREMENT 2: Multiple Products Stochastic Environment")
    print("-" * 60)
    
    simulation = PricingSimulation()
    envs, algs = create_requirement_2_experiment()
    
    print("Algorithms being tested:")
    for alg_name in algs.keys():
        print(f"   • {alg_name}")
    
    print(f"Running {n_runs} experiments...")
    results = simulation.run_comparison(envs, algs, n_runs=n_runs)
    
    # Print summary
    print("\nRESULTS SUMMARY:")
    for env_name, env_results in results.items():
        print(f"\n{env_name}:")
        for alg_name, alg_results in env_results.items():
            print(f"  {alg_name:25}: Revenue = {alg_results['avg_revenue']:.1f} ± {alg_results['std_revenue']:.1f}")
    
    if save_plots:
        simulation.plot_results(results, 2)
        print("Plots saved to 'results/requirement_2/plots/'")
    
    if save_csv:
        simulation.save_results_to_csv(results, 2)
        print("Data saved to 'results/requirement_2/data/'")
    
    return results

def run_requirement_3(n_runs=5, save_plots=False, save_csv=False):
    """Run Requirement 3: Best-of-both-worlds single product"""
    print("\nREQUIREMENT 3: Best-of-Both-Worlds Single Product")
    print("-" * 60)
    
    simulation = PricingSimulation()
    envs, algs = create_requirement_3_experiment()
    
    print("Algorithms being tested:")
    for alg_name in algs.keys():
        print(f"   • {alg_name}")
    
    print("Environments:")
    for env_name in envs.keys():
        print(f"   • {env_name}")
    
    print(f"Running {n_runs} experiments...")
    results = simulation.run_comparison(envs, algs, n_runs=n_runs)
    
    # Print summary
    print("\nRESULTS SUMMARY:")
    for env_name, env_results in results.items():
        print(f"\n{env_name}:")
        for alg_name, alg_results in env_results.items():
            print(f"  {alg_name:25}: Revenue = {alg_results['avg_revenue']:.1f} ± {alg_results['std_revenue']:.1f}")
    
    if save_plots:
        simulation.plot_results(results, 3)
        print("Plots saved to 'results/requirement_3/plots/'")
    
    if save_csv:
        simulation.save_results_to_csv(results, 3)
        print("Data saved to 'results/requirement_3/data/'")
    
    return results

def run_requirement_4(n_runs=5, save_plots=False, save_csv=False):
    """Run Requirement 4: Best-of-both-worlds multiple products"""
    print("\nREQUIREMENT 4: Best-of-Both-Worlds Multiple Products")
    print("-" * 60)
    
    simulation = PricingSimulation()
    envs, algs = create_requirement_4_experiment()
    
    print("Algorithms being tested:")
    for alg_name in algs.keys():
        print(f"   • {alg_name}")
    
    print("Environments:")
    for env_name in envs.keys():
        print(f"   • {env_name}")
    
    print(f"Running {n_runs} experiments...")
    results = simulation.run_comparison(envs, algs, n_runs=n_runs)
    
    # Print summary
    print("\nRESULTS SUMMARY:")
    for env_name, env_results in results.items():
        print(f"\n{env_name}:")
        for alg_name, alg_results in env_results.items():
            print(f"  {alg_name:25}: Revenue = {alg_results['avg_revenue']:.1f} ± {alg_results['std_revenue']:.1f}")
    
    if save_plots:
        simulation.plot_results(results, 4)
        print("Plots saved to 'results/requirement_4/plots/'")
    
    if save_csv:
        simulation.save_results_to_csv(results, 4)
        print("Data saved to 'results/requirement_4/data/'")
    
    return results

def run_requirement_5(n_runs=5, save_plots=False, save_csv=False):
    """Run Requirement 5: Slightly non-stationary with sliding window"""
    print("\nREQUIREMENT 5: Slightly Non-Stationary with Sliding Window")
    print("-" * 60)
    
    simulation = PricingSimulation()
    envs, algs = create_requirement_5_experiment()
    
    print("Algorithms being tested:")
    for alg_name in algs.keys():
        print(f"   • {alg_name}")
    
    print("Environment: Slightly Non-Stationary (intervals with different distributions)")
    
    print(f"Running {n_runs} experiments...")
    results = simulation.run_comparison(envs, algs, n_runs=n_runs)
    
    # Print summary
    print("\nRESULTS SUMMARY:")
    for env_name, env_results in results.items():
        print(f"\n{env_name}:")
        for alg_name, alg_results in env_results.items():
            print(f"  {alg_name:25}: Revenue = {alg_results['avg_revenue']:.1f} ± {alg_results['std_revenue']:.1f}")
    
    if save_plots:
        simulation.plot_results(results, 5)
        print("Plots saved to 'results/requirement_5/plots/'")
    
    if save_csv:
        simulation.save_results_to_csv(results, 5)
        print("Data saved to 'results/requirement_5/data/'")
    
    return results

def run_all_requirements(n_runs=5, save_plots=False, save_csv=False):
    """Run all project requirements"""
    print_project_header()
    
    all_results = {}
    

    all_results['req1'] = run_requirement_1(n_runs, save_plots, save_csv)
    all_results['req2'] = run_requirement_2(n_runs, save_plots, save_csv)
    all_results['req3'] = run_requirement_3(n_runs, save_plots, save_csv)
    all_results['req4'] = run_requirement_4(n_runs, save_plots, save_csv)
    all_results['req5'] = run_requirement_5(n_runs, save_plots, save_csv)
    
    print("\n" + "="*80)
    print("ALL REQUIREMENTS COMPLETED SUCCESSFULLY!")
    print("="*80)
    
    # Summary of all requirements
    print("\nCOMPLETE PROJECT SUMMARY:")
    print("-" * 40)
    
    req_names = [
        "Single Product Stochastic",
        "Multiple Products Stochastic", 
        "Best-of-Both-Worlds Single",
        "Best-of-Both-Worlds Multi",
        "Slightly Non-Stationary"
    ]
    
    for i, (req_key, req_name) in enumerate(zip(all_results.keys(), req_names), 1):
        print(f"\n{i}. {req_name}:")
        req_results = all_results[req_key]
        for env_name, env_results in req_results.items():
            best_alg = max(env_results.items(), key=lambda x: x[1]['avg_revenue'])
            print(f"   Best on {env_name}: {best_alg[0]} ({best_alg[1]['avg_revenue']:.1f} revenue)")
    
    if save_plots:
        print(f"\nAll plots saved to 'results/requirement_X/plots/' directories")
    
    if save_csv:
        print(f"\nAll numerical results saved to 'results/requirement_X/data/' directories")
        

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Online Learning Pricing Project - Complete Implementation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --requirement 1                         # Run only requirement 1
  python main.py --requirement all --runs 10             # Run all requirements with 10 runs each
  python main.py --requirement 3 --save-plots            # Run requirement 3 and save plots
  python main.py --requirement all --save-csv            # Run all and save numerical results to CSV
  python main.py --requirement 1 --save-plots --save-csv # Run requirement 1 with plots and CSV
        """
    )
    
    parser.add_argument(
        '--requirement', '-r',
        choices=['1', '2', '3', '4', '5', 'all'],
        default='all',
        help='Which requirement to run (default: all)'
    )
    
    parser.add_argument(
        '--runs', '-n',
        type=int,
        default=5,
        help='Number of experimental runs (default: 5)'
    )
    
    parser.add_argument(
        '--save-plots', '-s',
        action='store_true',
        help='Save plots to PNG files',
        default=True
    )
    
    parser.add_argument(
        '--save-csv', '-c',
        action='store_true',
        help='Save numerical results to CSV files',
        default=True
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Run the specified requirement(s)
    if args.requirement == 'all':
        results = run_all_requirements(args.runs, args.save_plots, args.save_csv)
    elif args.requirement == '1':
        print_project_header()
        results = run_requirement_1(args.runs, args.save_plots, args.save_csv)
    elif args.requirement == '2':
        print_project_header()
        results = run_requirement_2(args.runs, args.save_plots, args.save_csv)
    elif args.requirement == '3':
        print_project_header()
        results = run_requirement_3(args.runs, args.save_plots, args.save_csv)
    elif args.requirement == '4':
        print_project_header()
        results = run_requirement_4(args.runs, args.save_plots, args.save_csv)
    elif args.requirement == '5':
        print_project_header()
        results = run_requirement_5(args.runs, args.save_plots, args.save_csv)
    
    if results is not None:
        print(f"\nExperiment completed with {args.runs} runs per algorithm.")
        if args.save_plots:
            print("Plots have been saved to structured 'results/requirement_X/plots/' directories.")
        if args.save_csv:
            print("Numerical results have been saved to structured 'results/requirement_X/data/' directories.")
        print("Check the results above for detailed performance analysis.")

if __name__ == "__main__":
    main()
