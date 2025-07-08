import numpy as np
from .base import PricingEnvironment
from typing import List, Dict
import random

class NonStationaryEnvironment(PricingEnvironment):
    """Non-stationary environment with changing distributions"""
    
    def __init__(self, n_products: int, prices: List[float], T: int, B: int,
                 change_points: List[int], distributions: List[Dict]):
        super().__init__(n_products, prices, T, B)
        self.change_points = sorted(change_points)
        self.distributions = distributions
        self.current_dist_idx = 0
    
    def get_current_distribution(self, round_t: int) -> Dict:
        """Get the distribution parameters for current round"""
        for i, cp in enumerate(self.change_points):
            if round_t < cp:
                return self.distributions[i]
        return self.distributions[-1]
    
    def generate_valuations(self, round_t: int) -> np.ndarray:
        """Generate valuations based on current distribution"""
        dist_params = self.get_current_distribution(round_t)
        
        if self.n_products == 1:
            valuation = np.random.normal(dist_params['mean'], dist_params['std'])
            return np.array([max(0, valuation)])
        else:
            valuations = np.random.multivariate_normal(
                dist_params['mean'], dist_params['cov']
            )
            return np.maximum(0, valuations)

class SlightlyNonStationaryEnvironment(PricingEnvironment):
    """Environment with intervals of fixed distributions"""
    
    def __init__(self, n_products: int, prices: List[float], T: int, B: int,
                 interval_length: int, n_intervals: int):
        super().__init__(n_products, prices, T, B)
        self.interval_length = interval_length
        self.n_intervals = n_intervals
        
        # Generate different distributions for each interval
        self.interval_distributions = []
        for _ in range(n_intervals):
            if n_products == 1:
                dist = {
                    'mean': np.random.uniform(2, 8),
                    'std': np.random.uniform(0.5, 2.0)
                }
            else:
                mean = np.random.uniform(2, 8, n_products)
                A = np.random.randn(n_products, n_products)
                cov = np.dot(A, A.T) + np.eye(n_products) * 0.5
                dist = {'mean': mean, 'cov': cov}
            self.interval_distributions.append(dist)
    
    def get_current_interval(self, round_t: int) -> int:
        """Get current interval index"""
        return min(round_t // self.interval_length, self.n_intervals - 1)
    
    def generate_valuations(self, round_t: int) -> np.ndarray:
        """Generate valuations based on current interval"""
        interval_idx = self.get_current_interval(round_t)
        dist_params = self.interval_distributions[interval_idx]
        
        if self.n_products == 1:
            valuation = np.random.normal(dist_params['mean'], dist_params['std'])
            return np.array([max(0, valuation)])
        else:
            valuations = np.random.multivariate_normal(
                dist_params['mean'], dist_params['cov']
            )
            return np.maximum(0, valuations) 