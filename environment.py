import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional
import random

class PricingEnvironment(ABC):
    """Base class for pricing environments"""
    
    def __init__(self, n_products: int, prices: List[float], T: int, B: int):
        self.n_products = n_products
        self.prices = np.array(prices)
        self.T = T  # Number of rounds
        self.remaining_budget = B  # Production capacity
        self.initial_B = B
        self.current_round = 0
        self.history = []
        
    @abstractmethod
    def generate_valuations(self, round_t: int) -> np.ndarray:
        """Generate customer valuations for round t"""
        pass
    
    def simulate_purchase(self, prices_set: np.ndarray, valuations: np.ndarray) -> Tuple[np.ndarray, float, int]:
        """
        Simulate customer purchase decision
        Returns: (purchases, revenue, units_sold)
        """
        # Only consider products that are being sold (finite prices)
        valid_products = np.isfinite(prices_set)
        purchases = np.zeros_like(prices_set)
        
        # Customer buys products where price <= valuation AND product is being sold
        purchases[valid_products] = (prices_set[valid_products] <= valuations[valid_products]).astype(int)
        
        revenue = np.sum(purchases * np.where(np.isfinite(prices_set), prices_set, 0))
        units_sold = np.sum(purchases)
        return purchases, revenue, units_sold
    
    def step(self, prices_set: np.ndarray) -> Dict:
        """Execute one round of interaction"""
        if self.current_round >= self.T:
            raise ValueError("Environment exhausted")
            
        valuations = self.generate_valuations(self.current_round)
        purchases, revenue, units_sold = self.simulate_purchase(prices_set, valuations)
        
        # Update budget
        self.remaining_budget -= units_sold
        
        result = {
            'round': self.current_round,
            'valuations': valuations,
            'prices_set': prices_set,
            'purchases': purchases,
            'revenue': revenue,
            'units_sold': units_sold,
            'remaining_budget': self.remaining_budget
        }
        
        self.history.append(result)
        self.current_round += 1
        
        return result
    
    def reset(self):
        """Reset environment to initial state"""
        self.remaining_budget = self.initial_B
        self.current_round = 0
        self.history = []

class StochasticSingleProductEnvironment(PricingEnvironment):
    """Stochastic environment for single product"""
    
    def __init__(self, prices: List[float], T: int, B: int, 
                 valuation_mean: float = 5.0, valuation_std: float = 2.0):
        super().__init__(1, prices, T, B)
        self.valuation_mean = valuation_mean
        self.valuation_std = valuation_std
    
    def generate_valuations(self, round_t: int) -> np.ndarray:
        """Generate valuation from normal distribution"""
        valuation = np.random.normal(self.valuation_mean, self.valuation_std)
        return np.array([max(0, valuation)])  # Ensure non-negative

class StochasticMultiProductEnvironment(PricingEnvironment):
    """Stochastic environment for multiple products with correlated valuations"""
    
    def __init__(self, n_products: int, prices: List[float], T: int, B: int,
                 mean_valuations: Optional[np.ndarray] = None,
                 correlation_matrix: Optional[np.ndarray] = None):
        super().__init__(n_products, prices, T, B)
        
        if mean_valuations is None:
            self.mean_valuations = np.random.uniform(3, 8, n_products)
        else:
            self.mean_valuations = mean_valuations
            
        if correlation_matrix is None:
            # Generate random correlation matrix
            A = np.random.randn(n_products, n_products)
            self.cov_matrix = np.dot(A, A.T) + np.eye(n_products)
        else:
            self.cov_matrix = correlation_matrix
    
    def generate_valuations(self, round_t: int) -> np.ndarray:
        """Generate correlated valuations from multivariate normal"""
        valuations = np.random.multivariate_normal(self.mean_valuations, self.cov_matrix)
        return np.maximum(0, valuations)  # Ensure non-negative

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