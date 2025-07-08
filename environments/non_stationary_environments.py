import numpy as np
from .base import PricingEnvironment
from typing import List, Dict

class HighlyNonStationaryEnvironment(PricingEnvironment):
    """Highly non-stationary environment with quickly changing distributions"""
    
    def __init__(self, n_products: int, prices: List[float], T: int, B: int,
                 base_mean: float = 5.0, base_std: float = 2.0, 
                 change_rate: float = 0.1, noise_scale: float = 1.0):
        """
        Initialize highly non-stationary environment with quickly changing distributions.
        
        Args:
            base_mean: Base mean valuation around which distributions fluctuate
            base_std: Base standard deviation for valuations
            change_rate: How quickly the distribution parameters change (0.1 = 10% change per round)
            noise_scale: Scale of the random fluctuations in distribution parameters
        """
        super().__init__(n_products, prices, T, B)
        self.base_mean = base_mean
        self.base_std = base_std
        self.change_rate = change_rate
        self.noise_scale = noise_scale
        
        # Initialize current distribution parameters
        if n_products == 1:
            self.current_mean = base_mean
            self.current_std = base_std
        else:
            self.current_mean = np.full(n_products, base_mean)
            # Initialize covariance matrix with base_std on diagonal
            self.current_cov = np.eye(n_products) * (base_std ** 2)
    
    def generate_valuations(self, round_t: int) -> np.ndarray:
        """Generate valuations with quickly changing distribution parameters"""
        
        # Update distribution parameters every round (highly non-stationary)
        if self.n_products == 1:
            # Add random walk to mean and std
            mean_change = np.random.normal(0, self.change_rate * self.noise_scale)
            std_change = np.random.normal(0, self.change_rate * self.noise_scale * 0.5)
            
            self.current_mean += mean_change
            self.current_std = max(0.5, self.current_std + std_change)  # Ensure positive std
            
            # Generate valuation with current (quickly changing) parameters
            valuation = np.random.normal(self.current_mean, self.current_std)
            return np.array([max(0, valuation)])
        else:
            # Multi-product case: update mean vector
            mean_changes = np.random.normal(0, self.change_rate * self.noise_scale, self.n_products)
            self.current_mean += mean_changes
            
            # Occasionally update covariance structure (less frequently than mean)
            if round_t % 10 == 0:  # Update covariance every 10 rounds
                noise_matrix = np.random.normal(0, self.change_rate * 0.1, (self.n_products, self.n_products))
                noise_matrix = (noise_matrix + noise_matrix.T) / 2  # Make symmetric
                self.current_cov += noise_matrix
                # Ensure positive definite by adding small diagonal term if needed
                eigenvals = np.linalg.eigvals(self.current_cov)
                if np.min(eigenvals) <= 0:
                    self.current_cov += np.eye(self.n_products) * (0.1 - np.min(eigenvals))
            
            # Generate valuations with current parameters
            valuations = np.random.multivariate_normal(self.current_mean, self.current_cov)
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