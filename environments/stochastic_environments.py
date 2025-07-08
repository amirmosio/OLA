import numpy as np
from .base import PricingEnvironment
from typing import List, Optional

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