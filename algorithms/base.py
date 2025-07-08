import numpy as np
from abc import ABC, abstractmethod
from typing import List, Optional

class PricingAlgorithm(ABC):
    """Base class for pricing algorithms"""
    
    def __init__(self, prices: List[float], n_products: int, T: int, B: int, seed: int = None):
        self.prices = np.array(prices)
        self.n_prices = len(prices)
        self.n_products = n_products
        self.T = T
        self.initial_B = B
        self.current_round = 0
        
        # Set up random number generator with seed for reproducibility
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        else:
            self.rng = np.random.RandomState()
    
    @abstractmethod
    def select_prices(self, remaining_budget: int) -> np.ndarray:
        """Select prices for all products"""
        pass
    
    @abstractmethod
    def update(self, prices_set: np.ndarray, valuations: np.ndarray, 
               purchases: np.ndarray, revenue: float):
        """Update algorithm state with observed outcomes"""
        pass
    
    def reset(self):
        """Reset algorithm to initial state"""
        self.current_round = 0
        # Note: Subclasses should override this to reset their specific state 