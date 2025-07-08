import numpy as np
from abc import ABC, abstractmethod
from typing import List

class PricingAlgorithm(ABC):
    """Base class for pricing algorithms"""
    
    def __init__(self, prices: List[float], n_products: int, T: int, B: int):
        self.prices = np.array(prices)
        self.n_prices = len(prices)
        self.n_products = n_products
        self.T = T
        self.B = B
        self.initial_B = B
        self.current_round = 0
        self.history = []
        
    @abstractmethod
    def select_prices(self, remaining_budget: int) -> np.ndarray:
        """Select prices for current round"""
        pass
    
    def update(self, prices_set: np.ndarray, valuations: np.ndarray, 
               purchases: np.ndarray, revenue: float):
        """Update algorithm with observed outcomes"""
        pass
    
    def reset(self):
        """Reset algorithm to initial state"""
        self.current_round = 0
        self.history = [] 