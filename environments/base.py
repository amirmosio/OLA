import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple

class PricingEnvironment(ABC):
    """Base class for pricing environments"""
    
    def __init__(self, n_products: int, prices: List[float], T: int, B: int, seed: int = None):
        self.n_products = n_products
        self.prices = np.array(prices)
        self.n_prices = len(prices)
        self.T = T
        self.initial_B = B
        self.remaining_budget = B
        self.current_round = 0
        self.history = []
        
        # Set up random number generator with seed for reproducibility
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        else:
            self.rng = np.random.RandomState()
    
    @abstractmethod
    def generate_valuations(self, round_t: int) -> np.ndarray:
        """Generate customer valuations for round t"""
        pass
    
    def simulate_purchase(self, prices_set: np.ndarray, valuations: np.ndarray) -> Tuple[np.ndarray, float, int]:
        """Simulate customer purchase decision given prices and valuations"""
        purchases = (prices_set <= valuations).astype(int)
        revenue = np.sum(purchases * prices_set)
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