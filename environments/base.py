import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple

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