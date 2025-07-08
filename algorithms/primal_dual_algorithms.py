import numpy as np
from .base import PricingAlgorithm
from typing import List, Dict, Tuple, Optional

class PrimalDualSingleProduct(PricingAlgorithm):
    """Primal-dual algorithm for single product (best-of-both-worlds)"""
    
    def __init__(self, prices: List[float], T: int, B: int, eta: float = None, seed: int = None):
        super().__init__(prices, 1, T, B, seed)
        self.eta = eta if eta else min(0.1, np.sqrt(np.log(len(prices)) / T))
        self.lambda_t = 0  # Dual variable for budget constraint
        self.log_weights = np.zeros(self.n_prices)  # Use log weights for stability
        
        # For tracking consumption
        self.counts = np.zeros(self.n_prices)
        self.total_consumption = np.zeros(self.n_prices)
        self.c_hat = np.zeros(self.n_prices)
        
    def select_prices(self, remaining_budget: int) -> np.ndarray:
        """Select price using exponential weights with dual variable"""
        if remaining_budget <= 0:
            return np.array([float('inf')])
        
        # Update dual variable based on budget consumption rate
        budget_used = self.initial_B - remaining_budget
        if self.current_round > 0:
            consumption_rate = budget_used / self.current_round
            target_rate = self.initial_B / self.T
            
            dual_update = self.eta * (consumption_rate - target_rate)
            self.lambda_t += dual_update
            self.lambda_t = max(0, min(self.lambda_t, 10.0))  # Bound dual variable
        
        # Compute probability distribution over prices using log-sum-exp trick
        log_adjusted_weights = self.log_weights - self.lambda_t * self.c_hat
        
        # Numerical stability: subtract max before exponentiating
        max_log_weight = np.max(log_adjusted_weights)
        stable_weights = np.exp(log_adjusted_weights - max_log_weight)
        
        # Normalize to get probabilities
        weight_sum = np.sum(stable_weights)
        if weight_sum == 0 or not np.isfinite(weight_sum):
            probabilities = np.ones(self.n_prices) / self.n_prices  # Uniform fallback
        else:
            probabilities = stable_weights / weight_sum
        
        # Ensure probabilities are valid
        probabilities = np.clip(probabilities, 1e-10, 1.0)
        probabilities /= np.sum(probabilities)
        
        # Sample price according to probabilities using seeded RNG
        price_idx = self.rng.choice(self.n_prices, p=probabilities)
        
        selected_prices = np.zeros(self.n_products)
        selected_prices[0] = self.prices[price_idx]
        return selected_prices
    
    def update(self, prices_set: np.ndarray, valuations: np.ndarray, 
               purchases: np.ndarray, revenue: float):
        """Update weights using multiplicative weights with numerical stability"""
        price_idx = np.where(self.prices == prices_set[0])[0][0]
        consumption = np.sum(purchases)

        # Update consumption estimates
        self.counts[price_idx] += 1
        self.total_consumption[price_idx] += consumption
        if self.counts[price_idx] > 0:
            self.c_hat[price_idx] = self.total_consumption[price_idx] / self.counts[price_idx]
        
        # Update weight for selected price based on observed reward
        reward = revenue - self.lambda_t * consumption  # Lagrangian reward
        # Clip reward to prevent overflow
        reward = np.clip(reward, -10.0, 10.0)
        self.log_weights[price_idx] += self.eta * reward
        
        # Prevent log weights from growing too large
        self.log_weights = np.clip(self.log_weights, -50.0, 50.0)
        
        self.current_round += 1
        
    def reset(self):
        """Reset algorithm to initial state"""
        super().reset()
        self.lambda_t = 0
        self.log_weights = np.zeros(self.n_prices)
        self.counts = np.zeros(self.n_prices)
        self.total_consumption = np.zeros(self.n_prices)
        self.c_hat = np.zeros(self.n_prices)

class PrimalDualMultiProduct(PricingAlgorithm):
    """Primal-dual algorithm for multiple products"""
    
    def __init__(self, prices: List[float], n_products: int, T: int, B: int, eta: float = None, seed: int = None):
        super().__init__(prices, n_products, T, B, seed)
        self.eta = eta if eta else min(0.1, np.sqrt(np.log(len(prices)) / T))
        self.lambda_t = 0
        # Separate log weights for each product for numerical stability
        self.log_weights = np.zeros((n_products, self.n_prices))
        
        # For tracking consumption
        self.counts = np.zeros((n_products, self.n_prices))
        self.total_consumption = np.zeros((n_products, self.n_prices))
        self.c_hat = np.zeros((n_products, self.n_prices))
        
    def select_prices(self, remaining_budget: int) -> np.ndarray:
        """Select prices using independent exponential weights per product"""
        if remaining_budget <= 0:
            return np.full(self.n_products, float('inf'))
        
        # Update dual variable
        budget_used = self.initial_B - remaining_budget
        if self.current_round > 0:
            consumption_rate = budget_used / self.current_round
            target_rate = self.initial_B / self.T
            self.lambda_t += self.eta * (consumption_rate - target_rate)
            self.lambda_t = max(0, min(self.lambda_t, 10.0))  # Bound dual variable
        
        selected_prices = np.zeros(self.n_products)
        
        for i in range(self.n_products):
            # Compute probabilities for product i using log-sum-exp trick
            log_adjusted_weights = self.log_weights[i] - self.lambda_t * self.c_hat[i]
            
            # Numerical stability: subtract max before exponentiating
            max_log_weight = np.max(log_adjusted_weights)
            stable_weights = np.exp(log_adjusted_weights - max_log_weight)
            
            # Normalize to get probabilities
            weight_sum = np.sum(stable_weights)
            if weight_sum == 0 or not np.isfinite(weight_sum):
                probabilities = np.ones(self.n_prices) / self.n_prices  # Uniform fallback
            else:
                probabilities = stable_weights / weight_sum
            
            # Ensure probabilities are valid
            probabilities = np.clip(probabilities, 1e-10, 1.0)
            probabilities /= np.sum(probabilities)
            
            # Sample price for product i using seeded RNG
            price_idx = self.rng.choice(self.n_prices, p=probabilities)
            selected_prices[i] = self.prices[price_idx]
        
        return selected_prices
    
    def update(self, prices_set: np.ndarray, valuations: np.ndarray, 
               purchases: np.ndarray, revenue: float):
        """Update weights for each product independently with numerical stability"""
        for i in range(self.n_products):
            price_idx = np.where(self.prices == prices_set[i])[0][0]
            
            # Update consumption estimates for product i
            self.counts[i, price_idx] += 1
            self.total_consumption[i, price_idx] += purchases[i]
            if self.counts[i, price_idx] > 0:
                self.c_hat[i, price_idx] = self.total_consumption[i, price_idx] / self.counts[i, price_idx]
            
            # Compute reward for product i
            product_revenue = purchases[i] * prices_set[i]
            reward = product_revenue - self.lambda_t * purchases[i]
            
            # Clip reward to prevent overflow
            reward = np.clip(reward, -10.0, 10.0)
            
            # Update log weight
            self.log_weights[i, price_idx] += self.eta * reward
            
        # Prevent log weights from growing too large
        self.log_weights = np.clip(self.log_weights, -50.0, 50.0)
        
        self.current_round += 1
        
    def reset(self):
        """Reset algorithm to initial state"""
        super().reset()
        self.lambda_t = 0
        self.log_weights = np.zeros((self.n_products, self.n_prices))
        self.counts = np.zeros((self.n_products, self.n_prices))
        self.total_consumption = np.zeros((self.n_products, self.n_prices))
        self.c_hat = np.zeros((self.n_products, self.n_prices)) 