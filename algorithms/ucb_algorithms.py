import numpy as np
from collections import deque
from scipy.optimize import linear_sum_assignment
from .base import PricingAlgorithm
from typing import List, Dict, Tuple, Optional

class UCB1SingleProduct(PricingAlgorithm):
    """UCB1 algorithm for single product pricing (ignoring inventory constraint)"""
    
    def __init__(self, prices: List[float], T: int, B: int, seed: int = None):
        super().__init__(prices, 1, T, B, seed)
        self.counts = np.zeros(self.n_prices)
        self.values = np.zeros(self.n_prices)
        self.total_rewards = np.zeros(self.n_prices)
        
    def select_prices(self, remaining_budget: int) -> np.ndarray:
        """Select price using UCB1 formula"""
        if self.current_round < self.n_prices:
            # Initialize: try each price once
            price_idx = self.current_round
        else:
            # UCB1 selection
            confidence_bounds = self.values + np.sqrt(
                2 * np.log(self.T) / np.maximum(self.counts, 1)
            )
            price_idx = np.argmax(confidence_bounds)
        
        selected_prices = np.zeros(self.n_products)
        selected_prices[0] = self.prices[price_idx]
        return selected_prices
    
    def update(self, prices_set: np.ndarray, valuations: np.ndarray, 
               purchases: np.ndarray, revenue: float):
        """Update estimates with observed reward"""
        price_idx = np.where(self.prices == prices_set[0])[0][0]
        self.counts[price_idx] += 1
        self.total_rewards[price_idx] += revenue
        self.values[price_idx] = self.total_rewards[price_idx] / self.counts[price_idx]
        self.current_round += 1
        
    def reset(self):
        """Reset algorithm to initial state"""
        super().reset()
        self.counts = np.zeros(self.n_prices)
        self.values = np.zeros(self.n_prices)
        self.total_rewards = np.zeros(self.n_prices)

class UCBWithInventoryConstraintSingleProduct(PricingAlgorithm):
    """UCB algorithm with inventory constraint using UCB-like approach from images"""
    
    def __init__(self, prices: List[float], T: int, B: int, seed: int = None):
        super().__init__(prices, 1, T, B, seed)
        self.counts = np.zeros(self.n_prices)
        self.f_hat = np.zeros(self.n_prices)  # Revenue estimates
        self.c_hat = np.zeros(self.n_prices)  # Cost/consumption estimates
        self.total_revenue = np.zeros(self.n_prices)
        self.total_consumption = np.zeros(self.n_prices)
        
    def select_prices(self, remaining_budget: int) -> np.ndarray:
        """Select price using UCB with inventory constraint"""
        if remaining_budget <= 0:
            return np.array([float('inf')])  # No inventory left
            
        if self.current_round < self.n_prices:
            # Initialize: try each price once
            price_idx = self.current_round
        else:
            # Compute UCB bounds as shown in the algorithm
            f_ucb = np.zeros(self.n_prices)
            c_lcb = np.zeros(self.n_prices)
            
            for i in range(self.n_prices):
                if self.counts[i] > 0:
                    confidence_radius = np.sqrt(2 * np.log(self.T) / self.counts[i])
                    f_ucb[i] = self.f_hat[i] + confidence_radius
                    c_lcb[i] = self.c_hat[i] - confidence_radius
                else:
                    f_ucb[i] = float('inf')
                    c_lcb[i] = 0
            
            # Solve the optimization problem: maximize f_ucb subject to c_lcb <= rho
            # For single product, this is simple: find best price with acceptable consumption
            rho = remaining_budget / (self.T - self.current_round + 1)  # Average consumption budget
            
            valid_prices = c_lcb <= rho
            if np.any(valid_prices):
                price_idx = np.argmax(f_ucb * valid_prices + (-np.inf) * (~valid_prices))
            else:
                # If no price satisfies constraint, choose least consuming one
                price_idx = np.argmin(c_lcb)
        
        selected_prices = np.zeros(self.n_products)
        selected_prices[0] = self.prices[price_idx]
        return selected_prices
    
    def update(self, prices_set: np.ndarray, valuations: np.ndarray, 
               purchases: np.ndarray, revenue: float):
        """Update estimates with observed outcomes"""
        price_idx = np.where(self.prices == prices_set[0])[0][0]
        consumption = np.sum(purchases)  # Number of units sold
        
        self.counts[price_idx] += 1
        self.total_revenue[price_idx] += revenue
        self.total_consumption[price_idx] += consumption
        
        self.f_hat[price_idx] = self.total_revenue[price_idx] / self.counts[price_idx]
        self.c_hat[price_idx] = self.total_consumption[price_idx] / self.counts[price_idx]
        
        self.current_round += 1
        
    def reset(self):
        """Reset algorithm to initial state"""
        super().reset()
        self.counts = np.zeros(self.n_prices)
        self.f_hat = np.zeros(self.n_prices)
        self.c_hat = np.zeros(self.n_prices)
        self.total_revenue = np.zeros(self.n_prices)
        self.total_consumption = np.zeros(self.n_prices)

class CombinatorialUCB(PricingAlgorithm):
    """Combinatorial UCB for multiple products with inventory constraint"""
    
    def __init__(self, prices: List[float], n_products: int, T: int, B: int, seed: int = None):
        super().__init__(prices, n_products, T, B, seed)
        # For each product and price combination
        self.counts = np.zeros((n_products, self.n_prices))
        self.f_hat = np.zeros((n_products, self.n_prices))  # Revenue estimates per product-price
        self.c_hat = np.zeros((n_products, self.n_prices))  # Consumption estimates per product-price
        self.total_revenue = np.zeros((n_products, self.n_prices))
        self.total_consumption = np.zeros((n_products, self.n_prices))
        
    def select_prices(self, remaining_budget: int) -> np.ndarray:
        """Select prices using combinatorial optimization"""
        if remaining_budget <= 0:
            return np.full(self.n_products, float('inf'))
            
        if self.current_round < self.n_products * self.n_prices:
            # Initialize: try each product-price combination
            combo_idx = self.current_round
            product_idx = combo_idx // self.n_prices
            price_idx = combo_idx % self.n_prices
            
            selected_prices = np.full(self.n_products, float('inf'))  # Don't sell other products
            selected_prices[product_idx] = self.prices[price_idx]
            return selected_prices
        
        # Compute confidence bounds
        f_ucb = np.zeros((self.n_products, self.n_prices))
        c_lcb = np.zeros((self.n_products, self.n_prices))
        
        for i in range(self.n_products):
            for j in range(self.n_prices):
                if self.counts[i, j] > 0:
                    confidence_radius = np.sqrt(2 * np.log(self.T) / self.counts[i, j])
                    f_ucb[i, j] = self.f_hat[i, j] + confidence_radius
                    c_lcb[i, j] = self.c_hat[i, j] - confidence_radius
                else:
                    f_ucb[i, j] = float('inf')
                    c_lcb[i, j] = 0
        
        # Solve combinatorial optimization problem using Hungarian algorithm
        # Create cost matrix for assignment problem: products x prices
        
        rho = remaining_budget / (self.T - self.current_round + 1)
        cost_matrix = np.zeros((self.n_products, self.n_prices))
        
        # Estimate total budget consumption if we use the lowest consumption prices
        min_consumption_per_product = np.min(c_lcb, axis=1)
        base_consumption = np.sum(min_consumption_per_product)
        
        # Create cost matrix: negative revenue + penalty for budget violations
        for i in range(self.n_products):
            for j in range(self.n_prices):
                # Handle infinite values properly
                if f_ucb[i, j] == float('inf') or np.isnan(f_ucb[i, j]):
                    base_cost = 1000.0  # Large positive cost for invalid/infinite revenue
                else:
                    # Base cost is negative revenue (since Hungarian minimizes, we want to maximize revenue)
                    base_cost = -f_ucb[i, j]
                
                # Add penalty for budget constraint violations
                # Calculate penalty based on how much this choice exceeds the average budget allocation
                excess_consumption = max(0, c_lcb[i, j] - rho / self.n_products)
                
                # Calculate penalty based on total budget violation
                # If total consumption exceeds remaining budget, add large penalty
                if base_consumption - min_consumption_per_product[i] + c_lcb[i, j] > rho:
                    budget_penalty = 1000.0  # Large penalty for budget violation
                else:
                    budget_penalty = 10.0 * excess_consumption  # Smaller penalty for excess
                
                cost_matrix[i, j] = base_cost + budget_penalty
        
        # Ensure cost matrix contains only finite values
        cost_matrix = np.where(np.isfinite(cost_matrix), cost_matrix, 1000.0)
        
        # Solve the assignment problem using Hungarian algorithm
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        # Extract selected prices
        selected_prices = np.zeros(self.n_products)
        total_expected_consumption = 0
        
        for i, j in zip(row_indices, col_indices):
            # Check if assignment violates budget constraint
            if total_expected_consumption + c_lcb[i, j] <= rho:
                selected_prices[i] = self.prices[j]
                total_expected_consumption += c_lcb[i, j]
            else:
                selected_prices[i] = float('inf')  # Don't sell this product
        
        return selected_prices
    
    def update(self, prices_set: np.ndarray, valuations: np.ndarray, 
               purchases: np.ndarray, revenue: float):
        """Update estimates for each product-price combination used"""
        for i in range(self.n_products):
            if prices_set[i] < float('inf'):
                price_idx = np.where(self.prices == prices_set[i])[0][0]
                
                self.counts[i, price_idx] += 1
                # Revenue for this specific product
                product_revenue = purchases[i] * prices_set[i]
                self.total_revenue[i, price_idx] += product_revenue
                self.total_consumption[i, price_idx] += purchases[i]
                
                self.f_hat[i, price_idx] = self.total_revenue[i, price_idx] / self.counts[i, price_idx]
                self.c_hat[i, price_idx] = self.total_consumption[i, price_idx] / self.counts[i, price_idx]
        
        self.current_round += 1
        
    def reset(self):
        """Reset algorithm to initial state"""
        super().reset()
        self.counts = np.zeros((self.n_products, self.n_prices))
        self.f_hat = np.zeros((self.n_products, self.n_prices))
        self.c_hat = np.zeros((self.n_products, self.n_prices))
        self.total_revenue = np.zeros((self.n_products, self.n_prices))
        self.total_consumption = np.zeros((self.n_products, self.n_prices))

class SlidingWindowCombinatorialUCB(CombinatorialUCB):
    """Combinatorial UCB with sliding window for non-stationary environments"""
    
    def __init__(self, prices: List[float], n_products: int, T: int, B: int, window_size: int, seed: int = None):
        super().__init__(prices, n_products, T, B, seed)
        self.window_size = window_size
        # Store recent observations for sliding window
        self.recent_observations = deque(maxlen=window_size)
        
    def select_prices(self, remaining_budget: int) -> np.ndarray:
        """Select prices using sliding window estimates"""
        if remaining_budget <= 0:
            return np.full(self.n_products, float('inf'))
        
        # Recompute estimates using only recent observations
        self._update_sliding_window_estimates()
        
        # Use parent class logic with updated estimates
        return super().select_prices(remaining_budget)
    
    def _update_sliding_window_estimates(self):
        """Update estimates using only observations in sliding window"""
        # Reset estimates
        self.counts.fill(0)
        self.total_revenue.fill(0)
        self.total_consumption.fill(0)
        self.f_hat.fill(0)
        self.c_hat.fill(0)
        
        # Recompute from recent observations
        for obs in self.recent_observations:
            prices_set, purchases, revenue = obs
            
            for i in range(self.n_products):
                if prices_set[i] < float('inf'):
                    price_idx = np.where(self.prices == prices_set[i])[0][0]
                    
                    self.counts[i, price_idx] += 1
                    product_revenue = purchases[i] * prices_set[i]
                    self.total_revenue[i, price_idx] += product_revenue
                    self.total_consumption[i, price_idx] += purchases[i]
        
        # Update estimates
        for i in range(self.n_products):
            for j in range(self.n_prices):
                if self.counts[i, j] > 0:
                    self.f_hat[i, j] = self.total_revenue[i, j] / self.counts[i, j]
                    self.c_hat[i, j] = self.total_consumption[i, j] / self.counts[i, j]
    
    def update(self, prices_set: np.ndarray, valuations: np.ndarray, 
               purchases: np.ndarray, revenue: float):
        """Store observation in sliding window"""
        self.recent_observations.append((prices_set.copy(), purchases.copy(), revenue))
        self.current_round += 1
        
    def reset(self):
        """Reset algorithm to initial state"""
        super().reset()
        self.recent_observations.clear() 