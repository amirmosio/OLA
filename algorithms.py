import numpy as np
import cvxpy as cp
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional
from collections import defaultdict, deque
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

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

class UCB1SingleProduct(PricingAlgorithm):
    """UCB1 algorithm for single product pricing (ignoring inventory constraint)"""
    
    def __init__(self, prices: List[float], T: int, B: int):
        super().__init__(prices, 1, T, B)
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
                2 * np.log(self.current_round) / np.maximum(self.counts, 1)
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

class UCBWithInventoryConstraintSingleProduct(PricingAlgorithm):
    """UCB algorithm with inventory constraint using UCB-like approach from images"""
    
    def __init__(self, prices: List[float], T: int, B: int):
        super().__init__(prices, 1, T, B)
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

class CombinatorialUCB(PricingAlgorithm):
    """Combinatorial UCB for multiple products with inventory constraint"""
    
    def __init__(self, prices: List[float], n_products: int, T: int, B: int):
        super().__init__(prices, n_products, T, B)
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
        
        # Solve the assignment problem using Hungarian algorithm
        try:
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
            
        except Exception as e:
            # Fallback to greedy approach if Hungarian algorithm fails
            selected_prices = np.zeros(self.n_products)
            total_expected_consumption = 0
            
            for i in range(self.n_products):
                best_ratio = -1
                best_price_idx = -1
                
                for j in range(self.n_prices):
                    if c_lcb[i, j] <= rho - total_expected_consumption:
                        ratio = f_ucb[i, j] / max(c_lcb[i, j], 1e-6)
                        if ratio > best_ratio:
                            best_ratio = ratio
                            best_price_idx = j
                
                if best_price_idx >= 0:
                    selected_prices[i] = self.prices[best_price_idx]
                    total_expected_consumption += c_lcb[i, best_price_idx]
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

class HungarianUCB(PricingAlgorithm):
    """Hungarian Algorithm-based UCB for optimal product-price assignment"""
    
    def __init__(self, prices: List[float], n_products: int, T: int, B: int, penalty_weight: float = 100.0):
        super().__init__(prices, n_products, T, B)
        self.penalty_weight = penalty_weight  # Weight for budget constraint penalties
        
        # For each product and price combination
        self.counts = np.zeros((n_products, self.n_prices))
        self.f_hat = np.zeros((n_products, self.n_prices))  # Revenue estimates per product-price
        self.c_hat = np.zeros((n_products, self.n_prices))  # Consumption estimates per product-price
        self.total_revenue = np.zeros((n_products, self.n_prices))
        self.total_consumption = np.zeros((n_products, self.n_prices))
        
    def select_prices(self, remaining_budget: int) -> np.ndarray:
        """Select prices using Hungarian algorithm for optimal assignment"""
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
                    c_lcb[i, j] = max(0, self.c_hat[i, j] - confidence_radius)
                else:
                    f_ucb[i, j] = float('inf')
                    c_lcb[i, j] = 0
        
        # Apply Hungarian algorithm for optimal assignment
        return self._hungarian_assignment(f_ucb, c_lcb, remaining_budget)
    
    def _hungarian_assignment(self, f_ucb: np.ndarray, c_lcb: np.ndarray, remaining_budget: int) -> np.ndarray:
        """Use Hungarian algorithm to find optimal product-price assignment"""
        
        # Average budget per remaining round
        rho = remaining_budget / (self.T - self.current_round + 1)
        
        # Create cost matrix for Hungarian algorithm
        # Rows = products, Columns = prices
        cost_matrix = np.zeros((self.n_products, self.n_prices))
        
        # Estimate minimum total consumption
        min_consumption_per_product = np.min(c_lcb, axis=1)
        total_min_consumption = np.sum(min_consumption_per_product)
        
        for i in range(self.n_products):
            for j in range(self.n_prices):
                # Primary objective: maximize revenue (negative because Hungarian minimizes)
                revenue_cost = -f_ucb[i, j] if f_ucb[i, j] != float('inf') else 1000.0
                
                # Budget constraint penalty
                # Calculate what total consumption would be if we assign price j to product i
                total_consumption_estimate = total_min_consumption - min_consumption_per_product[i] + c_lcb[i, j]
                
                if total_consumption_estimate > rho:
                    # Hard constraint violation - large penalty
                    budget_penalty = self.penalty_weight * (total_consumption_estimate - rho)
                else:
                    # Soft penalty for consumption above average allocation
                    avg_allocation = rho / self.n_products
                    excess = max(0, c_lcb[i, j] - avg_allocation)
                    budget_penalty = 0.1 * self.penalty_weight * excess
                
                cost_matrix[i, j] = revenue_cost + budget_penalty
        
        try:
            # Solve assignment problem using Hungarian algorithm
            row_indices, col_indices = linear_sum_assignment(cost_matrix)
            
            # Extract solution and verify budget constraints
            selected_prices = np.zeros(self.n_products)
            total_expected_consumption = 0
            
            # Sort assignments by cost to prioritize better assignments
            assignment_costs = [(cost_matrix[i, j], i, j) for i, j in zip(row_indices, col_indices)]
            assignment_costs.sort()
            
            assigned_products = set()
            for cost, i, j in assignment_costs:
                if i not in assigned_products and total_expected_consumption + c_lcb[i, j] <= rho:
                    selected_prices[i] = self.prices[j]
                    total_expected_consumption += c_lcb[i, j]
                    assigned_products.add(i)
                else:
                    selected_prices[i] = float('inf')  # Don't sell this product
            
            # For unassigned products, set price to infinity
            for i in range(self.n_products):
                if i not in assigned_products:
                    selected_prices[i] = float('inf')
            
            return selected_prices
            
        except Exception as e:
            # Fallback to safe assignment if Hungarian algorithm fails
            print(f"Hungarian algorithm failed: {e}. Using fallback.")
            return self._fallback_assignment(f_ucb, c_lcb, remaining_budget)
    
    def _fallback_assignment(self, f_ucb: np.ndarray, c_lcb: np.ndarray, remaining_budget: int) -> np.ndarray:
        """Fallback greedy assignment if Hungarian algorithm fails"""
        rho = remaining_budget / (self.T - self.current_round + 1)
        selected_prices = np.zeros(self.n_products)
        total_expected_consumption = 0
        
        # Sort products by best revenue-to-consumption ratio
        product_ratios = []
        for i in range(self.n_products):
            best_ratio = -1
            best_price_idx = -1
            
            for j in range(self.n_prices):
                if f_ucb[i, j] != float('inf') and c_lcb[i, j] > 0:
                    ratio = f_ucb[i, j] / c_lcb[i, j]
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_price_idx = j
            
            product_ratios.append((best_ratio, i, best_price_idx))
        
        product_ratios.sort(reverse=True)  # Sort by ratio (descending)
        
        for ratio, i, j in product_ratios:
            if j >= 0 and total_expected_consumption + c_lcb[i, j] <= rho:
                selected_prices[i] = self.prices[j]
                total_expected_consumption += c_lcb[i, j]
            else:
                selected_prices[i] = float('inf')
        
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

class PrimalDualSingleProduct(PricingAlgorithm):
    """Primal-dual algorithm for single product (best-of-both-worlds)"""
    
    def __init__(self, prices: List[float], T: int, B: int, eta: float = None):
        super().__init__(prices, 1, T, B)
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
        
        # Sample price according to probabilities
        price_idx = np.random.choice(self.n_prices, p=probabilities)
        
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

class PrimalDualMultiProduct(PricingAlgorithm):
    """Primal-dual algorithm for multiple products"""
    
    def __init__(self, prices: List[float], n_products: int, T: int, B: int, eta: float = None):
        super().__init__(prices, n_products, T, B)
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
            
            # Sample price for product i
            price_idx = np.random.choice(self.n_prices, p=probabilities)
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

class SlidingWindowCombinatorialUCB(CombinatorialUCB):
    """Combinatorial UCB with sliding window for non-stationary environments"""
    
    def __init__(self, prices: List[float], n_products: int, T: int, B: int, window_size: int):
        super().__init__(prices, n_products, T, B)
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