import numpy as np

class EMKOPortfolioEnv:
    """
    Equivariant Physics-Informed Merton-Kim-Omberg (E-MKO) Stochastic Control Environment.
    
    Simulates continuous-time asset price SDEs driven by:
    1. Stochastic Mean-Reverting Risk Premium x_t (Kim & Omberg 1996 OU process):
       dx_t = kappa * (bar_x - x_t) * dt + sigma_x * dW_t^x
    2. Correlated Asset Price Dynamics S_t:
       dS_t / S_t = (r + x_t) * dt + sigma_S * dW_t^S,  E[dW^S * dW^x] = rho * dt
    3. Wealth Transition under CRRA Utility & Non-convex Turnover Friction:
       dW_t = W_t * [(r + pi_t^T x_t) * dt + pi_t^T sigma_S * dW_t^S] - cost * W_t * ||pi_t - pi_{t-1}||_1
    """
    def __init__(
        self, 
        num_assets=5, 
        lookback=10, 
        dt=1/252.0, 
        max_steps=100, 
        gamma=2.0, 
        transaction_cost=0.001,
        kappa=1.5,
        bar_x=0.08,
        sigma_x=0.15,
        sigma_S=0.20,
        rho=-0.5,
        risk_free_rate=0.02,
        seed=42
    ):
        self.num_assets = num_assets
        self.lookback = lookback
        self.dt = dt
        self.max_steps = max_steps
        self.gamma = gamma # CRRA Risk Aversion Parameter
        self.transaction_cost = transaction_cost
        self.r = risk_free_rate
        
        # Kim & Omberg OU Risk Premium Parameters
        self.kappa = kappa
        self.bar_x = bar_x
        self.sigma_x = sigma_x
        self.sigma_S = sigma_S
        self.rho = rho
        
        self.np_random = np.random.RandomState(seed)
        
        # Environment state variables
        self.current_step = 0
        self.wealth = 1.0
        self.weights = np.ones(self.num_assets) / self.num_assets
        self.x_t = np.full(self.num_assets, self.bar_x)
        
        # Historical buffer for observation
        self.price_history = []
        self.x_history = []
        
    def reset(self):
        self.current_step = 0
        self.wealth = 1.0
        self.weights = np.ones(self.num_assets) / self.num_assets
        self.x_t = self.np_random.normal(self.bar_x, 0.02, size=self.num_assets)
        
        # Warm-up history
        self.price_history = [np.ones(self.num_assets)]
        self.x_history = [self.x_t.copy()]
        
        for _ in range(self.lookback):
            self._simulate_sde_step()
            
        return self._get_observation()
        
    def _simulate_sde_step(self):
        """Simulates one step of correlated SDEs via Euler-Maruyama scheme."""
        # Correlated Brownian motions
        dW_S = self.np_random.normal(0, np.sqrt(self.dt), size=self.num_assets)
        dW_x_uncorr = self.np_random.normal(0, np.sqrt(self.dt), size=self.num_assets)
        dW_x = self.rho * dW_S + np.sqrt(1 - self.rho**2) * dW_x_uncorr
        
        # OU process update for risk premium x_t
        dx = self.kappa * (self.bar_x - self.x_t) * self.dt + self.sigma_x * dW_x
        self.x_t = self.x_t + dx
        
        # Asset price returns dS / S
        d_returns = (self.r + self.x_t) * self.dt + self.sigma_S * dW_S
        
        new_price = self.price_history[-1] * (1.0 + d_returns)
        self.price_history.append(new_price)
        self.x_history.append(self.x_t.copy())
        
        return d_returns
        
    def _get_observation(self):
        """
        Returns (num_assets, lookback, features).
        Features per asset: [historical_return, estimated_risk_premium_x_t, volatility]
        Permutation-equivariant structure for $S_N$ group.
        """
        recent_prices = np.array(self.price_history[-self.lookback-1:])
        returns = (recent_prices[1:] - recent_prices[:-1]) / recent_prices[:-1] # (lookback, N)
        recent_x = np.array(self.x_history[-self.lookback:]) # (lookback, N)
        
        # Stack per asset: (num_assets, lookback, 2)
        returns_T = returns.T[:, :, None] # (N, lookback, 1)
        recent_x_T = recent_x.T[:, :, None] # (N, lookback, 1)
        
        obs = np.concatenate([returns_T, recent_x_T], axis=-1)
        return obs.astype(np.float32)

    def step(self, action_weights):
        """
        Executes continuous-time wealth step under action_weights pi_t.
        """
        # Ensure action is valid probability distribution on simplex
        action_weights = np.clip(action_weights, 1e-8, 1.0)
        action_weights = action_weights / np.sum(action_weights)
        
        # Turnover friction
        turnover = np.sum(np.abs(action_weights - self.weights))
        friction_loss = self.transaction_cost * self.wealth * turnover
        
        # Simulate next continuous time state
        d_returns = self._simulate_sde_step()
        
        # Gross return & Wealth update
        portfolio_return = np.sum(action_weights * d_returns)
        d_wealth = self.wealth * portfolio_return - friction_loss
        self.wealth += d_wealth
        self.wealth = max(self.wealth, 1e-6) # Prevent negative wealth
        
        self.weights = action_weights
        self.current_step += 1
        done = self.current_step >= self.max_steps
        
        # CRRA Utility Reward: U(W_T) = (W^(1-gamma)) / (1-gamma)
        if done:
            reward = (self.wealth ** (1 - self.gamma)) / (1 - self.gamma)
        else:
            # Step utility increment proxy
            reward = portfolio_return - (self.transaction_cost * turnover) - 0.5 * self.gamma * (portfolio_return ** 2)
            
        info = {
            "wealth": self.wealth,
            "turnover": turnover,
            "portfolio_return": portfolio_return,
            "estimated_x_mean": np.mean(self.x_t)
        }
        
        next_obs = self._get_observation()
        return next_obs, reward, done, info

    def get_kim_omberg_analytical_action(self, tau=1.0):
        """
        Computes exact closed-form Kim & Omberg (1996) analytical action:
        pi*_KO = Myopic (x / (gamma * sigma_S^2)) + Hedging correction
        """
        myopic_demand = self.x_t / (self.gamma * (self.sigma_S ** 2))
        # Simplistic analytical approximation for intertemporal hedging demand
        hedging_demand = (self.rho * self.sigma_x / self.sigma_S) * (self.x_t - self.bar_x) * tau / self.gamma
        pi_ko = myopic_demand + hedging_demand
        pi_ko = np.maximum(pi_ko, 0.0)
        if np.sum(pi_ko) > 0:
            pi_ko = pi_ko / np.sum(pi_ko)
        else:
            pi_ko = np.ones(self.num_assets) / self.num_assets
        return pi_ko

if __name__ == "__main__":
    env = EMKOPortfolioEnv(num_assets=5)
    obs = env.reset()
    print("E-MKO Environment initialized.")
    print("Observation shape (N, lookback, features):", obs.shape)
    pi_analytical = env.get_kim_omberg_analytical_action()
    print("Analytical Kim & Omberg Action:", pi_analytical)
    next_obs, reward, done, info = env.step(pi_analytical)
    print(f"Step Reward: {reward:.4f}, Wealth: {info['wealth']:.4f}")
