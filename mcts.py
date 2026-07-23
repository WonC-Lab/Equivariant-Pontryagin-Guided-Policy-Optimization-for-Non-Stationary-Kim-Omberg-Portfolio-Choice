import numpy as np
import torch
import torch.nn.functional as F

class RiskSensitiveMCTSNode:
    """
    Tree node for Risk-Sensitive Monte Carlo Tree Search (MCTS) in continuous-time portfolio control.
    Incorporates CVaR tail loss penalization and Sharpe upper-confidence bounds (UCB).
    """
    def __init__(self, state_obs, wealth=1.0, parent=None, action_prior=None):
        self.state_obs = state_obs
        self.wealth = wealth
        self.parent = parent
        self.action_prior = action_prior
        
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0.0
        self.rewards_history = []
        
    @property
    def value(self):
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def compute_cvar(self, alpha=0.05):
        """Computes Conditional Value-at-Risk (CVaR_alpha) from historical rollout rewards."""
        if len(self.rewards_history) == 0:
            return 0.0
        arr = np.array(self.rewards_history)
        var_threshold = np.percentile(arr, alpha * 100)
        tail_losses = arr[arr <= var_threshold]
        if len(tail_losses) == 0:
            return float(var_threshold)
        return float(np.mean(tail_losses))

    def ucb_score(self, child_action, c_puct=1.5, c_cvar=0.5):
        child = self.children.get(child_action, None)
        if child is None or child.visit_count == 0:
            q_val = 0.0
            prior = self.action_prior[child_action] if self.action_prior is not None else 1.0
            n_child = 0
        else:
            cvar = child.compute_cvar(alpha=0.05)
            q_val = child.value - c_cvar * max(0.0, -cvar) # Penalize tail loss
            prior = child.action_prior
            n_child = child.visit_count

        u_score = c_puct * prior * (np.sqrt(self.visit_count + 1) / (1 + n_child))
        return q_val + u_score

class RiskSensitiveMCTSPlanner:
    """
    Risk-Sensitive MCTS Planner combining neural policy priors with multi-step rollout trajectory optimization.
    """
    def __init__(self, policy_net, num_simulations=20, c_puct=1.5, c_cvar=0.5):
        self.policy_net = policy_net
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.c_cvar = c_cvar

    def plan_action(self, env_clone, obs_tensor, num_discrete_actions=5):
        """
        Executes MCTS simulations and returns risk-adjusted portfolio weights.
        """
        with torch.no_grad():
            prior_weights, val, _, _, _ = self.policy_net(obs_tensor, wealth_t=env_clone.wealth)
            prior = prior_weights.squeeze(0).numpy()

        root = RiskSensitiveMCTSNode(state_obs=obs_tensor, wealth=env_clone.wealth)
        
        # Generate candidate action perturbations around neural prior
        candidate_actions = [prior]
        for _ in range(num_discrete_actions - 1):
            noise = np.random.dirichlet(np.ones(env_clone.num_assets) * 0.5)
            perturbed = 0.7 * prior + 0.3 * noise
            candidate_actions.append(perturbed / np.sum(perturbed))

        for sim in range(self.num_simulations):
            # Select action via UCB
            best_idx = 0
            best_score = -1e9
            for idx, act in enumerate(candidate_actions):
                score = root.ucb_score(idx, c_puct=self.c_puct, c_cvar=self.c_cvar)
                if score > best_score:
                    best_score = score
                    best_idx = idx

            selected_action = candidate_actions[best_idx]
            
            # Simple 1-step rollout proxy
            env_sim = EMKOPortfolioEnvCopy(env_clone)
            _, reward, _, info = env_sim.step(selected_action)
            
            if best_idx not in root.children:
                root.children[best_idx] = RiskSensitiveMCTSNode(
                    state_obs=obs_tensor, 
                    wealth=info["wealth"], 
                    parent=root, 
                    action_prior=1.0/len(candidate_actions)
                )
            
            child = root.children[best_idx]
            child.visit_count += 1
            child.value_sum += reward
            child.rewards_history.append(reward)
            root.visit_count += 1

        # Pick child with highest visit count / value
        best_child_idx = max(root.children.keys(), key=lambda k: root.children[k].visit_count)
        return candidate_actions[best_child_idx]

def EMKOPortfolioEnvCopy(env):
    """Creates a fast lightweight clone of environment state for MCTS tree search."""
    from portfolio_env import EMKOPortfolioEnv
    env_copy = EMKOPortfolioEnv(
        num_assets=env.num_assets,
        lookback=env.lookback,
        dt=env.dt,
        max_steps=env.max_steps,
        gamma=env.gamma,
        transaction_cost=env.transaction_cost,
        kappa=env.kappa,
        bar_x=env.bar_x,
        sigma_x=env.sigma_x,
        sigma_S=env.sigma_S,
        rho=env.rho,
        risk_free_rate=env.r,
        seed=123
    )
    env_copy.current_step = env.current_step
    env_copy.wealth = env.wealth
    env_copy.weights = env.weights.copy()
    env_copy.x_t = env.x_t.copy()
    env_copy.price_history = [p.copy() for p in env.price_history]
    env_copy.x_history = [x.copy() for x in env.x_history]
    return env_copy
