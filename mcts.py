import numpy as np
import torch

class RiskSensitiveMCTSNode:
    """
    Tree node for Risk-Sensitive MCTS in continuous-time portfolio control.
    Tracks full reward distributions for rigorous CVaR-penalized UCB selection.
    """
    def __init__(self, action_idx, parent=None, prior=1.0):
        self.action_idx = action_idx
        self.parent = parent
        self.prior = prior

        self.visit_count = 0
        self.value_sum  = 0.0
        self.rewards_history = []

    @property
    def q_value(self):
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def compute_cvar(self, alpha=0.05):
        """CVaR_{alpha}: mean of the worst-alpha fraction of observed rewards."""
        if len(self.rewards_history) < 2:
            return 0.0
        arr = np.array(self.rewards_history)
        n_tail = max(1, int(np.floor(alpha * len(arr))))
        return float(np.mean(np.sort(arr)[:n_tail]))

    def ucb_score(self, total_visits, c_puct=1.0, c_cvar=5.0):
        """
        UCB score with explicit CVaR penalization:
          score = Q(a) - c_cvar * max(0, -CVaR_{5%}(a)) + c_puct * P(a) * sqrt(N) / (1 + n(a))
        A large c_cvar strongly steers the planner away from heavy-tail-loss actions.
        """
        cvar_pen = c_cvar * max(0.0, -self.compute_cvar(alpha=0.05))
        explore  = c_puct * self.prior * (np.sqrt(total_visits + 1) / (1 + self.visit_count))
        return self.q_value - cvar_pen + explore


class RiskSensitiveMCTSPlanner:
    """
    Risk-Sensitive MCTS Planner.

    Combines neural policy priors with multi-step trajectory simulation.
    Uses strong CVaR penalization (c_cvar >= 5.0) to actively suppress left-tail losses.
    """
    def __init__(self, policy_net, num_simulations=50, c_puct=1.0, c_cvar=5.0,
                 rollout_depth=5, alpha_cvar=0.05):
        self.policy_net    = policy_net
        self.num_simulations = num_simulations
        self.c_puct        = c_puct
        self.c_cvar        = c_cvar
        self.rollout_depth = rollout_depth
        self.alpha_cvar    = alpha_cvar

    def _build_candidates(self, prior, N):
        """
        Generate a diverse set of portfolio weight candidates.
        Includes: neural prior, defensive allocations, low-concentration,
        and high-exploration Dirichlet perturbations.
        """
        candidates = [prior.copy()]                             # 1. Neural prior

        # 2. Uniform equal-weight (maximum diversification / defensive)
        candidates.append(np.ones(N) / N)

        # 3. High-alpha Dirichlet (near-uniform, low-concentration)
        candidates.append(np.random.dirichlet(np.ones(N) * 3.0))

        # 4. Low-alpha Dirichlet (high-concentration, exploratory)
        candidates.append(np.random.dirichlet(np.ones(N) * 0.3))

        # 5. Softmax-dampened prior (shrink extremes toward equal-weight)
        dampened = 0.5 * prior + 0.5 * np.ones(N) / N
        candidates.append(dampened / dampened.sum())

        # 6. Squared-prior (emphasize top-conviction assets)
        sq = prior ** 2.0
        candidates.append(sq / sq.sum())

        # 7-8. Two additional Dirichlet draws for variance
        for alpha in [1.0, 0.5]:
            candidates.append(np.random.dirichlet(np.ones(N) * alpha))

        return candidates

    def _multi_step_rollout(self, env_clone, action, depth):
        """
        Roll out a portfolio policy for `depth` steps using action cloning.
        Returns cumulative discounted reward and list of step rewards.
        """
        rewards = []
        gamma_disc = 0.99
        env_sim = _clone_env(env_clone)

        # Apply selected action first
        _, r0, done0, _ = env_sim.step(action)
        rewards.append(r0)

        # Continue with neural prior for remaining steps
        for _ in range(depth - 1):
            if done0:
                break
            obs_t = _get_obs_tensor(env_sim)
            with torch.no_grad():
                w, _, _, _, _ = self.policy_net(obs_t, wealth_t=env_sim.wealth)
                act_next = w.squeeze(0).numpy()
            _, r, done0, _ = env_sim.step(act_next)
            rewards.append(r)

        # Cumulative discounted return
        G = 0.0
        for r in reversed(rewards):
            G = r + gamma_disc * G
        return G, rewards

    def plan_action(self, env_clone, obs_tensor):
        """
        Run MCTS simulations with CVaR-penalized UCB selection.
        Returns the portfolio weight vector with best risk-adjusted value.
        """
        with torch.no_grad():
            prior_w, _, _, _, _ = self.policy_net(obs_tensor, wealth_t=env_clone.wealth)
            prior = prior_w.squeeze(0).numpy()

        N = env_clone.num_assets
        candidates = self._build_candidates(prior, N)
        K = len(candidates)

        nodes = [RiskSensitiveMCTSNode(action_idx=i, prior=1.0/K) for i in range(K)]
        total_visits = 0

        for _ in range(self.num_simulations):
            # UCB selection
            scores  = [nd.ucb_score(total_visits, self.c_puct, self.c_cvar) for nd in nodes]
            best_i  = int(np.argmax(scores))

            # Multi-step rollout for selected candidate
            G, step_rewards = self._multi_step_rollout(
                env_clone, candidates[best_i], depth=self.rollout_depth
            )

            # Back-propagate
            nodes[best_i].visit_count  += 1
            nodes[best_i].value_sum    += G
            nodes[best_i].rewards_history.extend(step_rewards)
            total_visits += 1

        # Final selection: highest visit count (most explored = best risk-adjusted)
        best_idx = max(range(K), key=lambda i: nodes[i].visit_count)
        return candidates[best_idx]


# ======================================================
# Environment utilities
# ======================================================
def _clone_env(env):
    """Fast shallow clone of portfolio environment state for MCTS simulation."""
    import copy
    from portfolio_env import EMKOPortfolioEnv
    
    env_seed = getattr(env, "seed_val", None)
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
        seed=env_seed if env_seed is not None else 42
    )
    if hasattr(env, "np_random") and env.np_random is not None:
        env_copy.np_random = copy.deepcopy(env.np_random)
        
    env_copy.current_step  = env.current_step
    env_copy.wealth        = env.wealth
    env_copy.weights       = env.weights.copy()
    env_copy.x_t           = env.x_t.copy()
    env_copy.price_history = [p.copy() for p in env.price_history]
    env_copy.x_history     = [x.copy() for x in env.x_history]
    return env_copy


def _get_obs_tensor(env):
    """Build observation tensor from environment state."""
    # EMKOPortfolioEnv uses _get_observation() as its internal method name
    obs = env._get_observation()
    return torch.tensor(obs, dtype=torch.float32).unsqueeze(0)


# Backward-compatible alias
def EMKOPortfolioEnvCopy(env):
    return _clone_env(env)
