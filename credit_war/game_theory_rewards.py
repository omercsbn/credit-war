"""
Game Theory-Based Reward Shaping for PPO Agents

Her oyun teorisi karakteri için özel reward function'lar.
Bu modül Gymnasium wrapper'dan çağrılır.
"""

from typing import Dict, Tuple
from credit_war.state import GlobalState
from credit_war.actions import Action


class RewardShaper:
    """Base class for reward shaping strategies."""
    
    def __init__(self, name: str):
        self.name = name
        self.prev_opponent_capital = None
        self.prev_own_capital = None
        self.cooperation_history = []
        self.betrayal_detected = False
        self.opponent_aggression_count = 0
        self.total_interactions = 0
    
    def reset(self):
        """Reset internal state for new episode."""
        self.prev_opponent_capital = None
        self.prev_own_capital = None
        self.cooperation_history = []
        self.betrayal_detected = False
        self.opponent_aggression_count = 0
        self.total_interactions = 0
    
    def compute_reward(
        self,
        action: Action,
        opponent_action: Action,
        state_before: GlobalState,
        state_after: GlobalState,
        agent_id: int,
        base_reward: float
    ) -> float:
        """
        Compute shaped reward based on game theory strategy.
        
        Args:
            action: Agent's action
            opponent_action: Opponent's action
            state_before: State before actions
            state_after: State after actions
            agent_id: 0 or 1 (which agent we are)
            base_reward: Default reward (1=win, -1=loss, 0=draw/continue)
        
        Returns:
            Shaped reward value
        """
        raise NotImplementedError


class NashEquilibriumRewardShaper(RewardShaper):
    """
    1️⃣ Rational Nash Banker
    
    Reward structure:
    - Penalize dominated strategies (pure aggression/passivity)
    - Reward balanced mixed strategies
    - Punish excessive risk accumulation
    - Reward stable equilibrium play
    """
    
    def __init__(self):
        super().__init__("NashEquilibrium")
    
    def compute_reward(self, action, opponent_action, state_before, state_after, agent_id, base_reward):
        reward = base_reward
        
        own_before = state_before.agent_a if agent_id == 0 else state_before.agent_b
        own_after = state_after.agent_a if agent_id == 0 else state_after.agent_b
        opp_before = state_before.agent_b if agent_id == 0 else state_before.agent_a
        opp_after = state_after.agent_b if agent_id == 0 else state_after.agent_a
        
        # Penalize excessive risk (Nash avoids dominated strategies)
        if own_after.risk >= 7:
            reward -= 0.3
        
        # Reward risk management
        if own_after.risk < own_before.risk:
            reward += 0.1
        
        # Penalize pure aggression (exploitable)
        if action in [Action.UNDERCUT, Action.UNDERCUT]:
            self.opponent_aggression_count += 1
            if self.opponent_aggression_count > 3:  # Too predictable
                reward -= 0.2
        
        # Reward balanced play (mixed strategy)
        if action in [Action.INVEST, Action.REJECT]:
            reward += 0.05
        
        # Survival is key in Nash
        if own_after.capital > own_before.capital:
            reward += 0.1
        
        return reward


class TitForTatRewardShaper(RewardShaper):
    """
    2️⃣ Tit-for-Tat Banker
    
    Reward structure:
    - Start cooperatively (INVEST/HOLD)
    - Mirror opponent aggression
    - Punish aggression immediately
    - Forgive after retaliation
    """
    
    def __init__(self):
        super().__init__("TitForTat")
        self.last_opponent_action = None
        self.retaliated = False
    
    def reset(self):
        super().reset()
        self.last_opponent_action = None
        self.retaliated = False
    
    def compute_reward(self, action, opponent_action, state_before, state_after, agent_id, base_reward):
        reward = base_reward
        
        own_after = state_after.agent_a if agent_id == 0 else state_after.agent_b
        
        # Detect opponent aggression
        is_opponent_aggressive = opponent_action in [Action.UNDERCUT, Action.UNDERCUT]
        is_own_aggressive = action in [Action.UNDERCUT, Action.UNDERCUT]
        
        # First move: reward cooperation
        if self.last_opponent_action is None:
            if action in [Action.INVEST, Action.REJECT]:
                reward += 0.3  # Strong incentive to cooperate first
            self.last_opponent_action = opponent_action
            return reward
        
        # Tit-for-tat logic: mirror last opponent action
        if is_opponent_aggressive:
            # Opponent was aggressive
            if is_own_aggressive and not self.retaliated:
                # Good! We retaliated
                reward += 0.4
                self.retaliated = True
            elif not is_own_aggressive:
                # Bad! We didn't retaliate
                reward -= 0.3
        else:
            # Opponent cooperated
            if not is_own_aggressive:
                # Good! We cooperated back
                reward += 0.3
                self.retaliated = False  # Forgive
            elif is_own_aggressive and self.retaliated:
                # Bad! We're being aggressive when opponent cooperated
                reward -= 0.2
        
        self.last_opponent_action = opponent_action
        return reward


class GrimTriggerRewardShaper(RewardShaper):
    """
    3️⃣ Grim Trigger Bank
    
    Reward structure:
    - Cooperate until first betrayal
    - After betrayal, permanent aggression
    - Never forgive
    """
    
    def __init__(self):
        super().__init__("GrimTrigger")
        self.betrayal_turn = None
    
    def reset(self):
        super().reset()
        self.betrayal_turn = None
    
    def compute_reward(self, action, opponent_action, state_before, state_after, agent_id, base_reward):
        reward = base_reward
        
        is_opponent_aggressive = opponent_action in [Action.UNDERCUT, Action.UNDERCUT]
        is_own_aggressive = action in [Action.UNDERCUT, Action.UNDERCUT]
        
        # Detect first betrayal
        if self.betrayal_turn is None and is_opponent_aggressive:
            self.betrayal_turn = self.total_interactions
        
        self.total_interactions += 1
        
        # Before betrayal: reward cooperation
        if self.betrayal_turn is None:
            if action in [Action.INVEST, Action.REJECT]:
                reward += 0.3
            else:
                reward -= 0.4  # Strong penalty for premature aggression
        else:
            # After betrayal: reward permanent aggression
            if is_own_aggressive:
                reward += 0.4  # Grim punishment
            else:
                reward -= 0.5  # Never forgive!
        
        return reward


class BayesianAdaptiveRewardShaper(RewardShaper):
    """
    4️⃣ Adaptive Bayesian Bank
    
    Reward structure:
    - Track opponent action frequencies (belief update)
    - Reward actions that exploit learned patterns
    - Bonus for accurate predictions
    """
    
    def __init__(self):
        super().__init__("BayesianAdaptive")
        self.opponent_action_counts = {action: 0 for action in Action}
        self.correct_predictions = 0
    
    def reset(self):
        super().reset()
        self.opponent_action_counts = {action: 0 for action in Action}
        self.correct_predictions = 0
    
    def compute_reward(self, action, opponent_action, state_before, state_after, agent_id, base_reward):
        reward = base_reward
        
        # Update belief distribution
        self.opponent_action_counts[opponent_action] += 1
        self.total_interactions += 1
        
        if self.total_interactions < 3:
            # Learning phase: small bonus for exploration
            reward += 0.05
            return reward
        
        # Predict most likely opponent action
        most_likely_opponent_action = max(
            self.opponent_action_counts,
            key=self.opponent_action_counts.get
        )
        
        # Reward exploiting predictions
        if most_likely_opponent_action in [Action.UNDERCUT, Action.UNDERCUT]:
            # Opponent is aggressive, defend or counter
            if action in [Action.INSURE, Action.UNDERCUT]:
                reward += 0.3
        elif most_likely_opponent_action in [Action.INVEST, Action.REJECT]:
            # Opponent is passive, invest or undercut
            if action in [Action.INVEST, Action.UNDERCUT]:
                reward += 0.3
        
        # Bonus for information gain (entropy reduction)
        action_entropy = len([c for c in self.opponent_action_counts.values() if c > 0])
        if action_entropy <= 2:  # Opponent is predictable
            reward += 0.2
        
        return reward


class PredatorRewardShaper(RewardShaper):
    """
    5️⃣ Predator Bank
    
    Reward structure:
    - Attack when opponent is weak (high risk, low capital)
    - Disengage when retaliation is costly
    - Seek domination, not equilibrium
    """
    
    def __init__(self):
        super().__init__("Predator")
    
    def compute_reward(self, action, opponent_action, state_before, state_after, agent_id, base_reward):
        reward = base_reward
        
        own_after = state_after.agent_a if agent_id == 0 else state_after.agent_b
        opp_before = state_before.agent_b if agent_id == 0 else state_before.agent_a
        opp_after = state_after.agent_b if agent_id == 0 else state_after.agent_a
        
        # Detect weakness (high risk OR low capital)
        opponent_is_weak = opp_after.risk >= 5 or opp_after.capital <= 30
        
        # Reward aggression against weak opponents
        if opponent_is_weak and action in [Action.UNDERCUT, Action.UNDERCUT]:
            reward += 0.5  # Strong hunting incentive
        
        # Penalize aggression against strong opponents
        if not opponent_is_weak and action in [Action.UNDERCUT, Action.UNDERCUT]:
            reward -= 0.3
        
        # Reward causing opponent damage
        if opp_after.capital < opp_before.capital:
            reward += 0.2
        if opp_after.risk > opp_before.risk:
            reward += 0.1
        
        # Penalize taking unnecessary risk
        if own_after.risk >= 7:
            reward -= 0.4
        
        return reward


class MinimaxRewardShaper(RewardShaper):
    """
    6️⃣ Risk-Averse Regulator-Minded Bank
    
    Reward structure:
    - Minimize maximum possible loss
    - Survival > profit
    - Avoid catastrophic failure
    - Defensive play
    """
    
    def __init__(self):
        super().__init__("Minimax")
        self.max_risk_seen = 0
    
    def reset(self):
        super().reset()
        self.max_risk_seen = 0
    
    def compute_reward(self, action, opponent_action, state_before, state_after, agent_id, base_reward):
        reward = base_reward
        
        own_after = state_after.agent_a if agent_id == 0 else state_after.agent_b
        
        # Track worst-case risk
        self.max_risk_seen = max(self.max_risk_seen, own_after.risk)
        
        # Heavy penalty for high risk (worst-case scenario)
        if own_after.risk >= 8:
            reward -= 1.0  # Catastrophic
        elif own_after.risk >= 6:
            reward -= 0.5
        elif own_after.risk >= 4:
            reward -= 0.2
        
        # Reward low risk
        if own_after.risk <= 2:
            reward += 0.3
        
        # Reward defensive actions
        if action == Action.INSURE:
            reward += 0.2
        if action == Action.REJECT:
            reward += 0.1
        
        # Survival bonus
        if own_after.capital >= 50:
            reward += 0.2
        
        return reward


class OpportunisticSwitcherRewardShaper(RewardShaper):
    """
    7️⃣ Opportunistic Switcher
    
    Reward structure:
    - Alternate between cooperation and aggression
    - Create unpredictability
    - Time regime shifts based on opponent state
    """
    
    def __init__(self):
        super().__init__("OpportunisticSwitcher")
        self.regime = "cooperative"  # "cooperative" or "aggressive"
        self.turns_in_regime = 0
        self.regime_switch_threshold = 5
    
    def reset(self):
        super().reset()
        self.regime = "cooperative"
        self.turns_in_regime = 0
    
    def compute_reward(self, action, opponent_action, state_before, state_after, agent_id, base_reward):
        reward = base_reward
        
        opp_after = state_after.agent_b if agent_id == 0 else state_after.agent_a
        
        self.turns_in_regime += 1
        
        # Check if opponent is vulnerable (time to switch to aggressive)
        opponent_vulnerable = opp_after.risk >= 5 or opp_after.capital <= 35
        
        # Reward regime-appropriate actions
        if self.regime == "cooperative":
            if action in [Action.INVEST, Action.REJECT]:
                reward += 0.2
            
            # Switch to aggressive if opponent vulnerable OR threshold reached
            if opponent_vulnerable or self.turns_in_regime >= self.regime_switch_threshold:
                self.regime = "aggressive"
                self.turns_in_regime = 0
                reward += 0.1  # Bonus for timing
        
        else:  # aggressive regime
            if action in [Action.UNDERCUT, Action.UNDERCUT]:
                reward += 0.3
            
            # Switch back to cooperative after threshold
            if self.turns_in_regime >= self.regime_switch_threshold:
                self.regime = "cooperative"
                self.turns_in_regime = 0
        
        # Bonus for unpredictability (regime switching)
        if self.turns_in_regime == 0:
            reward += 0.15
        
        return reward


class EvolutionaryLearnerRewardShaper(RewardShaper):
    """
    8️⃣ Evolutionary Learner
    
    Reward structure:
    - Imitate successful behaviors
    - Discard unsuccessful ones
    - Survival-based selection
    - Gradual adaptation
    """
    
    def __init__(self):
        super().__init__("EvolutionaryLearner")
        self.successful_actions = []
        self.failed_actions = []
    
    def reset(self):
        super().reset()
        # Keep history across episodes (evolutionary memory)
        # Only reset per-episode tracking
    
    def compute_reward(self, action, opponent_action, state_before, state_after, agent_id, base_reward):
        reward = base_reward
        
        own_before = state_before.agent_a if agent_id == 0 else state_before.agent_b
        own_after = state_after.agent_a if agent_id == 0 else state_after.agent_b
        
        # Track success/failure
        capital_improved = own_after.capital > own_before.capital
        risk_reduced = own_after.risk < own_before.risk
        
        if capital_improved or risk_reduced:
            self.successful_actions.append(action)
            reward += 0.2
        else:
            self.failed_actions.append(action)
        
        # Reward imitating previously successful actions
        if len(self.successful_actions) > 0:
            most_successful = max(
                set(self.successful_actions),
                key=self.successful_actions.count
            )
            if action == most_successful:
                reward += 0.15
        
        # Penalize repeating failed actions
        if action in self.failed_actions:
            failure_count = self.failed_actions.count(action)
            reward -= 0.1 * min(failure_count, 3)
        
        # Survival bonus
        if own_after.capital > 0:
            reward += 0.05
        
        return reward


class ZeroSumWarriorRewardShaper(RewardShaper):
    """
    9️⃣ Zero-Sum Warrior
    
    Reward structure:
    - Maximize relative advantage (not absolute profit)
    - Willing to sacrifice own capital to hurt opponent
    - Pure competition
    """
    
    def __init__(self):
        super().__init__("ZeroSumWarrior")
    
    def compute_reward(self, action, opponent_action, state_before, state_after, agent_id, base_reward):
        reward = base_reward
        
        own_before = state_before.agent_a if agent_id == 0 else state_before.agent_b
        own_after = state_after.agent_a if agent_id == 0 else state_after.agent_b
        opp_before = state_before.agent_b if agent_id == 0 else state_before.agent_a
        opp_after = state_after.agent_b if agent_id == 0 else state_after.agent_a
        
        # Relative advantage is key
        own_advantage_before = own_before.capital - opp_before.capital
        own_advantage_after = own_after.capital - opp_after.capital
        
        # Reward increasing relative advantage
        if own_advantage_after > own_advantage_before:
            reward += 0.4
        
        # Reward hurting opponent even if it costs us
        opponent_lost_capital = opp_before.capital - opp_after.capital
        if opponent_lost_capital > 0:
            reward += 0.3  # Worth it even if we also lose
        
        # Aggressive actions are good in zero-sum
        if action in [Action.UNDERCUT, Action.UNDERCUT]:
            reward += 0.2
        
        # Penalize being ahead but not aggressive
        if own_advantage_after > 10 and action not in [Action.UNDERCUT, Action.UNDERCUT]:
            reward -= 0.2  # Finish them!
        
        return reward


class MetaLearnerRewardShaper(RewardShaper):
    """
    🔟 Meta-Learner Bank
    
    Reward structure:
    - Classify opponent into archetypes
    - Switch strategies based on classification
    - Learn which strategy works against which opponent type
    - Adaptable across all scenarios
    """
    
    def __init__(self):
        super().__init__("MetaLearner")
        self.opponent_profile = {
            "aggression_score": 0.0,  # 0-1
            "risk_tolerance": 0.0,    # 0-1
            "cooperativeness": 0.0,   # 0-1
        }
        self.current_strategy = "balanced"  # "aggressive", "defensive", "balanced"
        self.strategy_performance = {
            "aggressive": [],
            "defensive": [],
            "balanced": []
        }
    
    def reset(self):
        super().reset()
        # Keep opponent profile and strategy performance (meta-learning)
        self.current_strategy = "balanced"
    
    def compute_reward(self, action, opponent_action, state_before, state_after, agent_id, base_reward):
        reward = base_reward
        
        own_before = state_before.agent_a if agent_id == 0 else state_before.agent_b
        own_after = state_after.agent_a if agent_id == 0 else state_after.agent_b
        opp_before = state_before.agent_b if agent_id == 0 else state_before.agent_a
        opp_after = state_after.agent_b if agent_id == 0 else state_after.agent_a
        
        # Update opponent profile
        self.total_interactions += 1
        alpha = 0.1  # Learning rate
        
        # Aggression: opponent uses UNDERCUT/SABOTAGE
        if opponent_action in [Action.UNDERCUT, Action.UNDERCUT]:
            self.opponent_profile["aggression_score"] += alpha * (1.0 - self.opponent_profile["aggression_score"])
        else:
            self.opponent_profile["aggression_score"] += alpha * (0.0 - self.opponent_profile["aggression_score"])
        
        # Risk tolerance: opponent maintains high risk
        if opp_after.risk >= 5:
            self.opponent_profile["risk_tolerance"] += alpha * (1.0 - self.opponent_profile["risk_tolerance"])
        else:
            self.opponent_profile["risk_tolerance"] += alpha * (0.0 - self.opponent_profile["risk_tolerance"])
        
        # Cooperativeness: opponent uses INVEST/HOLD
        if opponent_action in [Action.INVEST, Action.REJECT]:
            self.opponent_profile["cooperativeness"] += alpha * (1.0 - self.opponent_profile["cooperativeness"])
        else:
            self.opponent_profile["cooperativeness"] += alpha * (0.0 - self.opponent_profile["cooperativeness"])
        
        # Classify opponent and select strategy
        if self.opponent_profile["aggression_score"] > 0.6:
            best_strategy = "defensive"
        elif self.opponent_profile["cooperativeness"] > 0.6:
            best_strategy = "aggressive"
        else:
            best_strategy = "balanced"
        
        # Track strategy performance
        capital_change = own_after.capital - own_before.capital
        self.strategy_performance[self.current_strategy].append(capital_change)
        
        # Reward using best strategy
        if self.current_strategy == best_strategy:
            reward += 0.3
        
        # Reward strategy-appropriate actions
        if self.current_strategy == "aggressive" and action in [Action.UNDERCUT, Action.INVEST]:
            reward += 0.2
        elif self.current_strategy == "defensive" and action in [Action.INSURE, Action.REJECT]:
            reward += 0.2
        elif self.current_strategy == "balanced" and action in [Action.INVEST, Action.REJECT, Action.INSURE]:
            reward += 0.15
        
        # Switch to best performing strategy
        if self.total_interactions > 10 and self.total_interactions % 5 == 0:
            self.current_strategy = best_strategy
            reward += 0.1  # Bonus for adaptation
        
        return reward


# Registry of all reward shapers
REWARD_SHAPER_REGISTRY = {
    "nash": NashEquilibriumRewardShaper,
    "titfortat": TitForTatRewardShaper,
    "grimtrigger": GrimTriggerRewardShaper,
    "bayesian": BayesianAdaptiveRewardShaper,
    "predator": PredatorRewardShaper,
    "minimax": MinimaxRewardShaper,
    "switcher": OpportunisticSwitcherRewardShaper,
    "evolutionary": EvolutionaryLearnerRewardShaper,
    "zerosum": ZeroSumWarriorRewardShaper,
    "metalearner": MetaLearnerRewardShaper,
}


def get_reward_shaper(name: str) -> RewardShaper:
    """Get reward shaper by name."""
    if name not in REWARD_SHAPER_REGISTRY:
        raise ValueError(f"Unknown reward shaper: {name}. Available: {list(REWARD_SHAPER_REGISTRY.keys())}")
    return REWARD_SHAPER_REGISTRY[name]()
