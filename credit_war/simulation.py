"""
Simulation Runner for CREDIT WAR

Executes episodes and collects performance metrics for analysis.
"""

from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, field

from credit_war.env import CreditWarEnv
from credit_war.agents.base import BaseAgent
from credit_war.state import AgentState


@dataclass
class EpisodeMetrics:
    """Metrics collected from a single episode."""
    
    turns: int
    outcome: str
    winner: str
    
    agent_a_final_capital: int
    agent_a_final_risk: int
    agent_a_final_liquidity: int
    agent_a_survival: bool
    agent_a_reward: float
    
    agent_b_final_capital: int
    agent_b_final_risk: int
    agent_b_final_liquidity: int
    agent_b_survival: bool
    agent_b_reward: float
    
    action_counts_a: Dict[str, int] = field(default_factory=dict)
    action_counts_b: Dict[str, int] = field(default_factory=dict)


@dataclass
class AggregateMetrics:
    """Aggregate statistics across multiple episodes."""
    
    total_episodes: int
    
    # Survival statistics
    agent_a_survival_rate: float
    agent_b_survival_rate: float
    
    # Win statistics
    agent_a_wins: int
    agent_b_wins: int
    draws: int
    
    # Performance statistics
    agent_a_avg_capital: float
    agent_b_avg_capital: float
    agent_a_avg_reward: float
    agent_b_avg_reward: float
    
    # Action frequency (normalized)
    agent_a_action_freq: Dict[str, float]
    agent_b_action_freq: Dict[str, float]
    
    # Episode length
    avg_turns: float


class SimulationRunner:
    """
    Execute episodes and collect metrics for agent evaluation.
    """
    
    def __init__(self, env: CreditWarEnv):
        """
        Initialize simulation runner.
        
        Args:
            env: CREDIT WAR environment instance
        """
        self.env = env
        self.episode_history: List[EpisodeMetrics] = []
    
    def run_episode(
        self,
        agent_a: BaseAgent,
        agent_b: BaseAgent,
        verbose: bool = False
    ) -> EpisodeMetrics:
        """
        Run a single episode between two agents.
        
        Args:
            agent_a: First agent
            agent_b: Second agent
            verbose: Print turn-by-turn information
            
        Returns:
            Episode metrics
        """
        state = self.env.reset()
        done = False
        
        total_reward_a = 0.0
        total_reward_b = 0.0
        
        action_counts_a: Dict[str, int] = {}
        action_counts_b: Dict[str, int] = {}
        
        while not done:
            # Get valid actions
            valid_a = self.env.get_valid_actions(state.agent_a)
            valid_b = self.env.get_valid_actions(state.agent_b)
            
            # Agents select actions
            action_a = agent_a.select_action(state.agent_a, state.agent_b, valid_a)
            action_b = agent_b.select_action(state.agent_b, state.agent_a, valid_b)
            
            # Track action counts
            action_counts_a[action_a.name] = action_counts_a.get(action_a.name, 0) + 1
            action_counts_b[action_b.name] = action_counts_b.get(action_b.name, 0) + 1
            
            if verbose:
                print(f"Turn {state.turn}: {agent_a.name}={action_a.name}, {agent_b.name}={action_b.name}")
            
            # Execute step
            state, reward_a, reward_b, done, info = self.env.step(action_a, action_b)
            
            total_reward_a += reward_a
            total_reward_b += reward_b
            
            if verbose and done:
                print(f"Episode ended: {info['outcome']}")
        
        # Determine winner
        outcome = info.get("outcome", "unknown")
        if "agent_a_wins" in outcome:
            winner = "agent_a"
        elif "agent_b_wins" in outcome:
            winner = "agent_b"
        else:
            winner = "draw"
        
        # Create metrics
        metrics = EpisodeMetrics(
            turns=state.turn,
            outcome=outcome,
            winner=winner,
            agent_a_final_capital=state.agent_a.capital,
            agent_a_final_risk=state.agent_a.risk,
            agent_a_final_liquidity=state.agent_a.liquidity,
            agent_a_survival=not ("agent_b_wins" in outcome and "failure" in outcome),
            agent_a_reward=total_reward_a,
            agent_b_final_capital=state.agent_b.capital,
            agent_b_final_risk=state.agent_b.risk,
            agent_b_final_liquidity=state.agent_b.liquidity,
            agent_b_survival=not ("agent_a_wins" in outcome and "failure" in outcome),
            agent_b_reward=total_reward_b,
            action_counts_a=action_counts_a,
            action_counts_b=action_counts_b
        )
        
        self.episode_history.append(metrics)
        return metrics
    
    def run_tournament(
        self,
        agent_a: BaseAgent,
        agent_b: BaseAgent,
        num_episodes: int = 100,
        verbose: bool = False
    ) -> AggregateMetrics:
        """
        Run multiple episodes and compute aggregate statistics.
        
        Args:
            agent_a: First agent
            agent_b: Second agent
            num_episodes: Number of episodes to run
            verbose: Print episode information
            
        Returns:
            Aggregate metrics across all episodes
        """
        self.episode_history = []
        
        for episode_idx in range(num_episodes):
            if verbose and episode_idx % 10 == 0:
                print(f"Episode {episode_idx + 1}/{num_episodes}")
            
            self.run_episode(agent_a, agent_b, verbose=False)
        
        return self.compute_aggregate_metrics()
    
    def compute_aggregate_metrics(self) -> AggregateMetrics:
        """
        Compute aggregate statistics from episode history.
        
        Returns:
            Aggregate metrics
        """
        if not self.episode_history:
            raise ValueError("No episodes in history")
        
        total_episodes = len(self.episode_history)
        
        # Count outcomes
        agent_a_wins = sum(1 for ep in self.episode_history if ep.winner == "agent_a")
        agent_b_wins = sum(1 for ep in self.episode_history if ep.winner == "agent_b")
        draws = sum(1 for ep in self.episode_history if ep.winner == "draw")
        
        # Survival rates
        agent_a_survivals = sum(1 for ep in self.episode_history if ep.agent_a_survival)
        agent_b_survivals = sum(1 for ep in self.episode_history if ep.agent_b_survival)
        
        agent_a_survival_rate = agent_a_survivals / total_episodes
        agent_b_survival_rate = agent_b_survivals / total_episodes
        
        # Average capital (over all episodes, including failures)
        agent_a_avg_capital = sum(ep.agent_a_final_capital for ep in self.episode_history) / total_episodes
        agent_b_avg_capital = sum(ep.agent_b_final_capital for ep in self.episode_history) / total_episodes
        
        # Average rewards
        agent_a_avg_reward = sum(ep.agent_a_reward for ep in self.episode_history) / total_episodes
        agent_b_avg_reward = sum(ep.agent_b_reward for ep in self.episode_history) / total_episodes
        
        # Average turns
        avg_turns = sum(ep.turns for ep in self.episode_history) / total_episodes
        
        # Action frequencies (normalized across all episodes)
        action_totals_a: Dict[str, int] = {}
        action_totals_b: Dict[str, int] = {}
        
        for ep in self.episode_history:
            for action, count in ep.action_counts_a.items():
                action_totals_a[action] = action_totals_a.get(action, 0) + count
            for action, count in ep.action_counts_b.items():
                action_totals_b[action] = action_totals_b.get(action, 0) + count
        
        total_actions_a = sum(action_totals_a.values())
        total_actions_b = sum(action_totals_b.values())
        
        action_freq_a = {
            action: count / total_actions_a if total_actions_a > 0 else 0.0
            for action, count in action_totals_a.items()
        }
        action_freq_b = {
            action: count / total_actions_b if total_actions_b > 0 else 0.0
            for action, count in action_totals_b.items()
        }
        
        return AggregateMetrics(
            total_episodes=total_episodes,
            agent_a_survival_rate=agent_a_survival_rate,
            agent_b_survival_rate=agent_b_survival_rate,
            agent_a_wins=agent_a_wins,
            agent_b_wins=agent_b_wins,
            draws=draws,
            agent_a_avg_capital=agent_a_avg_capital,
            agent_b_avg_capital=agent_b_avg_capital,
            agent_a_avg_reward=agent_a_avg_reward,
            agent_b_avg_reward=agent_b_avg_reward,
            agent_a_action_freq=action_freq_a,
            agent_b_action_freq=action_freq_b,
            avg_turns=avg_turns
        )
