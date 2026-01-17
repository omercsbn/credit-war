"""
Agent Interface and Base Classes for CREDIT WAR
"""

__all__ = ["BaseAgent", "RandomAgent", "GreedyAgent", "ConservativeAgent", "RuleBasedAgent", "AggressorAgent", "PPOAgent"]

from credit_war.agents.base import BaseAgent
from credit_war.agents.random_agent import RandomAgent
from credit_war.agents.greedy_agent import GreedyAgent
from credit_war.agents.conservative_agent import ConservativeAgent
from credit_war.agents.rule_based_agent import RuleBasedAgent
from credit_war.agents.aggressor_agent import AggressorAgent

# PPO Agent (requires stable-baselines3)
try:
    from credit_war.agents.ppo_agent import PPOAgent
except ImportError:
    # If stable-baselines3 not installed, PPOAgent won't be available
    PPOAgent = None
