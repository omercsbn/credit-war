"""
CREDIT WAR: A Deterministic Multi-Agent Strategic Environment for Financial Risk Modeling

A research-grade simulation for studying competitive banking behavior, systemic risk,
and emergent coordination in Multi-Agent Reinforcement Learning (MARL).
"""

from credit_war.actions import Action
from credit_war.state import AgentState, GlobalState
from credit_war.env import CreditWarEnv
from credit_war.rules import GameRules

__version__ = "1.0.0"
__all__ = ["Action", "AgentState", "GlobalState", "CreditWarEnv", "GameRules"]
