"""
Random Agent for CREDIT WAR

Selects actions uniformly at random from valid action set.
Uses deterministic random number generator for reproducibility.
"""

import random
from typing import List

from credit_war.actions import Action
from credit_war.state import AgentState
from credit_war.agents.base import BaseAgent


class RandomAgent(BaseAgent):
    """
    Agent that selects actions uniformly at random.
    
    Uses seeded random.Random() for deterministic behavior.
    """
    
    def __init__(self, name: str = "RandomAgent", seed: int = 42):
        """
        Initialize random agent.
        
        Args:
            name: Agent identifier
            seed: Random seed for reproducibility
        """
        super().__init__(name, seed)
        self.rng = random.Random(seed)
    
    def select_action(
        self,
        own_state: AgentState,
        opponent_state: AgentState,
        valid_actions: List[Action]
    ) -> Action:
        """
        Select action uniformly at random from valid actions.
        
        Args:
            own_state: Current state of this agent
            opponent_state: Current state of the opponent
            valid_actions: List of valid actions (sorted by enum order)
            
        Returns:
            Randomly selected action
        """
        if not valid_actions:
            raise ValueError("No valid actions available")
        
        return self.rng.choice(valid_actions)
