"""
Base Agent Interface for CREDIT WAR

All agents must inherit from BaseAgent and implement select_action().
"""

from abc import ABC, abstractmethod
from typing import List

from credit_war.actions import Action
from credit_war.state import AgentState


class BaseAgent(ABC):
    """
    Abstract base class for all CREDIT WAR agents.
    
    Agents observe the complete global state (both own and opponent state)
    and must select an action from the valid action set.
    """
    
    def __init__(self, name: str = "Agent", seed: int = 42):
        """
        Initialize base agent.
        
        Args:
            name: Human-readable agent identifier
            seed: Random seed for reproducibility
        """
        self.name = name
        self.seed = seed
    
    @abstractmethod
    def select_action(
        self,
        own_state: AgentState,
        opponent_state: AgentState,
        valid_actions: List[Action]
    ) -> Action:
        """
        Select an action given current state observations.
        
        Args:
            own_state: Current state of this agent
            opponent_state: Current state of the opponent
            valid_actions: List of valid actions (liquidity-constrained)
            
        Returns:
            Selected action
        """
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
