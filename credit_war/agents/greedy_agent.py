"""
Greedy Agent for CREDIT WAR

Always attempts to issue loans (GIVE_LOAN) for maximum profit,
ignoring risk management. Used as baseline for degenerate aggressive strategy.
"""

from typing import List

from credit_war.actions import Action
from credit_war.state import AgentState
from credit_war.agents.base import BaseAgent


class GreedyAgent(BaseAgent):
    """
    Pure profit-maximization agent without risk management.
    
    Strategy:
    - Always play GIVE_LOAN if liquidity permits
    - Otherwise, play first valid action (fallback)
    
    Expected Behavior: Rapid capital growth followed by DEFAULT
    """
    
    def __init__(self, name: str = "GreedyAgent", seed: int = 42):
        """
        Initialize greedy agent.
        
        Args:
            name: Agent identifier
            seed: Random seed (unused, for interface compatibility)
        """
        super().__init__(name, seed)
    
    def select_action(
        self,
        own_state: AgentState,
        opponent_state: AgentState,
        valid_actions: List[Action]
    ) -> Action:
        """
        Select GIVE_LOAN if valid, otherwise fallback to first valid action.
        
        Args:
            own_state: Current state of this agent
            opponent_state: Current state of the opponent
            valid_actions: List of valid actions
            
        Returns:
            GIVE_LOAN or fallback action
        """
        if not valid_actions:
            raise ValueError("No valid actions available")
        
        # Prefer GIVE_LOAN for maximum profit
        if Action.GIVE_LOAN in valid_actions:
            return Action.GIVE_LOAN
        
        # Fallback to first valid action
        return valid_actions[0]
