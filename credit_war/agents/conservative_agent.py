"""
Conservative Agent for CREDIT WAR

Prioritizes risk minimization and survival over profit maximization.
Used as baseline for defensive strategy.
"""

from typing import List

from credit_war.actions import Action
from credit_war.state import AgentState
from credit_war.agents.base import BaseAgent


class ConservativeAgent(BaseAgent):
    """
    Risk-minimizing agent with defensive strategy.
    
    Strategy:
    - If Risk > 15: Prioritize INSURE (reduce risk exposure)
    - Else if Liquidity > 20: Play INVEST (safe capital growth)
    - Otherwise: Play REJECT (maintain stability)
    
    Expected Behavior: High survival rate, moderate capital accumulation
    """
    
    # Risk threshold for triggering insurance
    RISK_THRESHOLD: int = 15
    
    # Liquidity threshold for investing
    LIQUIDITY_THRESHOLD: int = 20
    
    def __init__(self, name: str = "ConservativeAgent", seed: int = 42):
        """
        Initialize conservative agent.
        
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
        Select action based on conservative risk management rules.
        
        Args:
            own_state: Current state of this agent
            opponent_state: Current state of the opponent
            valid_actions: List of valid actions
            
        Returns:
            Selected action based on risk/liquidity heuristics
        """
        if not valid_actions:
            raise ValueError("No valid actions available")
        
        # Priority 1: Reduce risk if threshold exceeded
        if own_state.risk > self.RISK_THRESHOLD:
            if Action.INSURE in valid_actions:
                return Action.INSURE
        
        # Priority 2: Safe investment if liquidity permits
        if own_state.liquidity > self.LIQUIDITY_THRESHOLD:
            if Action.INVEST in valid_actions:
                return Action.INVEST
        
        # Priority 3: Reject (conservative default)
        if Action.REJECT in valid_actions:
            return Action.REJECT
        
        # Fallback: First valid action
        return valid_actions[0]
