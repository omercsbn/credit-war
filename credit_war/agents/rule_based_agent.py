"""
Rule-Based Agent for CREDIT WAR

Sophisticated heuristic combining aggression, defense, and sabotage.
Represents domain knowledge baseline for competitive performance.
"""

from typing import List

from credit_war.actions import Action
from credit_war.state import AgentState
from credit_war.agents.base import BaseAgent


class RuleBasedAgent(BaseAgent):
    """
    Adaptive agent using multi-criteria decision rules.
    
    Strategy (priority order):
    1. If own Risk > 30: Emergency risk mitigation (INSURE)
    2. If opponent Risk > 25: Opportunistic sabotage (UNDERCUT)
    3. If own Liquidity > 25 and Risk < 20: Aggressive loan issuance (GIVE_LOAN)
    4. If own Liquidity > 15: Safe growth (INVEST)
    5. Default: Conservative stance (REJECT)
    
    Expected Behavior: Balanced performance with dynamic adaptation
    """
    
    # Thresholds for decision logic
    EMERGENCY_RISK_THRESHOLD: int = 30
    OPPONENT_RISK_THRESHOLD: int = 25
    AGGRESSIVE_LIQUIDITY_THRESHOLD: int = 25
    AGGRESSIVE_RISK_CEILING: int = 20
    INVEST_LIQUIDITY_THRESHOLD: int = 15
    
    def __init__(self, name: str = "RuleBasedAgent", seed: int = 42):
        """
        Initialize rule-based agent.
        
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
        Select action using adaptive rule hierarchy.
        
        Args:
            own_state: Current state of this agent
            opponent_state: Current state of the opponent
            valid_actions: List of valid actions
            
        Returns:
            Selected action based on rule priority
        """
        if not valid_actions:
            raise ValueError("No valid actions available")
        
        # Rule 1: Emergency risk mitigation
        if own_state.risk > self.EMERGENCY_RISK_THRESHOLD:
            if Action.INSURE in valid_actions:
                return Action.INSURE
        
        # Rule 2: Opportunistic sabotage
        # Attack opponent if they have exploitable risk exposure
        if opponent_state.risk > self.OPPONENT_RISK_THRESHOLD and opponent_state.risk > 0:
            if Action.UNDERCUT in valid_actions:
                return Action.UNDERCUT
        
        # Rule 3: Aggressive loan issuance
        # Issue loans if liquidity is healthy and risk is manageable
        if (own_state.liquidity > self.AGGRESSIVE_LIQUIDITY_THRESHOLD and 
            own_state.risk < self.AGGRESSIVE_RISK_CEILING):
            if Action.GIVE_LOAN in valid_actions:
                return Action.GIVE_LOAN
        
        # Rule 4: Safe investment
        if own_state.liquidity > self.INVEST_LIQUIDITY_THRESHOLD:
            if Action.INVEST in valid_actions:
                return Action.INVEST
        
        # Rule 5: Conservative default
        if Action.REJECT in valid_actions:
            return Action.REJECT
        
        # Fallback: First valid action
        return valid_actions[0]
