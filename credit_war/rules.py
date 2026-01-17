"""
Game Rules and Constants for CREDIT WAR Environment

All numerical parameters and thresholds are defined here for easy tuning.
"""

from typing import Dict
from credit_war.actions import Action


class GameRules:
    """
    Centralized game constants and action mechanics.
    
    All values are based on the formal specification in the research design document.
    """
    
    # Initial State
    INITIAL_LIQUIDITY: int = 50
    INITIAL_RISK: int = 0
    INITIAL_CAPITAL: int = 50
    
    # Termination Conditions
    MAX_TURNS: int = 40
    RISK_THRESHOLD: int = 40  # DEFAULT triggered when Risk >= this
    
    # Action Costs (Liquidity reduction)
    ACTION_COSTS: Dict[Action, int] = {
        Action.GIVE_LOAN: 10,
        Action.REJECT: 0,
        Action.INVEST: 8,
        Action.INSURE: 7,
        Action.UNDERCUT: 5,
    }
    
    # Loan Mechanics
    LOAN_DELAY: int = 3  # Loans issued at turn t pay out at turn t+3
    LOAN_RETURN: int = 15  # Capital increase when loan matures
    
    # Action Effects on Risk
    GIVE_LOAN_RISK: int = 5
    REJECT_RISK_REDUCTION: int = -2
    INVEST_RISK: int = 0
    INSURE_RISK_REDUCTION: int = -8
    UNDERCUT_SELF_RISK: int = 3
    
    # Action Effects on Capital
    INVEST_CAPITAL_GAIN: int = 10
    INSURE_CAPITAL_COST: int = -3
    
    # UNDERCUT Interaction Effects
    UNDERCUT_TARGET_RISK_DAMAGE: int = 7
    UNDERCUT_TARGET_CAPITAL_DAMAGE: int = -10
    UNDERCUT_BACKFIRE_RISK_PENALTY: int = 5
    
    # Rewards (Sparse)
    REWARD_WIN: float = 1.0
    REWARD_LOSS: float = -1.0
    REWARD_DRAW: float = 0.0
    REWARD_STEP: float = 0.0  # Intermediate rewards are zero
    
    @classmethod
    def get_action_cost(cls, action: Action) -> int:
        """Get liquidity cost for an action."""
        return cls.ACTION_COSTS[action]
