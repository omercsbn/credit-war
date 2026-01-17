"""
Action Definitions for CREDIT WAR Environment

All actions are discrete and symmetric for both agents.
"""

from enum import IntEnum, auto


class Action(IntEnum):
    """
    Action space for CREDIT WAR agents.
    
    The ordering matters for determinism: valid_actions will always be sorted
    by this enum order to ensure cross-platform consistency.
    """
    
    GIVE_LOAN = auto()  # Issue risky loan for high return
    REJECT = auto()     # Decline loan; reduce risk
    INVEST = auto()     # Safe investment; moderate growth
    INSURE = auto()     # Purchase risk mitigation
    UNDERCUT = auto()   # Sabotage opponent's portfolio
