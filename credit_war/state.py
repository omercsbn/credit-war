"""
State Representations for CREDIT WAR Environment

State space is fully observable and includes pending cash flow queues
to maintain the Markov property.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class AgentState:
    """
    State vector for a single agent (bank).
    
    Attributes:
        liquidity: Available liquid capital (L)
        risk: Cumulative risk exposure (R)
        capital: Long-term reserves and profit (C)
        pending_inflows: Queue [P1, P2, P3] for delayed loan repayments
    """
    
    liquidity: int
    risk: int
    capital: int
    pending_inflows: List[int] = field(default_factory=lambda: [0, 0, 0])
    
    def __post_init__(self):
        """Validate pending_inflows structure."""
        if len(self.pending_inflows) != 3:
            raise ValueError("pending_inflows must have exactly 3 elements [P1, P2, P3]")
    
    def copy(self) -> "AgentState":
        """Create deep copy of agent state."""
        return AgentState(
            liquidity=self.liquidity,
            risk=self.risk,
            capital=self.capital,
            pending_inflows=self.pending_inflows.copy()
        )


@dataclass
class GlobalState:
    """
    Complete environment state (fully observable).
    
    Attributes:
        agent_a: State of Agent A
        agent_b: State of Agent B
        turn: Current timestep (starts at 0)
        done: Whether episode has terminated
    """
    
    agent_a: AgentState
    agent_b: AgentState
    turn: int = 0
    done: bool = False
    
    def copy(self) -> "GlobalState":
        """Create deep copy of global state."""
        return GlobalState(
            agent_a=self.agent_a.copy(),
            agent_b=self.agent_b.copy(),
            turn=self.turn,
            done=self.done
        )
