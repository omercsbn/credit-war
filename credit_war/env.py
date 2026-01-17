"""
Core Environment Logic for CREDIT WAR

Implements deterministic state transitions with strict order of operations.
This environment satisfies the Markov property by including pending cash flows
in the state vector.
"""

from typing import List, Tuple, Dict, Any
from copy import deepcopy

from credit_war.actions import Action
from credit_war.state import AgentState, GlobalState
from credit_war.rules import GameRules


class CreditWarEnv:
    """
    Deterministic, fully observable, simultaneous-action environment for
    studying competitive banking behavior and systemic risk.
    
    State transitions follow a strict five-phase execution model to ensure
    determinism and correct timing of delayed payouts.
    """
    
    def __init__(self, seed: int = 42):
        """
        Initialize the CREDIT WAR environment.
        
        Args:
            seed: Random seed (for compatibility; environment is deterministic)
        """
        self.seed = seed
        self.state: GlobalState = self._create_initial_state()
    
    def reset(self) -> GlobalState:
        """
        Reset environment to initial state.
        
        Returns:
            Initial global state
        """
        self.state = self._create_initial_state()
        return self.state.copy()
    
    def _create_initial_state(self) -> GlobalState:
        """Create initial state with default values."""
        return GlobalState(
            agent_a=AgentState(
                liquidity=GameRules.INITIAL_LIQUIDITY,
                risk=GameRules.INITIAL_RISK,
                capital=GameRules.INITIAL_CAPITAL,
                pending_inflows=[0, 0, 0]
            ),
            agent_b=AgentState(
                liquidity=GameRules.INITIAL_LIQUIDITY,
                risk=GameRules.INITIAL_RISK,
                capital=GameRules.INITIAL_CAPITAL,
                pending_inflows=[0, 0, 0]
            ),
            turn=0,
            done=False
        )
    
    def get_valid_actions(self, agent_state: AgentState) -> List[Action]:
        """
        Get valid actions for an agent based on current liquidity.
        
        CRITICAL: Returns actions sorted by enum definition order for determinism.
        
        Args:
            agent_state: Current state of the agent
            
        Returns:
            Sorted list of valid actions (where liquidity >= action cost)
        """
        valid = []
        # Iterate over Action enum in definition order
        for action in Action:
            cost = GameRules.get_action_cost(action)
            if agent_state.liquidity >= cost:
                valid.append(action)
        return valid
    
    def step(
        self, 
        action_a: Action, 
        action_b: Action
    ) -> Tuple[GlobalState, float, float, bool, Dict[str, Any]]:
        """
        Execute one timestep of the environment.
        
        This method implements a strict five-phase execution model:
        
        PHASE 1: SNAPSHOT & VALIDATION
            - Create deep copy of current state (Pre-Action State)
            - Validate actions against current liquidity
            - Override invalid actions to REJECT
        
        PHASE 2: COMPUTE DELTAS (PARALLEL EXECUTION)
            - Calculate all state changes based on SNAPSHOT only
            - Do NOT mutate state yet
            - Compute UNDERCUT interactions using pre-action risk values
            - Calculate new_loan_inflow for GIVE_LOAN actions
        
        PHASE 3: APPLY DELTAS & QUEUE SHIFT
            1. Apply computed deltas to L, R, C
            2. Process payouts and shift pending_inflows queue:
               - Capital += P1 (current P1 value)
               - P1 = P2
               - P2 = P3
               - P3 = new_loan_inflow (newly issued loan)
        
        PHASE 4: CLAMPING
            - Liquidity = max(0, L)
            - Risk = max(0, R)
            - Capital is NOT clamped (can go negative)
        
        PHASE 5: TURN INCREMENT & TERMINATION
            - Increment turn counter
            - Check failure conditions (DEFAULT, BANKRUPTCY)
            - Check time limit (MAX_TURNS)
            - Compute terminal rewards
        
        Args:
            action_a: Action selected by Agent A
            action_b: Action selected by Agent B
            
        Returns:
            Tuple of (next_state, reward_a, reward_b, done, info)
        """
        
        # ===================================================================
        # PHASE 1: SNAPSHOT & VALIDATION
        # ===================================================================
        
        # Create snapshot of pre-action state
        snapshot = self.state.copy()
        
        # Validate actions and override if insufficient liquidity
        cost_a = GameRules.get_action_cost(action_a)
        cost_b = GameRules.get_action_cost(action_b)
        
        if snapshot.agent_a.liquidity < cost_a:
            action_a = Action.REJECT
            cost_a = 0
        
        if snapshot.agent_b.liquidity < cost_b:
            action_b = Action.REJECT
            cost_b = 0
        
        # ===================================================================
        # PHASE 2: COMPUTE DELTAS (PARALLEL EXECUTION)
        # ===================================================================
        
        # Initialize deltas for Agent A
        delta_L_a = -cost_a
        delta_R_a = 0
        delta_C_a = 0
        new_loan_inflow_a = 0
        
        # Initialize deltas for Agent B
        delta_L_b = -cost_b
        delta_R_b = 0
        delta_C_b = 0
        new_loan_inflow_b = 0
        
        # Apply base action effects for Agent A
        if action_a == Action.GIVE_LOAN:
            delta_R_a += GameRules.GIVE_LOAN_RISK
            new_loan_inflow_a = GameRules.LOAN_RETURN
        elif action_a == Action.REJECT:
            delta_R_a += GameRules.REJECT_RISK_REDUCTION
        elif action_a == Action.INVEST:
            delta_R_a += GameRules.INVEST_RISK
            delta_C_a += GameRules.INVEST_CAPITAL_GAIN
        elif action_a == Action.INSURE:
            delta_R_a += GameRules.INSURE_RISK_REDUCTION
            delta_C_a += GameRules.INSURE_CAPITAL_COST
        elif action_a == Action.UNDERCUT:
            delta_R_a += GameRules.UNDERCUT_SELF_RISK
        
        # Apply base action effects for Agent B
        if action_b == Action.GIVE_LOAN:
            delta_R_b += GameRules.GIVE_LOAN_RISK
            new_loan_inflow_b = GameRules.LOAN_RETURN
        elif action_b == Action.REJECT:
            delta_R_b += GameRules.REJECT_RISK_REDUCTION
        elif action_b == Action.INVEST:
            delta_R_b += GameRules.INVEST_RISK
            delta_C_b += GameRules.INVEST_CAPITAL_GAIN
        elif action_b == Action.INSURE:
            delta_R_b += GameRules.INSURE_RISK_REDUCTION
            delta_C_b += GameRules.INSURE_CAPITAL_COST
        elif action_b == Action.UNDERCUT:
            delta_R_b += GameRules.UNDERCUT_SELF_RISK
        
        # UNDERCUT LOGIC: Agent A attacking Agent B
        if action_a == Action.UNDERCUT:
            # Check SNAPSHOT risk (pre-action risk) of target
            if snapshot.agent_b.risk > 0:
                # UNDERCUT succeeds
                delta_R_b += GameRules.UNDERCUT_TARGET_RISK_DAMAGE
                delta_C_b += GameRules.UNDERCUT_TARGET_CAPITAL_DAMAGE
            else:
                # UNDERCUT backfires (target has no risk exposure)
                # Note: At Turn 0, initial risk is 0, so UNDERCUT always backfires
                delta_R_a += GameRules.UNDERCUT_BACKFIRE_RISK_PENALTY
        
        # UNDERCUT LOGIC: Agent B attacking Agent A
        if action_b == Action.UNDERCUT:
            # Check SNAPSHOT risk (pre-action risk) of target
            if snapshot.agent_a.risk > 0:
                # UNDERCUT succeeds
                delta_R_a += GameRules.UNDERCUT_TARGET_RISK_DAMAGE
                delta_C_a += GameRules.UNDERCUT_TARGET_CAPITAL_DAMAGE
            else:
                # UNDERCUT backfires (target has no risk exposure)
                delta_R_b += GameRules.UNDERCUT_BACKFIRE_RISK_PENALTY
        
        # ===================================================================
        # PHASE 3: APPLY DELTAS & QUEUE SHIFT
        # ===================================================================
        
        # Apply liquidity, risk, and capital deltas
        self.state.agent_a.liquidity += delta_L_a
        self.state.agent_a.risk += delta_R_a
        self.state.agent_a.capital += delta_C_a
        
        self.state.agent_b.liquidity += delta_L_b
        self.state.agent_b.risk += delta_R_b
        self.state.agent_b.capital += delta_C_b
        
        # Process payouts and shift pending_inflows queue for Agent A
        # CRITICAL ORDER: 
        #   1. Payout P1 to capital
        #   2. Shift P2->P1, P3->P2
        #   3. Place new loan at P3
        current_p1_a = self.state.agent_a.pending_inflows[0]
        current_p2_a = self.state.agent_a.pending_inflows[1]
        current_p3_a = self.state.agent_a.pending_inflows[2]
        
        self.state.agent_a.capital += current_p1_a
        self.state.agent_a.pending_inflows[0] = current_p2_a
        self.state.agent_a.pending_inflows[1] = current_p3_a
        self.state.agent_a.pending_inflows[2] = new_loan_inflow_a
        
        # Process payouts and shift pending_inflows queue for Agent B
        current_p1_b = self.state.agent_b.pending_inflows[0]
        current_p2_b = self.state.agent_b.pending_inflows[1]
        current_p3_b = self.state.agent_b.pending_inflows[2]
        
        self.state.agent_b.capital += current_p1_b
        self.state.agent_b.pending_inflows[0] = current_p2_b
        self.state.agent_b.pending_inflows[1] = current_p3_b
        self.state.agent_b.pending_inflows[2] = new_loan_inflow_b
        
        # ===================================================================
        # PHASE 4: CLAMPING
        # ===================================================================
        
        # Liquidity cannot be negative
        self.state.agent_a.liquidity = max(0, self.state.agent_a.liquidity)
        self.state.agent_b.liquidity = max(0, self.state.agent_b.liquidity)
        
        # Risk cannot be negative
        self.state.agent_a.risk = max(0, self.state.agent_a.risk)
        self.state.agent_b.risk = max(0, self.state.agent_b.risk)
        
        # Capital is NOT clamped (can go negative, leading to bankruptcy)
        
        # ===================================================================
        # PHASE 5: TURN INCREMENT & TERMINATION
        # ===================================================================
        
        # Increment turn counter
        self.state.turn += 1
        
        # Check failure conditions (use post-clamp, post-payout values)
        agent_a_failed = (
            self.state.agent_a.risk >= GameRules.RISK_THRESHOLD or
            self.state.agent_a.capital <= 0
        )
        agent_b_failed = (
            self.state.agent_b.risk >= GameRules.RISK_THRESHOLD or
            self.state.agent_b.capital <= 0
        )
        
        # Check time limit
        time_limit_reached = self.state.turn >= GameRules.MAX_TURNS
        
        # Compute rewards and termination
        reward_a = GameRules.REWARD_STEP
        reward_b = GameRules.REWARD_STEP
        done = False
        info: Dict[str, Any] = {
            "action_a": action_a.name,
            "action_b": action_b.name,
            "turn": self.state.turn,
        }
        
        if agent_a_failed or agent_b_failed or time_limit_reached:
            done = True
            self.state.done = True
            
            if agent_a_failed and agent_b_failed:
                # Mutual failure: Draw
                reward_a = GameRules.REWARD_DRAW
                reward_b = GameRules.REWARD_DRAW
                info["outcome"] = "draw_mutual_failure"
            elif agent_a_failed:
                # Agent A failed, Agent B wins
                reward_a = GameRules.REWARD_LOSS
                reward_b = GameRules.REWARD_WIN
                info["outcome"] = "agent_b_wins"
                info["failure_reason_a"] = self._get_failure_reason(self.state.agent_a)
            elif agent_b_failed:
                # Agent B failed, Agent A wins
                reward_a = GameRules.REWARD_WIN
                reward_b = GameRules.REWARD_LOSS
                info["outcome"] = "agent_a_wins"
                info["failure_reason_b"] = self._get_failure_reason(self.state.agent_b)
            else:
                # Time limit reached, both survived
                # Winner determined by capital (or draw if equal)
                if self.state.agent_a.capital > self.state.agent_b.capital:
                    reward_a = GameRules.REWARD_WIN
                    reward_b = GameRules.REWARD_LOSS
                    info["outcome"] = "agent_a_wins_capital"
                elif self.state.agent_b.capital > self.state.agent_a.capital:
                    reward_a = GameRules.REWARD_LOSS
                    reward_b = GameRules.REWARD_WIN
                    info["outcome"] = "agent_b_wins_capital"
                else:
                    reward_a = GameRules.REWARD_DRAW
                    reward_b = GameRules.REWARD_DRAW
                    info["outcome"] = "draw_equal_capital"
        
        return self.state.copy(), reward_a, reward_b, done, info
    
    def _get_failure_reason(self, agent_state: AgentState) -> str:
        """Determine why an agent failed."""
        if agent_state.risk >= GameRules.RISK_THRESHOLD:
            return "default_risk_threshold"
        elif agent_state.capital <= 0:
            return "bankruptcy_capital_depleted"
        else:
            return "unknown"
