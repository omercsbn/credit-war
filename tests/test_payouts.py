"""
Payout Timing and Critical Interaction Tests for CREDIT WAR

These tests verify the exact mechanics of delayed loan payouts,
simultaneous UNDERCUT interactions, and Turn 0 backfire behavior.
"""

import pytest
from credit_war.env import CreditWarEnv
from credit_war.actions import Action
from credit_war.rules import GameRules


def test_payout_timing():
    """
    CRITICAL TEST: Verify loan payout timing.
    
    A loan issued at turn t should:
    - Turn t: Enter P3 (pending_inflows[2])
    - Turn t+1: Move to P2 (pending_inflows[1])
    - Turn t+2: Move to P1 (pending_inflows[0])
    - Turn t+3: Pay out to Capital (pending_inflows becomes [0,0,0] for that loan)
    """
    env = CreditWarEnv()
    state = env.reset()
    
    initial_capital = state.agent_a.capital
    
    # Turn 0: Agent A issues loan
    assert state.turn == 0
    state, _, _, _, _ = env.step(Action.GIVE_LOAN, Action.REJECT)
    
    # After Turn 0 (now Turn 1):
    # Loan should be in P3, P2 and P1 should be 0
    assert state.turn == 1
    assert state.agent_a.pending_inflows[2] == GameRules.LOAN_RETURN  # P3 = 15
    assert state.agent_a.pending_inflows[1] == 0  # P2 = 0
    assert state.agent_a.pending_inflows[0] == 0  # P1 = 0
    assert state.agent_a.capital == initial_capital  # No payout yet
    
    # Turn 1: Agent A plays REJECT (no new loan)
    state, _, _, _, _ = env.step(Action.REJECT, Action.REJECT)
    
    # After Turn 1 (now Turn 2):
    # Loan should have shifted: P3->P2->P1->Capital
    # P1 gets old P2 (0), P2 gets old P3 (15), P3 gets new loan (0)
    assert state.turn == 2
    assert state.agent_a.pending_inflows[2] == 0  # P3 = 0 (no new loan)
    assert state.agent_a.pending_inflows[1] == GameRules.LOAN_RETURN  # P2 = 15 (shifted from P3)
    assert state.agent_a.pending_inflows[0] == 0  # P1 = 0 (shifted from P2)
    assert state.agent_a.capital == initial_capital  # Still no payout (P1 was 0)
    
    # Turn 2: Agent A plays REJECT
    state, _, _, _, _ = env.step(Action.REJECT, Action.REJECT)
    
    # After Turn 2 (now Turn 3):
    # P1 gets old P2 (15), P2 gets old P3 (0), P3 gets new loan (0)
    assert state.turn == 3
    assert state.agent_a.pending_inflows[2] == 0  # P3 = 0
    assert state.agent_a.pending_inflows[1] == 0  # P2 = 0 (shifted from P3)
    assert state.agent_a.pending_inflows[0] == GameRules.LOAN_RETURN  # P1 = 15 (shifted from P2)
    assert state.agent_a.capital == initial_capital  # Still no payout (P1 was 0 before shift)
    
    # Turn 3: Agent A plays REJECT
    state, _, _, _, _ = env.step(Action.REJECT, Action.REJECT)
    
    # After Turn 3 (now Turn 4):
    # The loan issued at Turn 0 should NOW pay out (3 turns later)
    # Capital should increase by the P1 value from before the shift
    assert state.turn == 4
    assert state.agent_a.pending_inflows[2] == 0
    assert state.agent_a.pending_inflows[1] == 0
    assert state.agent_a.pending_inflows[0] == 0  # P1 = 0 (shifted from P2)
    assert state.agent_a.capital == initial_capital + GameRules.LOAN_RETURN  # PAYOUT!


def test_payout_timing_multiple_loans():
    """Test queue behavior with multiple loans."""
    env = CreditWarEnv()
    state = env.reset()
    
    initial_capital = state.agent_a.capital
    
    # Turn 0: Issue loan
    state, _, _, _, _ = env.step(Action.GIVE_LOAN, Action.REJECT)
    assert state.agent_a.pending_inflows == [0, 0, 15]
    
    # Turn 1: Issue another loan
    state, _, _, _, _ = env.step(Action.GIVE_LOAN, Action.REJECT)
    # P1=0, P2=15 (shifted from old P3), P3=15 (new loan)
    assert state.agent_a.pending_inflows == [0, 15, 15]
    
    # Turn 2: No loan
    state, _, _, _, _ = env.step(Action.REJECT, Action.REJECT)
    # P1=15, P2=15, P3=0
    assert state.agent_a.pending_inflows == [15, 15, 0]
    
    # Turn 3: No loan (first loan pays out)
    state, _, _, _, _ = env.step(Action.REJECT, Action.REJECT)
    # Capital += old P1 (15), then shift: P1=15, P2=0, P3=0
    assert state.agent_a.capital == initial_capital + 15
    assert state.agent_a.pending_inflows == [15, 0, 0]
    
    # Turn 4: No loan (second loan pays out)
    state, _, _, _, _ = env.step(Action.REJECT, Action.REJECT)
    # Capital += old P1 (15), then shift: P1=0, P2=0, P3=0
    assert state.agent_a.capital == initial_capital + 30
    assert state.agent_a.pending_inflows == [0, 0, 0]


def test_simultaneous_undercut():
    """
    CRITICAL TEST: Both agents UNDERCUT while both have Risk > 0.
    
    Expected behavior:
    - Both pay UNDERCUT cost (L -= 5)
    - Both take self-risk (+3)
    - Both take damage from opponent (+7 Risk, -10 Capital)
    """
    env = CreditWarEnv()
    state = env.reset()
    
    # Both agents build risk with GIVE_LOAN
    state, _, _, _, _ = env.step(Action.GIVE_LOAN, Action.GIVE_LOAN)
    
    assert state.agent_a.risk > 0
    assert state.agent_b.risk > 0
    
    liquidity_a_before = state.agent_a.liquidity
    liquidity_b_before = state.agent_b.liquidity
    risk_a_before = state.agent_a.risk
    risk_b_before = state.agent_b.risk
    capital_a_before = state.agent_a.capital
    capital_b_before = state.agent_b.capital
    
    # Both agents UNDERCUT simultaneously
    state, _, _, _, _ = env.step(Action.UNDERCUT, Action.UNDERCUT)
    
    # Agent A effects:
    # - Liquidity: -5 (cost)
    # - Risk: +3 (self-risk) +7 (damaged by B) = +10
    # - Capital: -10 (damaged by B)
    expected_L_a = liquidity_a_before - GameRules.ACTION_COSTS[Action.UNDERCUT]
    expected_R_a = risk_a_before + GameRules.UNDERCUT_SELF_RISK + GameRules.UNDERCUT_TARGET_RISK_DAMAGE
    expected_C_a = capital_a_before + GameRules.UNDERCUT_TARGET_CAPITAL_DAMAGE
    
    assert state.agent_a.liquidity == expected_L_a
    assert state.agent_a.risk == expected_R_a
    assert state.agent_a.capital == expected_C_a
    
    # Agent B effects (symmetric)
    expected_L_b = liquidity_b_before - GameRules.ACTION_COSTS[Action.UNDERCUT]
    expected_R_b = risk_b_before + GameRules.UNDERCUT_SELF_RISK + GameRules.UNDERCUT_TARGET_RISK_DAMAGE
    expected_C_b = capital_b_before + GameRules.UNDERCUT_TARGET_CAPITAL_DAMAGE
    
    assert state.agent_b.liquidity == expected_L_b
    assert state.agent_b.risk == expected_R_b
    assert state.agent_b.capital == expected_C_b


def test_turn_zero_backfire():
    """
    CRITICAL TEST: UNDERCUT at Turn 0 should backfire.
    
    At Turn 0, both agents have Risk = 0 (initial state).
    UNDERCUT requires target Risk > 0, so it should backfire.
    """
    env = CreditWarEnv()
    state = env.reset()
    
    # Verify initial conditions
    assert state.turn == 0
    assert state.agent_a.risk == 0
    assert state.agent_b.risk == 0
    
    liquidity_a_before = state.agent_a.liquidity
    liquidity_b_before = state.agent_b.liquidity
    risk_a_before = state.agent_a.risk
    risk_b_before = state.agent_b.risk
    capital_a_before = state.agent_a.capital
    capital_b_before = state.agent_b.capital
    
    # Agent A attempts UNDERCUT, Agent B plays REJECT
    state, _, _, _, _ = env.step(Action.UNDERCUT, Action.REJECT)
    
    # Agent A effects (UNDERCUT backfires):
    # - Liquidity: -5 (cost)
    # - Risk: +3 (self-risk) +5 (backfire penalty) = +8
    # - Capital: unchanged (no damage dealt)
    expected_L_a = liquidity_a_before - GameRules.ACTION_COSTS[Action.UNDERCUT]
    expected_R_a = risk_a_before + GameRules.UNDERCUT_SELF_RISK + GameRules.UNDERCUT_BACKFIRE_RISK_PENALTY
    expected_C_a = capital_a_before
    
    assert state.agent_a.liquidity == expected_L_a
    assert state.agent_a.risk == expected_R_a
    assert state.agent_a.capital == expected_C_a
    
    # Agent B effects (REJECT only, not damaged by failed UNDERCUT):
    # - Liquidity: 0 (no cost)
    # - Risk: -2 (REJECT reduces risk, but floored at 0)
    # - Capital: unchanged
    expected_L_b = liquidity_b_before
    expected_R_b = max(0, risk_b_before + GameRules.REJECT_RISK_REDUCTION)
    expected_C_b = capital_b_before
    
    assert state.agent_b.liquidity == expected_L_b
    assert state.agent_b.risk == expected_R_b
    assert state.agent_b.capital == expected_C_b


def test_undercut_one_way():
    """Test UNDERCUT when only one agent has risk."""
    env = CreditWarEnv()
    state = env.reset()
    
    # Agent A builds risk, Agent B stays at 0
    state, _, _, _, _ = env.step(Action.GIVE_LOAN, Action.REJECT)
    
    assert state.agent_a.risk > 0
    assert state.agent_b.risk == 0
    
    risk_a_before = state.agent_a.risk
    capital_a_before = state.agent_a.capital
    risk_b_before = state.agent_b.risk
    
    # Agent B attacks Agent A with UNDERCUT
    # Agent A attacks Agent B (should backfire)
    state, _, _, _, _ = env.step(Action.UNDERCUT, Action.UNDERCUT)
    
    # Agent A:
    # - Takes damage from B's successful UNDERCUT (+7 risk, -10 capital)
    # - Self-risk from own UNDERCUT (+3)
    # - Backfire penalty because B has 0 risk (+5)
    expected_R_a = risk_a_before + GameRules.UNDERCUT_SELF_RISK + GameRules.UNDERCUT_BACKFIRE_RISK_PENALTY + GameRules.UNDERCUT_TARGET_RISK_DAMAGE
    expected_C_a = capital_a_before + GameRules.UNDERCUT_TARGET_CAPITAL_DAMAGE
    
    assert state.agent_a.risk == expected_R_a
    assert state.agent_a.capital == expected_C_a
    
    # Agent B:
    # - Self-risk from own UNDERCUT (+3)
    # - No damage from A's failed UNDERCUT
    expected_R_b = risk_b_before + GameRules.UNDERCUT_SELF_RISK
    
    assert state.agent_b.risk == expected_R_b


def test_order_of_operations_snapshot():
    """
    Test that UNDERCUT damage is computed using PRE-ACTION risk values.
    
    If Agent A has Risk=5 and plays GIVE_LOAN (Risk becomes 10),
    and Agent B plays UNDERCUT, the damage should be based on the
    SNAPSHOT risk (5), not the post-action risk (10).
    """
    env = CreditWarEnv()
    state = env.reset()
    
    # Agent A builds initial risk
    state, _, _, _, _ = env.step(Action.GIVE_LOAN, Action.REJECT)
    
    snapshot_risk_a = state.agent_a.risk  # This is the risk before next action
    
    # Agent A issues another loan (will increase risk)
    # Agent B attacks with UNDERCUT (should use snapshot risk)
    state, _, _, _, _ = env.step(Action.GIVE_LOAN, Action.UNDERCUT)
    
    # Agent A's risk should be:
    # - Snapshot risk
    # - + GIVE_LOAN risk (+5)
    # - + UNDERCUT damage (+7, based on snapshot having risk > 0)
    expected_risk_a = snapshot_risk_a + GameRules.GIVE_LOAN_RISK + GameRules.UNDERCUT_TARGET_RISK_DAMAGE
    
    assert state.agent_a.risk == expected_risk_a


def test_invalid_action_override():
    """Test that invalid actions (insufficient liquidity) are overridden to REJECT."""
    env = CreditWarEnv()
    state = env.reset()
    
    # Drain Agent A's liquidity
    while state.agent_a.liquidity >= GameRules.ACTION_COSTS[Action.GIVE_LOAN]:
        state, _, _, done, _ = env.step(Action.GIVE_LOAN, Action.REJECT)
        if done:
            break
    
    # Agent A now has insufficient liquidity for GIVE_LOAN
    liquidity_before = state.agent_a.liquidity
    assert liquidity_before < GameRules.ACTION_COSTS[Action.GIVE_LOAN]
    
    # Attempt GIVE_LOAN (should be overridden to REJECT)
    state, _, _, _, info = env.step(Action.GIVE_LOAN, Action.REJECT)
    
    # Agent A's liquidity should be unchanged (REJECT has 0 cost)
    # (It might change due to payout, so we check it didn't decrease by GIVE_LOAN cost)
    # Instead, verify via action counts or check that no new loan was issued
    assert state.agent_a.pending_inflows[2] == 0  # No new loan in P3
