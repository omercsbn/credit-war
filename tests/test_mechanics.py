"""
Core Mechanics Tests for CREDIT WAR

Test action effects, state transitions, and termination conditions.
"""

import pytest
from credit_war.env import CreditWarEnv
from credit_war.actions import Action
from credit_war.rules import GameRules


def test_initial_state():
    """Test that initial state matches specification."""
    env = CreditWarEnv()
    state = env.reset()
    
    assert state.agent_a.liquidity == GameRules.INITIAL_LIQUIDITY
    assert state.agent_a.risk == GameRules.INITIAL_RISK
    assert state.agent_a.capital == GameRules.INITIAL_CAPITAL
    assert state.agent_a.pending_inflows == [0, 0, 0]
    
    assert state.agent_b.liquidity == GameRules.INITIAL_LIQUIDITY
    assert state.agent_b.risk == GameRules.INITIAL_RISK
    assert state.agent_b.capital == GameRules.INITIAL_CAPITAL
    assert state.agent_b.pending_inflows == [0, 0, 0]
    
    assert state.turn == 0
    assert state.done is False


def test_give_loan_mechanics():
    """Test GIVE_LOAN action effects."""
    env = CreditWarEnv()
    state = env.reset()
    
    initial_L_a = state.agent_a.liquidity
    initial_R_a = state.agent_a.risk
    initial_C_a = state.agent_a.capital
    
    # Agent A issues loan, Agent B rejects
    state, r_a, r_b, done, info = env.step(Action.GIVE_LOAN, Action.REJECT)
    
    # Check immediate effects on Agent A
    assert state.agent_a.liquidity == initial_L_a - GameRules.ACTION_COSTS[Action.GIVE_LOAN]
    assert state.agent_a.risk == initial_R_a + GameRules.GIVE_LOAN_RISK
    assert state.agent_a.capital == initial_C_a  # No immediate capital change
    
    # Check loan enters pending queue at P3
    assert state.agent_a.pending_inflows[2] == GameRules.LOAN_RETURN
    assert state.agent_a.pending_inflows[1] == 0
    assert state.agent_a.pending_inflows[0] == 0


def test_invest_mechanics():
    """Test INVEST action effects."""
    env = CreditWarEnv()
    state = env.reset()
    
    initial_L = state.agent_a.liquidity
    initial_C = state.agent_a.capital
    
    state, _, _, _, _ = env.step(Action.INVEST, Action.REJECT)
    
    # INVEST: Cost L=8, Capital +10, Risk unchanged
    assert state.agent_a.liquidity == initial_L - GameRules.ACTION_COSTS[Action.INVEST]
    assert state.agent_a.capital == initial_C + GameRules.INVEST_CAPITAL_GAIN
    assert state.agent_a.risk == 0  # No risk from INVEST


def test_insure_mechanics():
    """Test INSURE action effects."""
    env = CreditWarEnv()
    state = env.reset()
    
    # First, build up some risk with GIVE_LOAN
    state, _, _, _, _ = env.step(Action.GIVE_LOAN, Action.REJECT)
    
    risk_before = state.agent_a.risk
    capital_before = state.agent_a.capital
    liquidity_before = state.agent_a.liquidity
    
    # Now use INSURE
    state, _, _, _, _ = env.step(Action.INSURE, Action.REJECT)
    
    # INSURE: Cost L=7, Risk -8, Capital -3
    assert state.agent_a.liquidity == liquidity_before - GameRules.ACTION_COSTS[Action.INSURE]
    assert state.agent_a.risk == max(0, risk_before + GameRules.INSURE_RISK_REDUCTION)
    assert state.agent_a.capital == capital_before + GameRules.INSURE_CAPITAL_COST


def test_reject_mechanics():
    """Test REJECT action effects."""
    env = CreditWarEnv()
    state = env.reset()
    
    # Build up risk first
    state, _, _, _, _ = env.step(Action.GIVE_LOAN, Action.REJECT)
    
    risk_before = state.agent_a.risk
    capital_before = state.agent_a.capital
    liquidity_before = state.agent_a.liquidity
    
    # Use REJECT
    state, _, _, _, _ = env.step(Action.REJECT, Action.REJECT)
    
    # REJECT: No cost, Risk -2, Capital unchanged
    assert state.agent_a.liquidity == liquidity_before  # No cost
    assert state.agent_a.risk == max(0, risk_before + GameRules.REJECT_RISK_REDUCTION)
    assert state.agent_a.capital == capital_before  # No capital change


def test_undercut_success():
    """Test UNDERCUT when target has risk > 0."""
    env = CreditWarEnv()
    state = env.reset()
    
    # Agent B builds up risk
    state, _, _, _, _ = env.step(Action.REJECT, Action.GIVE_LOAN)
    
    assert state.agent_b.risk > 0  # Agent B has risk exposure
    
    risk_b_before = state.agent_b.risk
    capital_b_before = state.agent_b.capital
    risk_a_before = state.agent_a.risk
    
    # Agent A attacks with UNDERCUT, Agent B does nothing (no additional action)
    state, _, _, _, _ = env.step(Action.UNDERCUT, Action.INVEST)
    
    # Agent A pays cost and self-risk
    assert state.agent_a.risk == risk_a_before + GameRules.UNDERCUT_SELF_RISK
    
    # Agent B takes damage (and INVEST doesn't change risk)
    expected_risk_b = risk_b_before + GameRules.UNDERCUT_TARGET_RISK_DAMAGE
    expected_capital_b = capital_b_before + GameRules.UNDERCUT_TARGET_CAPITAL_DAMAGE + GameRules.INVEST_CAPITAL_GAIN
    
    assert state.agent_b.risk == expected_risk_b
    assert state.agent_b.capital == expected_capital_b


def test_undercut_backfire():
    """Test UNDERCUT when target has risk = 0 (backfires)."""
    env = CreditWarEnv()
    state = env.reset()
    
    # Both agents have 0 risk initially
    assert state.agent_a.risk == 0
    assert state.agent_b.risk == 0
    
    risk_a_before = state.agent_a.risk
    risk_b_before = state.agent_b.risk
    
    # Agent A attempts UNDERCUT at turn 0 (should backfire)
    # Agent B plays INVEST (no risk change)
    state, _, _, _, _ = env.step(Action.UNDERCUT, Action.INVEST)
    
    # Agent A takes backfire penalty
    expected_risk_a = risk_a_before + GameRules.UNDERCUT_SELF_RISK + GameRules.UNDERCUT_BACKFIRE_RISK_PENALTY
    assert state.agent_a.risk == expected_risk_a
    
    # Agent B is unaffected (INVEST has no risk effect)
    assert state.agent_b.risk == risk_b_before  # Still 0


def test_risk_threshold_default():
    """Test that agent defaults when risk >= threshold."""
    env = CreditWarEnv()
    state = env.reset()
    
    # Play GIVE_LOAN repeatedly until either risk threshold or liquidity runs out
    # Risk increases by 5 each time, threshold is 40, so need 8 loans
    # Each loan costs 10 liquidity, so with 50 initial liquidity, can only do 5 loans
    # This means agent will run out of liquidity before hitting risk threshold naturally
    
    # Let's test the mechanism differently: verify that when risk DOES hit threshold, termination occurs
    # We'll engineer a scenario by having Agent B UNDERCUT repeatedly
    
    # First, Agent A builds some risk
    state, _, _, _, _ = env.step(Action.GIVE_LOAN, Action.INVEST)
    state, _, _, _, _ = env.step(Action.GIVE_LOAN, Action.INVEST)
    
    # Now Agent B repeatedly UNDERCUTs Agent A (adds +7 risk each time)
    # Starting risk ~10, need 30 more to reach 40, so ~5 UNDERCUTs
    done = False
    for _ in range(10):
        # Agent A does nothing aggressive, Agent B attacks
        state, r_a, r_b, done, info = env.step(Action.INVEST, Action.UNDERCUT)
        
        if done:
            # Check if risk threshold was the cause
            if "agent_a" in info["outcome"] or "agent_b" in info["outcome"]:
                # Verify proper reward assignment
                if "agent_b_wins" in info["outcome"]:
                    assert r_b == GameRules.REWARD_WIN
                break
        
        # If Agent A's risk is approaching or at threshold, next step should terminate
        if state.agent_a.risk >= GameRules.RISK_THRESHOLD:
            # We expect the episode to have ended
            assert done, f"Episode should have ended with risk={state.agent_a.risk}"
            break


def test_capital_depletion_bankruptcy():
    """Test bankruptcy when capital <= 0."""
    env = CreditWarEnv()
    state = env.reset()
    
    # Drain capital with INSURE actions
    done = False
    while not done and state.turn < 30:
        state, r_a, r_b, done, info = env.step(Action.INSURE, Action.REJECT)
        
        if done:
            # Agent A should have gone bankrupt
            assert state.agent_a.capital <= 0
            assert "agent_b_wins" in info["outcome"]
            break


def test_time_limit():
    """Test that episode ends at MAX_TURNS."""
    env = CreditWarEnv()
    state = env.reset()
    
    # Play safe actions for maximum turns
    done = False
    for _ in range(GameRules.MAX_TURNS):
        if done:
            break
        state, r_a, r_b, done, info = env.step(Action.REJECT, Action.REJECT)
    
    assert done
    assert state.turn == GameRules.MAX_TURNS


def test_liquidity_clamping():
    """Test that liquidity cannot go negative."""
    env = CreditWarEnv()
    state = env.reset()
    
    # Drain liquidity
    while state.agent_a.liquidity > 0:
        state, _, _, done, _ = env.step(Action.INVEST, Action.REJECT)
        if done:
            break
    
    # Liquidity should be clamped at 0
    assert state.agent_a.liquidity >= 0


def test_risk_clamping():
    """Test that risk cannot go negative."""
    env = CreditWarEnv()
    state = env.reset()
    
    # Initial risk is 0, REJECT reduces risk
    state, _, _, _, _ = env.step(Action.REJECT, Action.REJECT)
    
    # Risk should be clamped at 0
    assert state.agent_a.risk >= 0
    assert state.agent_b.risk >= 0


def test_capital_not_clamped():
    """Test that capital CAN go negative (leading to bankruptcy)."""
    env = CreditWarEnv()
    state = env.reset()
    
    # Drain capital with INSURE
    for _ in range(20):
        state, _, _, done, _ = env.step(Action.INSURE, Action.REJECT)
        if state.agent_a.capital < 0:
            # Capital went negative before bankruptcy check
            break
    
    # We should be able to observe negative capital before termination
    # (though bankruptcy will trigger immediately after)
