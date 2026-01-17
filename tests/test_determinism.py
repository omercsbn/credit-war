"""
Determinism Tests for CREDIT WAR

Verify that environment is fully deterministic and reproducible.
"""

import pytest
from credit_war.env import CreditWarEnv
from credit_war.actions import Action
from credit_war.agents import RandomAgent


def test_state_reproducibility():
    """Test that identical action sequences produce identical states."""
    
    # Create two independent environments with same seed
    env1 = CreditWarEnv(seed=42)
    env2 = CreditWarEnv(seed=42)
    
    # Reset both
    state1 = env1.reset()
    state2 = env2.reset()
    
    # Verify initial states are identical
    assert state1.agent_a.liquidity == state2.agent_a.liquidity
    assert state1.agent_a.risk == state2.agent_a.risk
    assert state1.agent_a.capital == state2.agent_a.capital
    assert state1.agent_b.liquidity == state2.agent_b.liquidity
    
    # Execute identical action sequences
    actions = [
        (Action.GIVE_LOAN, Action.INVEST),
        (Action.REJECT, Action.INSURE),
        (Action.INVEST, Action.GIVE_LOAN),
        (Action.INSURE, Action.REJECT),
        (Action.UNDERCUT, Action.UNDERCUT)
    ]
    
    for action_a, action_b in actions:
        state1, r1_a, r1_b, done1, info1 = env1.step(action_a, action_b)
        state2, r2_a, r2_b, done2, info2 = env2.step(action_a, action_b)
        
        # Verify states are identical
        assert state1.agent_a.liquidity == state2.agent_a.liquidity
        assert state1.agent_a.risk == state2.agent_a.risk
        assert state1.agent_a.capital == state2.agent_a.capital
        assert state1.agent_a.pending_inflows == state2.agent_a.pending_inflows
        
        assert state1.agent_b.liquidity == state2.agent_b.liquidity
        assert state1.agent_b.risk == state2.agent_b.risk
        assert state1.agent_b.capital == state2.agent_b.capital
        assert state1.agent_b.pending_inflows == state2.agent_b.pending_inflows
        
        assert state1.turn == state2.turn
        assert done1 == done2
        
        # Verify rewards are identical
        assert r1_a == r2_a
        assert r1_b == r2_b


def test_random_agent_determinism():
    """Test that seeded random agents produce identical behavior."""
    
    env1 = CreditWarEnv(seed=100)
    env2 = CreditWarEnv(seed=100)
    
    agent1_a = RandomAgent(seed=100)
    agent1_b = RandomAgent(seed=200)
    
    agent2_a = RandomAgent(seed=100)
    agent2_b = RandomAgent(seed=200)
    
    state1 = env1.reset()
    state2 = env2.reset()
    
    for _ in range(10):
        valid_a1 = env1.get_valid_actions(state1.agent_a)
        valid_b1 = env1.get_valid_actions(state1.agent_b)
        action_a1 = agent1_a.select_action(state1.agent_a, state1.agent_b, valid_a1)
        action_b1 = agent1_b.select_action(state1.agent_b, state1.agent_a, valid_b1)
        
        valid_a2 = env2.get_valid_actions(state2.agent_a)
        valid_b2 = env2.get_valid_actions(state2.agent_b)
        action_a2 = agent2_a.select_action(state2.agent_a, state2.agent_b, valid_a2)
        action_b2 = agent2_b.select_action(state2.agent_b, state2.agent_a, valid_b2)
        
        # Verify agents selected same actions
        assert action_a1 == action_a2
        assert action_b1 == action_b2
        
        state1, _, _, done1, _ = env1.step(action_a1, action_b1)
        state2, _, _, done2, _ = env2.step(action_a2, action_b2)
        
        assert done1 == done2
        if done1:
            break


def test_valid_actions_ordering():
    """Test that valid_actions returns actions in consistent order."""
    
    env = CreditWarEnv()
    state = env.reset()
    
    # Get valid actions multiple times
    valid1 = env.get_valid_actions(state.agent_a)
    valid2 = env.get_valid_actions(state.agent_a)
    valid3 = env.get_valid_actions(state.agent_a)
    
    # Verify same order
    assert valid1 == valid2 == valid3
    
    # Verify sorted by enum order (GIVE_LOAN < REJECT < INVEST < INSURE < UNDERCUT)
    for i in range(len(valid1) - 1):
        assert valid1[i].value < valid1[i + 1].value


def test_no_randomness_in_transitions():
    """Test that no random number generation occurs in state transitions."""
    
    env = CreditWarEnv(seed=999)
    state = env.reset()
    
    # Execute same action pair multiple times from reset state
    results = []
    for _ in range(5):
        env_temp = CreditWarEnv(seed=999)
        state_temp = env_temp.reset()
        state_result, r_a, r_b, done, info = env_temp.step(Action.GIVE_LOAN, Action.INVEST)
        
        results.append({
            'liquidity_a': state_result.agent_a.liquidity,
            'risk_a': state_result.agent_a.risk,
            'capital_a': state_result.agent_a.capital,
            'pending_a': state_result.agent_a.pending_inflows,
            'reward_a': r_a,
            'reward_b': r_b
        })
    
    # All results should be identical
    for i in range(1, len(results)):
        assert results[i] == results[0]
