# CREDIT WAR Test Suite

This directory contains comprehensive tests for the CREDIT WAR environment.

## Test Modules

### test_determinism.py
Tests that verify full determinism and reproducibility:
- `test_state_reproducibility`: Identical action sequences produce identical states
- `test_random_agent_determinism`: Seeded random agents behave identically
- `test_valid_actions_ordering`: Valid actions always returned in same order
- `test_no_randomness_in_transitions`: State transitions are deterministic

### test_mechanics.py
Tests for core environment mechanics:
- `test_initial_state`: Verify initial conditions
- `test_give_loan_mechanics`: GIVE_LOAN action effects
- `test_invest_mechanics`: INVEST action effects
- `test_insure_mechanics`: INSURE action effects
- `test_reject_mechanics`: REJECT action effects
- `test_undercut_success`: UNDERCUT when target has risk
- `test_undercut_backfire`: UNDERCUT when target has no risk
- `test_risk_threshold_default`: DEFAULT triggered at Risk≥40
- `test_capital_depletion_bankruptcy`: BANKRUPTCY at Capital≤0
- `test_time_limit`: Episode ends at MAX_TURNS
- `test_liquidity_clamping`: Liquidity floored at 0
- `test_risk_clamping`: Risk floored at 0
- `test_capital_not_clamped`: Capital can go negative

### test_payouts.py
Critical tests for timing and interactions:
- `test_payout_timing`: Loan issued at turn t pays out at t+3
- `test_payout_timing_multiple_loans`: Multiple loans in queue
- `test_simultaneous_undercut`: Both agents UNDERCUT with Risk>0
- `test_turn_zero_backfire`: UNDERCUT at Turn 0 backfires
- `test_undercut_one_way`: Asymmetric UNDERCUT scenario
- `test_order_of_operations_snapshot`: UNDERCUT uses pre-action risk
- `test_invalid_action_override`: Invalid actions become REJECT

## Running Tests

```bash
# All tests with verbose output
pytest tests/ -v

# Single test module
pytest tests/test_determinism.py -v

# Single test function
pytest tests/test_payouts.py::test_payout_timing -v

# With coverage report
pytest tests/ --cov=credit_war --cov-report=html
```

## Test Design Principles

1. **Isolation**: Each test is independent and can run in any order
2. **Determinism**: All tests use fixed seeds for reproducibility
3. **Clarity**: Test names clearly describe what is being tested
4. **Comprehensiveness**: Critical game mechanics have multiple test cases
5. **Documentation**: Each test includes docstring explaining purpose

## Critical Test Cases

The following tests are CRITICAL for verifying correct implementation:

✅ **test_payout_timing**: Ensures loan payout happens exactly at turn t+3  
✅ **test_simultaneous_undercut**: Verifies parallel interaction mechanics  
✅ **test_turn_zero_backfire**: Confirms UNDERCUT backfire at initial state  
✅ **test_order_of_operations_snapshot**: Validates Phase 2 snapshot usage

These tests directly correspond to the requirements in the implementation specification.
