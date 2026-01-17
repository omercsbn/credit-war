# CREDIT WAR: Multi-Agent Financial Risk Simulation

A deterministic, fully observable, simultaneous-action research environment for studying competitive banking behavior, systemic risk, and emergent coordination in Multi-Agent Reinforcement Learning (MARL).

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit App](https://img.shields.io/badge/demo-streamlit-FF4B4B.svg)](http://localhost:8501)

**🎮 [Try Interactive Demo](http://localhost:8501)** | **📊 [Tournament Results](tournament_results/)** | **📖 [Training Guide](RL_TRAINING_GUIDE.md)**

---

## Overview

CREDIT WAR is a **research-grade simulation** designed for academic work in:
- Multi-Agent Reinforcement Learning (MARL)
- Agent-Based Modeling (ABM)
- Game Theory
- Quantitative Finance and Risk Modeling

Unlike stochastic simulations or data-driven approaches, CREDIT WAR provides a tractable yet strategically complex micro-world where competitive banking behavior, systemic risk accumulation, and emergent coordination can be systematically studied under controlled conditions.

### Key Features

- ✅ **Fully Deterministic**: No random state transitions; perfect reproducibility
- ✅ **Markov Property**: Pending cash flows included in state vector
- ✅ **Strict Order of Operations**: Five-phase execution model with explicit timing
- ✅ **Type-Safe**: Complete type hints for all functions
- ✅ **Comprehensive Tests**: Unit tests for mechanics, payout timing, and interactions
- ✅ **Baseline Agents**: Random, Greedy, Conservative, Rule-Based, and Aggressor agents
- 🤖 **10 Game Theory PPO Agents**: Nash, Tit-for-Tat, Grim Trigger, Bayesian, Predator, Minimax, Switcher, Evolutionary, Zero-Sum, Meta-Learner
- 🏆 **Battle Royale Tournament**: Complete 90-matchup round-robin with ELO ratings
- 🎮 **Interactive Streamlit Demo**: Play against AI with real-time visualization
- 📊 **Publication-Ready Visualizations**: Heatmaps, capital charts, action analysis

---

## Installation

### Quick Setup

```bash
# Clone the repository
git clone https://github.com/omercsbn/credit-war.git
cd CREDITWAR

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -e .
pip install -r requirements_rl.txt

# Install Streamlit for interactive demo
pip install streamlit plotly

# Run tests
pytest tests/ -v

# Launch interactive demo
streamlit run streamlit_app.py
```

**Requirements**:
- Python 3.11+
- numpy, pandas
- stable-baselines3[extra] (for PPO agents)
- gymnasium (OpenAI Gym wrapper)
- streamlit, plotly (for interactive demo)
- matplotlib, seaborn (for tournament visualizations)
- pytest (for testing)

---

## Quick Start

### 🎮 Interactive Demo (Human vs AI)

Play against trained AI agents in your browser:

```bash
streamlit run streamlit_app.py
```

Open http://localhost:8501 and select from 15 opponents:
- **5 Rule-Based**: Random, Greedy, Cautious, Aggressive, Balanced
- **10 Trained PPO Agents**: Nash, Tit-for-Tat, Grim Trigger, Bayesian (Champion 🏆), Predator, Minimax, Switcher, Evolutionary, Zero-Sum, Meta-Learner

**Features**:
- Real-time gameplay with action buttons
- Live capital/risk/liquidity metrics
- Interactive capital history chart (Plotly)
- Action frequency analysis
- Turn-by-turn detailed history

### 🏆 Battle Royale Tournament

Run the complete round-robin tournament (90 matchups, 4,500 episodes):

```bash
python tournament_battle_royale.py --episodes 50
```

**Outputs**:
- `tournament_results/win_rate_heatmap.png` - 14x12" publication-ready heatmap
- `tournament_results/reward_heatmap.png` - Average rewards per matchup
- `tournament_results/elo_ratings.csv` - Final ELO rankings
- `tournament_results/tournament_results.json` - Complete data export

**Champion Results** (from latest tournament):
1. **Bayesian Adaptive** - 1599.5 ELO (77.8% win rate, 350-100-0)
2. **Switcher** - 1577.3 ELO (44.4% win rate, 250 draws)
3. **Evolutionary** - 1567.3 ELO (66.7% win rate)

### Running a Tournament

```bash
# Run 100 episodes between Random and Greedy agents
python -m credit_war.cli --agent-a random --agent-b greedy --episodes 100

# Run tournament with verbose output
python -m credit_war.cli --agent-a conservative --agent-b rulebased --episodes 1000 --verbose
```

### Using the API

```python
from credit_war import CreditWarEnv, Action
from credit_war.agents import RandomAgent, RuleBasedAgent
from credit_war.simulation import SimulationRunner

# Create environment
env = CreditWarEnv(seed=42)

# Create agents
agent_a = RandomAgent(seed=42)
agent_b = RuleBasedAgent(seed=100)

# Run tournament
runner = SimulationRunner(env)
metrics = runner.run_tournament(agent_a, agent_b, num_episodes=100)

print(f"Agent A Win Rate: {metrics.agent_a_wins / metrics.total_episodes:.2%}")
print(f"Agent A Survival Rate: {metrics.agent_a_survival_rate:.2%}")
print(f"Average Episode Length: {metrics.avg_turns:.1f} turns")
```

### Manual Step Execution

```python
from credit_war import CreditWarEnv, Action

env = CreditWarEnv()
state = env.reset()

# Execute one timestep
next_state, reward_a, reward_b, done, info = env.step(
    Action.GIVE_LOAN,  # Agent A action
    Action.INVEST      # Agent B action
)

print(f"Turn: {next_state.turn}")
print(f"Agent A Capital: {next_state.agent_a.capital}")
print(f"Agent A Risk: {next_state.agent_a.risk}")
print(f"Done: {done}")
```

---

## Environment Specification

### State Space

Each agent has a 6-dimensional state vector: `S_i = (L, R, C, P1, P2, P3)`

| Variable | Description | Initial | Range |
|----------|-------------|---------|-------|
| **L** (Liquidity) | Available liquid capital | 50 | [0, 100] |
| **R** (Risk) | Cumulative risk exposure | 0 | [0, 50] |
| **C** (Capital) | Long-term reserves/profit | 50 | ℝ₊ |
| **P1, P2, P3** | Pending loan inflows queue | [0,0,0] | ℤ₊ |

**Global State** (fully observable): `(Agent_A, Agent_B, turn, done)`

### Action Space

Five discrete actions available to each agent:

| Action | Cost (L) | Effect | Strategic Role |
|--------|----------|--------|----------------|
| **GIVE_LOAN** | 10 | R+5, generates delayed +15 Capital | Aggressive profit-seeking |
| **REJECT** | 0 | R-2 | Defensive risk reduction |
| **INVEST** | 8 | C+10 | Safe moderate growth |
| **INSURE** | 7 | R-8, C-3 | Risk mitigation |
| **UNDERCUT** | 5 | R+3, damages opponent if they have risk | Competitive sabotage |

### Transition Dynamics

**Strict Five-Phase Execution**:

1. **PHASE 1**: Snapshot & Validation
   - Create deep copy of current state
   - Validate actions against liquidity
   - Override invalid actions to REJECT

2. **PHASE 2**: Compute Deltas (Parallel)
   - Calculate all state changes based on snapshot
   - Compute UNDERCUT interactions using pre-action risk
   - No state mutation yet

3. **PHASE 3**: Apply Deltas & Queue Shift
   - Apply computed deltas to L, R, C
   - Process payouts: Capital += P1
   - Shift queue: P1←P2, P2←P3, P3←new_loan

4. **PHASE 4**: Clamping
   - Liquidity: max(0, L)
   - Risk: max(0, R)
   - Capital: NOT clamped (can go negative → bankruptcy)

5. **PHASE 5**: Turn Increment & Termination
   - Increment turn counter
   - Check failure conditions (DEFAULT: R≥40, BANKRUPTCY: C≤0)
   - Check time limit (MAX_TURNS = 40)
   - Compute terminal rewards

### Loan Payout Timing

Loans issued at turn `t` pay out at turn `t+3`:

```
Turn 0: GIVE_LOAN → P3=15, P2=0, P1=0
Turn 1: (shift)   → P3=0,  P2=15, P1=0
Turn 2: (shift)   → P3=0,  P2=0,  P1=15
Turn 3: (payout)  → P3=0,  P2=0,  P1=0, Capital += 15
```

### UNDERCUT Mechanics

**Success Condition**: Target has Risk > 0 (uses snapshot risk)

**Effects**:
- Attacker: Cost L=5, Self-risk R+3
- Target: Damage R+7, C-10

**Backfire Condition**: Target has Risk = 0
- Attacker: Additional penalty R+5
- Target: No damage

**Special Case**: At Turn 0, both agents have Risk=0, so UNDERCUT always backfires.

### Termination & Rewards

**Failure Modes**:
- **DEFAULT**: Risk ≥ 40
- **BANKRUPTCY**: Capital ≤ 0

**Rewards** (Sparse):
- Intermediate (not done): 0
- Winner: +1
- Loser: -1
- Draw: 0

**Draw Conditions**:
- Mutual failure
- Time limit reached with equal capital

---

## Baseline Agents

### Random Agent
Selects actions uniformly at random from valid action set. Uses seeded RNG for reproducibility.

### Greedy Agent
Always plays GIVE_LOAN if liquidity permits. Pure profit maximization without risk management.

**Expected**: Rapid capital growth → inevitable DEFAULT

### Conservative Agent
Risk-minimizing strategy:
- If Risk > 15: INSURE
- Else if Liquidity > 20: INVEST
- Else: REJECT

**Expected**: High survival rate, moderate capital

### Rule-Based Agent
Sophisticated heuristic with adaptive behavior:
1. If Risk > 30: INSURE (emergency)
2. If Opponent Risk > 25: UNDERCUT (opportunistic)
3. If Liquidity > 25 and Risk < 20: GIVE_LOAN (aggressive)
4. If Liquidity > 15: INVEST
5. Else: REJECT

**Expected**: Balanced competitive performance

### Aggressor Agent (Opponent Modeling - Adversarial)

**🎯 Critical for MARL Research**: This agent demonstrates **opponent modeling** and **adversarial behavior** - essential for competitive MARL studies.

**Strategy**: Analyzes opponent's weaknesses and exploits them with aggressive UNDERCUT attacks.

**Decision Logic**:
1. **UNDERCUT Priority**: If opponent.risk ≥ 5 AND opponent.capital < 30 AND own.risk < 25 → UNDERCUT (Attack vulnerable opponent)
2. **Self-Preservation**: If own.risk ≥ 25 → INSURE (Protect yourself first)
3. **Liquidity Management**: If own.liquidity < 5 → INVEST (Maintain operational capacity)
4. **Aggressive Growth**: If own.capital ≥ 20 → GIVE_LOAN (Exploit strong position)
5. **Safe Growth**: Otherwise → INVEST (Steady accumulation)

**Why Essential**: 
- ✅ Forces other agents to develop robust survival strategies
- ✅ Makes GreedyAgent vulnerable (exposes its weakness)
- ✅ Creates adversarial environment for MARL training
- ✅ Demonstrates opponent modeling in practice

**Expected**: High win rate against naive agents (Greedy, Random), competitive against sophisticated agents

### PPO Agent (Reinforcement Learning - Learner)

**🤖 The Learning Agent**: Unlike rule-based agents, PPOAgent learns optimal strategies through self-play.

**How it works**:
1. **Neural Network**: 2-layer MLP (256x256 hidden units) processes state
2. **Training**: PPO algorithm with ~1M timesteps (~25k episodes)
3. **Opponent Modeling**: Learns to exploit specific opponent weaknesses
4. **Continuous Improvement**: Strategy evolves through experience

**Training Setup**:
```bash
python train_ppo.py --opponent rulebased --timesteps 1000000
```

**Using Trained Model**:
```python
from credit_war.agents import PPOAgent
agent = PPOAgent(model_path="models/ppo_rulebased_final.zip")
```

**Expected Performance**:
- vs Random: 90-95% win rate
- vs Conservative: 70-80% win rate
- vs RuleBased: 60-70% win rate (depends on training)
- vs Aggressor: 55-65% win rate (challenging adversarial opponent)

**Why Essential for Research**:
- ✅ Demonstrates MARL capability
- ✅ Learns emergent strategies not encoded by humans
- ✅ Adapts to opponent behavior
- ✅ Provides baseline for comparing learning algorithms

📖 **Complete training guide**: [RL_TRAINING_GUIDE.md](RL_TRAINING_GUIDE.md)

---

## Game Theory PPO Agents (10 Strategies)

**Custom Reward Shaping**: Each agent has unique reward function based on game theory principles.

### 1. Nash Equilibrium Seeker
**Strategy**: Penalizes dominated strategies, rewards best responses  
**Reward Shaping**: -10 for playing dominated action, +5 for mutual cooperation  
**Tournament**: 1451.5 ELO (middle tier)

### 2. Tit-for-Tat Reciprocator
**Strategy**: Mirrors opponent's previous action  
**Reward Shaping**: +8 for reciprocating cooperation, -5 for not punishing defection  
**Tournament**: 1386.0 ELO (failed - reciprocity doesn't work in simultaneous actions)

### 3. Grim Trigger Enforcer
**Strategy**: Cooperates until opponent defects, then punishes forever  
**Reward Shaping**: +10 for cooperation phase, +5 for permanent retaliation after trigger  
**Tournament**: 1467.0 ELO (moderate success)

### 4. Bayesian Adaptive Bank 🏆
**Strategy**: Tracks opponent action frequencies, exploits patterns  
**Reward Shaping**: +15 for exploiting high-frequency actions, +8 for adaptation  
**Tournament**: **1599.5 ELO (CHAMPION - 77.8% win rate, dominated 7/9 opponents)**

### 5. Predator Hawk
**Strategy**: Aggressively attacks weak opponents  
**Reward Shaping**: +12 for attacking low-capital/high-risk opponents, +5 for finishing blow  
**Tournament**: 1481.5 ELO (solid mid-tier)

### 6. Minimax Guardian
**Strategy**: Minimizes worst-case outcomes  
**Reward Shaping**: +10 for risk reduction when exposed, +5 for defensive plays  
**Tournament**: 1513.9 ELO (defensive stability)

### 7. Switcher Chameleon
**Strategy**: Cycles through strategies to remain unpredictable  
**Reward Shaping**: +8 for strategy diversity, +10 for opponent confusion  
**Tournament**: **1577.3 ELO (#2 - 250/450 draws, highly unpredictable)**

### 8. Evolutionary Fitness Maximizer
**Strategy**: Balances survival with resource accumulation  
**Reward Shaping**: +12 for high fitness (capital * survival probability)  
**Tournament**: 1567.3 ELO (#3 - 66.7% win rate)

### 9. Zero-Sum Competitor
**Strategy**: Focus on relative advantage over opponent  
**Reward Shaping**: +10 for capital gap increase, +5 for opponent damage  
**Tournament**: 1442.2 ELO (backfired - relative focus was detrimental)

### 10. Meta-Learner Strategist
**Strategy**: Learns opponent's strategy type, counter-plays  
**Reward Shaping**: +15 for correct opponent classification, +10 for counter-strategy  
**Tournament**: 1513.8 ELO (moderate adaptive success)

### Training All Agents

```bash
# Train all 10 agents (takes ~2 minutes)
python train_game_theory_agents.py --agents nash titfortat grimtrigger bayesian predator minimax switcher evolutionary zerosum metalearner --timesteps 10000 --seed 42
```

**Implementation**: [credit_war/game_theory_rewards.py](credit_war/game_theory_rewards.py) (750+ lines of custom reward shapers)

---

## Testing

Run the comprehensive test suite:

```bash
# All tests
pytest tests/ -v

# Specific test modules
pytest tests/test_determinism.py -v
pytest tests/test_mechanics.py -v
pytest tests/test_payouts.py -v

# Run with coverage
pytest tests/ --cov=credit_war --cov-report=html
```

### Critical Test Cases

- ✅ **test_payout_timing**: Verifies loan issued at turn t pays out exactly at turn t+3
- ✅ **test_simultaneous_undercut**: Both agents UNDERCUT with Risk>0, both take damage
- ✅ **test_aggressor_integration**: Opponent modeling behavior validation
- ✅ **test_game_theory_rewards**: All 10 custom reward shapers validated

---

## Project Structure

```
CREDITWAR/
├── credit_war/
│   ├── __init__.py          # Package exports
│   ├── actions.py           # Action enum definitions
│   ├── state.py             # State dataclasses (AgentState, GlobalState)
│   ├── rules.py             # Game constants and parameters
│   ├── env.py               # Core environment logic (CreditWarEnv)
│   ├── gym_wrapper.py       # Gymnasium wrapper with reward shaping
│   ├── game_theory_rewards.py  # 10 custom reward shapers (750+ lines)
│   └── agents/
│       ├── __init__.py
│       ├── base.py          # BaseAgent interface
│       ├── random_agent.py
│       ├── greedy_agent.py
│       ├── conservative_agent.py
│       ├── rule_based_agent.py
│       ├── aggressor_agent.py  # Opponent modeling agent
│       └── ppo_agent.py     # PPO learner with model loading
├── simulation.py            # SimulationRunner for metrics
├── cli.py                   # Command-line interface
├── train_ppo.py            # Single agent training script
├── train_game_theory_agents.py  # Batch training for 10 agents
├── evaluate_ppo.py         # Single agent evaluation
├── evaluate_game_theory_agents.py  # Multi-opponent evaluation
├── tournament_battle_royale.py  # Round-robin with ELO & heatmaps
├── streamlit_app.py        # Interactive web demo (450+ lines)
├── models/
│   └── game_theory/        # Trained models for 10 agents
│       ├── nash/
│       ├── titfortat/
│       ├── bayesian/       # Champion model
│       └── ...
├── tournament_results/
│   ├── README.md           # Tournament documentation
│   ├── win_rate_heatmap.png  # 14x12" publication figure
│   ├── reward_heatmap.png
│   ├── elo_ratings.csv
│   └── tournament_results.json
├── tests/
│   ├── test_determinism.py  # Reproducibility tests
│   ├── test_mechanics.py    # Action effects and termination
│   ├── test_payouts.py      # Critical timing and interaction tests
│   └── test_aggressor_agent.py  # Opponent modeling behavior
├── README.md               # This file
├── RL_TRAINING_GUIDE.md    # Comprehensive training documentation
└── requirements_rl.txt     # All dependencies
```

---

## Research Applications

### Tournament Key Findings

**From Battle Royale (4,500 episodes, 90 matchups)**:

**🏆 Success Stories**:
- **Bayesian Adaptive**: Dominated with pattern recognition - 100% win rate vs 7 opponents
- **Switcher**: High unpredictability (250 draws) prevented opponent adaptation
- **Evolutionary**: Balanced fitness function achieved 66.7% overall win rate

**❌ Failed Strategies**:
- **Tit-for-Tat**: Only 11.1% win rate - reciprocity ineffective in simultaneous actions
- **Zero-Sum**: Last place - focusing on relative advantage was counterproductive
- **Nash Equilibrium**: Middle tier - theoretical optimality didn't translate to practice

**📊 Statistical Insights**:
- Opponent modeling (Bayesian, Meta-Learner) outperformed fixed strategies
- Unpredictability (Switcher) more effective than pure aggression (Predator)
- Risk management critical - aggressive agents with poor risk control (Greedy, Zero-Sum) failed
- Adaptive strategies dominated non-adaptive ones by 2:1 margin

**Publication-Ready Visualizations**: [tournament_results/](tournament_results/)

### Master's Thesis (3-6 months)
- Implement CREDIT WAR environment ✅
- Train baseline MARL algorithms (PPO, A3C) ✅
- **10 game theory PPO agents with custom reward shaping** ✅
- **Battle Royale tournament (90 matchups, 4,500 episodes)** ✅
- **Interactive Streamlit demo for portfolio showcase** ✅
- Compare learned vs. rule-based strategies ✅
- Analyze emergent behavior patterns ✅
- **Publication-ready heatmaps and ELO analysis** ✅

### PhD Research (1-2 years)
- Comprehensive algorithmic comparison (PPO, SAC, MADDPG)
- Theoretical Nash equilibrium analysis
- Extensions: regulatory constraints, N>2 agents, partial observability
- Publication-ready experimental evaluation

### Target Venues
- **Tier 1**: AAMAS, ICML, NeurIPS
- **Tier 2**: JAIR, IEEE Trans. Neural Networks, J. Economic Dynamics & Control

---

## Design Rationale

### Why Deterministic?
- **Reproducibility**: Identical experiments yield identical results
- **Attribution**: Outcomes are solely due to strategic interaction, not environmental noise
- **Analytical Tractability**: Enables formal game-theoretic analysis

### Why Fully Observable?
- **Focus**: Isolate strategic complexity from information asymmetry
- **Baseline**: Establish performance ceiling before introducing partial observability
- **Simplicity**: Reduces implementation and debugging complexity

### Why Sparse Rewards?
- **Long-Horizon Planning**: Forces agents to anticipate delayed consequences
- **Realistic**: Terminal outcomes (bankruptcy, survival) are more meaningful than step rewards
- **Prevents Reward Hacking**: Avoids agents exploiting intermediate reward signals

---

## Extensions

Possible directions for future research:

1. **Regulatory Constraints**: Capital adequacy requirements (Basel III analogues)
2. **Macroeconomic Shocks**: Deterministic regime shifts at fixed timesteps
3. **Asymmetric Information**: Partial observability (hidden opponent risk)
4. **Multi-Agent**: Expand to N>2 agents with network effects
5. **Continuous Actions**: Replace discrete actions with continuous loan/investment amounts

---

## Citation

If you use CREDIT WAR in academic work, please cite:

```bibtex
@misc{creditwar2026,
  title={CREDIT WAR: A Deterministic Strategic Environment for Multi-Agent Reinforcement Learning in Financial Risk Modeling},
  author={Ömercan Sabun},
  year={2026},
  howpublished={\url{https://github.com/omercsbn/credit-war}}
}
```

---

## License

MIT License - See LICENSE file for details.

---

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass: `pytest tests/ -v`
5. Submit a pull request

---

## Contact

For questions, issues, or collaboration inquiries:
- GitHub Issues: [Repository Issues](https://github.com/omercsbn/credit-war/issues)
- Email: omercansabun@icloud.com

---

## Acknowledgments

This environment is designed for academic research in Multi-Agent Reinforcement Learning and Agent-Based Modeling. It draws inspiration from:
- Canonical MARL environments (GridWorld, multi-agent particle envs)
- Financial system ABM literature (Thurner et al., Battiston et al.)
- Game-theoretic models of systemic risk (Acemoglu et al.)

---

**Built for Research. Designed for Rigor. Tested for Determinism.**
