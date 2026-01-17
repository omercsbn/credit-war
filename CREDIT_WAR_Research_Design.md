# CREDIT WAR: A Deterministic Strategic Environment for Multi-Agent Reinforcement Learning in Financial Risk Modeling

**A Formal Specification and Research Framework**

---

## Abstract

We present CREDIT WAR, a deterministic, fully observable, simultaneous-action, two-agent strategic environment designed for controlled experimentation in Multi-Agent Reinforcement Learning (MARL), Agent-Based Modeling (ABM), and financial risk analysis. Unlike stochastic simulations or data-driven approaches, CREDIT WAR provides a tractable yet strategically complex micro-world where competitive banking behavior, systemic risk accumulation, and emergent coordination can be systematically studied under controlled conditions. This document provides a complete formal specification suitable for academic research, including state-space formalization, deterministic transition dynamics, reward structure design, and experimental protocols.

---

## 1. Problem Motivation and Research Context

### 1.1 The Role of Simplified Environments in Complex Systems Research

Agent-Based Modeling (ABM) and Multi-Agent Reinforcement Learning (MARL) have emerged as critical methodologies for understanding complex adaptive systems, particularly in financial economics and systemic risk modeling. However, real-world financial systems present fundamental challenges for rigorous scientific inquiry:

1. **Opacity and Data Limitations**: Actual banking data is proprietary, incomplete, and reflects confounded influences from regulatory interventions, macroeconomic shocks, and unobserved strategic interactions.

2. **Non-Reproducibility**: Historical financial crises are rare, path-dependent events that cannot be experimentally manipulated or replicated under controlled conditions.

3. **Attribution Problem**: In empirical data, it is nearly impossible to isolate the causal effect of strategic agent behavior from exogenous market forces, regulatory changes, or information asymmetries.

### 1.2 The Case for Micro-World Simulations

Following the tradition of simplified yet strategically rich environments in MARL research (e.g., matrix games, GridWorld, multi-agent particle environments), we argue that **deterministic micro-worlds** serve as essential experimental platforms for:

- **Hypothesis Testing**: Formulating and testing precise theoretical predictions about agent learning, equilibrium behavior, and systemic outcomes.
- **Counterfactual Analysis**: Exploring "what-if" scenarios (e.g., different regulatory constraints, risk thresholds, or market structures) impossible to observe empirically.
- **Algorithm Development**: Benchmarking and validating novel MARL algorithms in settings where ground truth about optimal or equilibrium strategies can be analytically characterized or computationally verified.

### 1.3 Strategic Complexity vs. Environmental Stochasticity

A key design principle of CREDIT WAR is the deliberate elimination of environmental randomness. All complexity emerges from:

- **Strategic Interaction**: Simultaneous decision-making under mutual interdependence
- **Delayed Consequences**: Actions accumulate risk and capital effects over multiple timesteps
- **Non-Stationary Learning**: Each agent faces a moving target as opponents adapt

This design choice ensures that observed behavioral patterns and systemic outcomes are attributable solely to learned strategies rather than environmental noise, enhancing interpretability and reproducibility.

---

## 2. Environment Overview: High-Level Design

CREDIT WAR is formally characterized as a **deterministic, fully observable, simultaneous-move, finite-horizon, two-player stochastic game** with the following properties:

| Property | Specification |
|----------|---------------|
| **Determinism** | State transitions are purely a function of current state and joint actions; no randomness |
| **Observability** | Both agents observe the complete global state $S_t$ at each timestep |
| **Action Space** | Discrete, symmetric action sets for both agents |
| **Time Horizon** | Finite, fixed maximum duration $T_{\text{max}}$ (e.g., 40 timesteps) |
| **Strategic Structure** | Simultaneous action selection; neither agent observes opponent's action before committing |
| **Termination** | Episodes end upon agent failure, mutual default, or time exhaustion |

The environment models a **duopolistic banking market** where two competing financial institutions make sequential strategic decisions regarding loan issuance, investment, risk management, and competitive sabotage.

---

## 3. Agents and Economic Interpretation

### 3.1 Agent Definitions

The environment contains exactly two agents:

- **Agent A**: Representing Bank A
- **Agent B**: Representing Bank B

Each agent is an autonomous decision-making entity that:
- Observes the complete state of the system
- Selects actions to maximize cumulative reward
- Operates under identical rules and constraints (symmetric game)

### 3.2 Economic Variables and Their Interpretations

Each agent $i \in \{A, B\}$ is characterized by three state variables at time $t$:

#### 3.2.1 Liquidity ($L_t^i$)

**Economic Meaning**: Available liquid capital that can be deployed for loans or investments.

**Operational Role**: 
- Consumed when issuing loans or making investments
- Replenished through loan repayments and investment returns
- Represents short-term financial flexibility

**Typical Range**: $L_t^i \in [0, 100]$ (normalized units)

#### 3.2.2 Risk Score ($R_t^i$)

**Economic Meaning**: Cumulative exposure to default risk from the bank's loan portfolio.

**Operational Role**:
- Increases when issuing risky loans
- Triggers DEFAULT condition when exceeding threshold $R_{\text{max}}$
- Represents systemic vulnerability and portfolio fragility
- Can be partially mitigated through INSURE action

**Typical Range**: $R_t^i \in [0, 50]$, with critical threshold at $R_{\text{max}} = 40$

#### 3.2.3 Capital ($C_t^i$)

**Economic Meaning**: Long-term financial reserves and accumulated profit.

**Operational Role**:
- Increases through successful loan repayments and investments
- Determines final performance (primary objective function)
- Provides buffer against negative shocks
- BANKRUPTCY occurs when $C_t^i \leq 0$

**Typical Range**: $C_t^i \in [0, \infty)$, initial endowment $C_0^i = 50$

### 3.3 Duopoly Market Structure

The two-agent setup models a **concentrated banking sector** where:
- Each bank's actions directly affect competitor profitability
- Market share is implicitly contested through loan issuance decisions
- Externalities (via UNDERCUT) create direct strategic interdependence
- Systemic risk can cascade through competitive pressure

This abstraction captures core strategic tensions in oligopolistic financial markets while maintaining analytical tractability.

---

## 4. State Space Formalization

### 4.1 Formal Definition

The environment state at discrete time $t$ is represented as a six-dimensional vector:

$$
S_t = (L_t^A, R_t^A, C_t^A, L_t^B, R_t^B, C_t^B) \in \mathcal{S}
$$

where the state space $\mathcal{S}$ is defined as:

$$
\mathcal{S} = [0, L_{\max}]^2 \times [0, R_{\max}]^2 \times [0, C_{\max}]^2
$$

### 4.2 State Variable Constraints

| Variable | Symbol | Domain | Initial Value | Constraints |
|----------|--------|--------|---------------|-------------|
| Liquidity (Agent A) | $L_t^A$ | $[0, 100]$ | 50 | Non-negative, capped |
| Risk Score (Agent A) | $R_t^A$ | $[0, 50]$ | 0 | Triggers default if $\geq 40$ |
| Capital (Agent A) | $C_t^A$ | $\mathbb{R}_+$ | 50 | Bankruptcy if $\leq 0$ |
| Liquidity (Agent B) | $L_t^B$ | $[0, 100]$ | 50 | Non-negative, capped |
| Risk Score (Agent B) | $R_t^B$ | $[0, 50]$ | 0 | Triggers default if $\geq 40$ |
| Capital (Agent B) | $C_t^B$ | $\mathbb{R}_+$ | 50 | Bankruptcy if $\leq 0$ |

### 4.3 Properties of the State Space

**Compactness**: The state space is bounded in all practical dimensions (liquidity and risk are capped; capital grows but episodes are finite).

**Low Dimensionality**: Six continuous variables provide sufficient expressiveness while remaining tractable for function approximation in deep RL.

**Full Observability**: Both agents observe $S_t$ completely, eliminating partial observability complications and enabling focus on strategic interaction.

**Strategic Richness**: Despite low dimensionality, the state space supports complex multi-step strategies involving risk-return trade-offs, defensive positioning, and competitive sabotage.

---

## 5. Action Space: Strategic Options

### 5.1 Formal Definition

Each agent $i$ selects an action $a_t^i$ from a discrete, finite action space:

$$
\mathcal{A} = \{\text{GIVE\_LOAN}, \text{REJECT}, \text{INVEST}, \text{INSURE}, \text{UNDERCUT}\}
$$

The joint action space is $\mathcal{A}^2$, with $|\mathcal{A}^2| = 25$ possible simultaneous action combinations per timestep.

### 5.2 Action Definitions and Financial Interpretations

#### 5.2.1 GIVE_LOAN

**Financial Interpretation**: Issue a risky loan to a borrower (simplified: modeled as deterministic repayment after delay).

**Strategic Role**: Aggressive profit-seeking action.

**Mechanics**:
- **Cost**: Decreases $L_t^i$ by loan amount (e.g., -10)
- **Risk**: Increases $R_t^i$ by risk increment (e.g., +5)
- **Delayed Benefit**: After fixed delay $\tau$ timesteps (e.g., 3), increases $C_t^i$ by return (e.g., +15)
- **Vulnerability**: Exposed to opponent's UNDERCUT action during loan period

**Interpretation**: Represents origination of high-yield, high-risk credit products (e.g., subprime mortgages, corporate leveraged loans).

#### 5.2.2 REJECT

**Financial Interpretation**: Decline loan application; adopt conservative stance.

**Strategic Role**: Defensive, risk-minimizing action.

**Mechanics**:
- **No immediate cost or benefit**
- **Risk Reduction**: Decreases $R_t^i$ by small amount (e.g., -2), representing portfolio de-risking
- **Capital Safety**: Minimal impact on $C_t^i$

**Interpretation**: Regulatory-compliant behavior, flight to quality during market uncertainty.

#### 5.2.3 INVEST

**Financial Interpretation**: Allocate liquidity to safe, productive assets (e.g., government bonds, infrastructure).

**Strategic Role**: Moderate growth, capital accumulation without risk exposure.

**Mechanics**:
- **Cost**: Decreases $L_t^i$ by investment amount (e.g., -8)
- **Benefit**: Increases $C_t^i$ by modest return (e.g., +10)
- **Risk**: No change to $R_t^i$

**Interpretation**: Prudent capital management balancing growth and safety.

#### 5.2.4 INSURE

**Financial Interpretation**: Purchase risk mitigation instruments (e.g., credit default swaps, reinsurance).

**Strategic Role**: Defensive risk management.

**Mechanics**:
- **Cost**: Decreases $L_t^i$ by premium (e.g., -7)
- **Benefit**: Decreases $R_t^i$ by coverage amount (e.g., -8)
- **Capital Impact**: Slight decrease to $C_t^i$ (e.g., -3) representing premium cost

**Interpretation**: Hedging strategies to reduce exposure to systemic shocks or portfolio losses.

#### 5.2.5 UNDERCUT

**Financial Interpretation**: Predatory competitive action targeting opponent's loan portfolio (e.g., poaching clients, spreading negative information, regulatory complaints).

**Strategic Role**: Aggressive sabotage; negative-sum interaction.

**Mechanics**:
- **Precondition**: Opponent $j$ must have active loan exposure ($R_t^j > 0$)
- **Cost to Agent $i$**: Moderate liquidity cost (e.g., -5), increases own risk slightly (e.g., +3)
- **Damage to Opponent $j$**: Increases opponent's risk (e.g., +7), decreases opponent's capital (e.g., -10)
- **Self-Harm**: Backfires if opponent has no exposure (penalties to self)

**Interpretation**: Models competitive externalities, predatory practices, and strategic market disruption in oligopolistic banking.

---

## 6. State Transition Function: Deterministic Dynamics

### 6.1 Formal Specification

The environment evolves according to a deterministic transition function:

$$
S_{t+1} = T(S_t, a_t^A, a_t^B)
$$

where $T: \mathcal{S} \times \mathcal{A}^2 \to \mathcal{S}$ is a deterministic mapping defined by explicit rules.

### 6.2 Transition Mechanics by Action Combination

For each agent $i \in \{A, B\}$, state updates follow compositional rules based on individual action $a_t^i$:

#### 6.2.1 Liquidity Updates ($L_t^i \to L_{t+1}^i$)

$$
\Delta L_t^i = \begin{cases}
-10 & \text{if } a_t^i = \text{GIVE\_LOAN} \\
0 & \text{if } a_t^i = \text{REJECT} \\
-8 & \text{if } a_t^i = \text{INVEST} \\
-7 & \text{if } a_t^i = \text{INSURE} \\
-5 & \text{if } a_t^i = \text{UNDERCUT} \\
\end{cases}
$$

$$
L_{t+1}^i = \max(0, \min(L_{\max}, L_t^i + \Delta L_t^i + \text{Repayments}_t^i))
$$

where $\text{Repayments}_t^i$ represents delayed returns from loans issued at $t - \tau$.

#### 6.2.2 Risk Updates ($R_t^i \to R_{t+1}^i$)

$$
\Delta R_t^i = \begin{cases}
+5 & \text{if } a_t^i = \text{GIVE\_LOAN} \\
-2 & \text{if } a_t^i = \text{REJECT} \\
0 & \text{if } a_t^i = \text{INVEST} \\
-8 & \text{if } a_t^i = \text{INSURE} \\
+3 & \text{if } a_t^i = \text{UNDERCUT} \\
\end{cases}
$$

**Interaction Effect (UNDERCUT Dynamics)**: If agent $j$ executes UNDERCUT and agent $i$ has $R_t^i > 0$:

$$
\Delta R_t^i \mathrel{+}= +7
$$

Risk is bounded: $R_{t+1}^i = \max(0, \min(R_{\max}, R_t^i + \Delta R_t^i))$

#### 6.2.3 Capital Updates ($C_t^i \to C_{t+1}^i$)

$$
\Delta C_t^i = \begin{cases}
0 & \text{if } a_t^i = \text{GIVE\_LOAN} \text{ (immediate)} \\
0 & \text{if } a_t^i = \text{REJECT} \\
+10 & \text{if } a_t^i = \text{INVEST} \\
-3 & \text{if } a_t^i = \text{INSURE} \\
0 & \text{if } a_t^i = \text{UNDERCUT} \\
\end{cases}
$$

**Delayed Loan Returns**: If loan was issued at $t - \tau$:

$$
\Delta C_t^i \mathrel{+}= +15
$$

**UNDERCUT Victim Penalty**: If agent $j$ executes UNDERCUT against agent $i$:

$$
\Delta C_t^i \mathrel{+}= -10
$$

Capital update: $C_{t+1}^i = C_t^i + \Delta C_t^i$

### 6.3 Critical Interaction: UNDERCUT Mechanism

The UNDERCUT action creates direct strategic interdependence:

**Condition**: Agent $j$ selects UNDERCUT at time $t$.

**Victim Identification**: Opponent $i$ (where $i \neq j$).

**Precondition Check**:
- If $R_t^i > 0$: UNDERCUT succeeds
  - $R_{t+1}^i \mathrel{+}= +7$
  - $C_{t+1}^i \mathrel{+}= -10$
  - Agent $j$ incurs cost: $L_{t+1}^j \mathrel{+}= -5$, $R_{t+1}^j \mathrel{+}= +3$

- If $R_t^i = 0$: UNDERCUT fails (no valid target)
  - Agent $j$ incurs full cost with no effect on opponent
  - Penalty: $R_{t+1}^j \mathrel{+}= +5$ (backfire risk)

**Game-Theoretic Implication**: UNDERCUT introduces negative-sum competition and retaliatory cycles, analogous to price wars or regulatory arbitrage in banking markets.

### 6.4 Determinism and Reproducibility

**Guarantee**: Given identical initial state $S_0$ and action sequences $\{a_t^A, a_t^B\}_{t=0}^{T-1}$, the trajectory $\{S_t\}_{t=0}^T$ is perfectly reproducible.

**Verification**: Transition function $T$ is implemented as pure function without random number generation, ensuring experimental reproducibility critical for scientific validation.

---

## 7. Failure, Default, and Terminal Conditions

### 7.1 Agent Failure Modes

Each agent can fail through two distinct mechanisms:

#### 7.1.1 DEFAULT (Risk-Induced Failure)

**Condition**: $R_t^i \geq R_{\text{max}}$

**Interpretation**: Excessive risk accumulation triggers regulatory intervention, depositor panic, or portfolio collapse (analogous to bank run or credit rating downgrade to junk status).

**Consequences**:
- Agent $i$ is removed from environment
- Receives large negative terminal penalty (e.g., $r_{\text{terminal}}^i = -100$)
- Opponent $j$ receives survival bonus (e.g., $r_{\text{terminal}}^j = +50$)
- Episode terminates immediately

**Real-World Analogue**: Lehman Brothers (2008), Washington Mutual bank failure.

#### 7.1.2 BANKRUPTCY (Capital Depletion)

**Condition**: $C_t^i \leq 0$

**Interpretation**: Complete erosion of financial reserves; insolvency.

**Consequences**:
- Agent $i$ is removed from environment
- Receives large negative terminal penalty (e.g., $r_{\text{terminal}}^i = -100$)
- Opponent $j$ receives survival bonus
- Episode terminates immediately

**Real-World Analogue**: Bank insolvency requiring government bailout or liquidation.

### 7.2 Time Horizon and Episode Termination

**Maximum Duration**: $T_{\text{max}} = 40$ timesteps

**Natural Termination**: If $t = T_{\text{max}}$ and both agents survive:
- Episode ends
- Terminal rewards assigned based on final capital levels

**Rationale for Finite Horizon**:
1. **Analytical Tractability**: Enables backward induction and finite-horizon dynamic programming analysis
2. **Market Cycle Modeling**: Represents fixed business cycle, regulatory review period, or financial quarter
3. **Prevents Indefinite Episodes**: Ensures learning algorithms receive terminal signals for credit assignment

### 7.3 Financial Fragility and Systemic Risk

The dual failure modes encode fundamental trade-offs in banking:

- **Risk-Return Dilemma**: Aggressive loan issuance (GIVE_LOAN) increases both capital and risk
- **Fragility**: Small increases in risk exposure can cascade to catastrophic default
- **Competitive Pressure**: UNDERCUT actions by opponent can push agents over failure thresholds
- **Survival vs. Profit**: Optimal strategies must balance capital maximization with risk management

This structure mirrors Minsky's Financial Instability Hypothesis and Kindleberger's crisis framework.

---

## 8. Reward Function Design

### 8.1 Design Principles for MARL Research

The reward structure must satisfy several criteria:

1. **Sparsity**: Minimize intermediate rewards to avoid reward hacking and encourage long-term planning
2. **Delayed Gratification**: Primary returns occur at episode termination
3. **Strategic Clarity**: Reward structure should not encode optimal strategy explicitly
4. **Measurability**: Enable clear comparison between learned policies

### 8.2 Reward Formulation (Primary Specification)

#### 8.2.1 Intermediate Rewards (Per-Timestep)

For agent $i$ at time $t < T_{\text{max}}$:

$$
r_t^i = \begin{cases}
-100 & \text{if DEFAULT or BANKRUPTCY at } t \\
0 & \text{otherwise}
\end{cases}
$$

**Rationale**: No positive intermediate feedback; agents must learn to anticipate future consequences.

#### 8.2.2 Terminal Rewards (Episode End)

If episode terminates naturally at $t = T_{\text{max}}$ with both agents surviving:

$$
r_{\text{terminal}}^i = C_T^i - \lambda \cdot R_T^i
$$

where:
- $C_T^i$ is final capital (primary objective)
- $R_T^i$ is final risk score (penalty term)
- $\lambda \in [0.5, 2.0]$ is risk aversion parameter

**If opponent $j$ fails at time $t < T_{\text{max}}$:**

$$
r_{\text{terminal}}^i = C_t^i + \beta_{\text{survival}}
$$

where $\beta_{\text{survival}} = 50$ is a survival bonus.

### 8.3 Alternative Reward Formulations

#### 8.3.1 Relative Performance Objective

$$
r_{\text{terminal}}^i = (C_T^i - C_T^j) - \lambda \cdot R_T^i
$$

**Rationale**: Explicitly competitive; agents optimize relative position rather than absolute capital.

**Game-Theoretic Implications**: Encourages zero-sum thinking; may increase UNDERCUT frequency.

#### 8.3.2 Risk-Adjusted Return (Sharpe-like Ratio)

$$
r_{\text{terminal}}^i = \frac{C_T^i - C_0^i}{1 + R_T^i}
$$

**Rationale**: Penalizes risk-taking more severely; encourages conservative strategies.

**Financial Interpretation**: Analogous to Sharpe ratio in portfolio theory.

#### 8.3.3 Incremental Capital Rewards (Less Sparse)

$$
r_t^i = \alpha \cdot (C_{t+1}^i - C_t^i) \quad \forall t < T_{\text{max}}
$$

where $\alpha = 0.1$ (small weight).

**Rationale**: Provides intermediate learning signal; reduces credit assignment problem complexity.

**Trade-off**: May encourage myopic behavior; agents optimize immediate capital gains rather than long-term survival.

### 8.4 Reward Structure and Emergent Strategies

The choice of reward function profoundly affects learned behavior:

- **Sparse Rewards**: Encourage exploration and long-horizon planning; risk of slow learning
- **Risk Penalty $\lambda$**: Higher values produce conservative agents; lower values increase aggression
- **Survival Bonus**: Creates explicit incentive to outlast opponent through defensive play or sabotage
- **Relative Rewards**: Transform environment into zero-sum game; may prevent cooperative equilibria

Experimental research should systematically vary reward parameters to characterize strategy sensitivity.

---

## 9. Learning Setup: Multi-Agent Reinforcement Learning Framework

### 9.1 Problem Formulation as Stochastic Game

CREDIT WAR is a **Markov Game** (Stochastic Game) defined by the tuple:

$$
\mathcal{G} = \langle \mathcal{N}, \mathcal{S}, \{\mathcal{A}^i\}_{i \in \mathcal{N}}, T, \{r^i\}_{i \in \mathcal{N}}, \gamma \rangle
$$

where:
- $\mathcal{N} = \{A, B\}$: Set of agents
- $\mathcal{S}$: State space (defined in Section 4)
- $\mathcal{A}^i$: Action space for agent $i$ (symmetric: $\mathcal{A}^A = \mathcal{A}^B = \mathcal{A}$)
- $T: \mathcal{S} \times \mathcal{A}^A \times \mathcal{A}^B \to \mathcal{S}$: Deterministic transition function
- $r^i: \mathcal{S} \times \mathcal{A}^A \times \mathcal{A}^B \to \mathbb{R}$: Reward function for agent $i$
- $\gamma \in [0, 1)$: Discount factor (e.g., $\gamma = 0.99$)

### 9.2 Policy Representation

Each agent $i$ learns a stochastic policy:

$$
\pi^i: \mathcal{S} \to \Delta(\mathcal{A})
$$

where $\Delta(\mathcal{A})$ is the probability simplex over actions.

**Objective**: Maximize expected discounted cumulative reward:

$$
J^i(\pi^i, \pi^{-i}) = \mathbb{E}_{\tau \sim \pi^i, \pi^{-i}} \left[ \sum_{t=0}^{T} \gamma^t r_t^i \right]
$$

where $\tau = (s_0, a_0^A, a_0^B, r_0^A, r_0^B, s_1, \ldots)$ is a trajectory.

### 9.3 Self-Play Training Protocol

**Setup**: Both agents are initialized with identical parameterized policies $\pi_{\theta}^A$ and $\pi_{\theta}^B$.

**Training Loop**:
1. Sample episodes by having agents interact via current policies
2. Collect trajectories $(s_t, a_t^A, a_t^B, r_t^A, r_t^B)$
3. Update policy parameters $\theta^A$ and $\theta^B$ independently using policy gradient methods
4. Repeat until convergence or computational budget exhausted

**Non-Stationarity**: From each agent's perspective, the environment is non-stationary because the opponent's policy $\pi^{-i}$ is evolving. This violates the Markov assumption underlying single-agent RL.

**Convergence**: Self-play in general-sum games may not converge to Nash equilibrium; strategies may cycle or exhibit chaotic dynamics.

### 9.4 Policy Gradient Methods: Proximal Policy Optimization (PPO)

**Rationale for PPO**:
- Stable in non-stationary environments
- Sample-efficient compared to vanilla policy gradients
- Handles continuous state spaces via function approximation

**Algorithm Sketch**:

For each agent $i$:

$$
\theta^i \leftarrow \theta^i + \alpha \nabla_{\theta^i} L^{\text{CLIP}}(\theta^i)
$$

where the clipped surrogate objective is:

$$
L^{\text{CLIP}}(\theta) = \mathbb{E}_t \left[ \min \left( \frac{\pi_\theta(a_t^i | s_t)}{\pi_{\theta_{\text{old}}}(a_t^i | s_t)} A_t^i, \text{clip}\left(\frac{\pi_\theta(a_t^i | s_t)}{\pi_{\theta_{\text{old}}}(a_t^i | s_t)}, 1-\epsilon, 1+\epsilon \right) A_t^i \right) \right]
$$

and $A_t^i$ is the advantage function estimate (e.g., Generalized Advantage Estimation).

**Hyperparameters** (typical):
- Learning rate: $\alpha = 3 \times 10^{-4}$
- Discount factor: $\gamma = 0.99$
- GAE parameter: $\lambda = 0.95$
- Clip parameter: $\epsilon = 0.2$
- Minibatch size: 64
- Epochs per update: 10

### 9.5 Why Q-Learning is Insufficient

**Value Function Instability**: In multi-agent settings, the Q-function $Q^i(s, a^i, a^{-i})$ depends on opponent actions. As opponent policy changes, previously learned Q-values become invalid.

**Curse of Dimensionality**: Joint action space $\mathcal{A}^2$ requires maintaining $|\mathcal{A}|^2 = 25$ action values per state.

**Credit Assignment**: Sparse terminal rewards make temporal-difference (TD) learning slow to propagate value information.

**Policy Gradient Advantage**: Directly optimize policy via gradient ascent; more stable under non-stationarity; naturally handles stochastic policies for exploration.

### 9.6 Neural Network Architecture (Conceptual)

**Input**: State vector $s_t \in \mathbb{R}^6$

**Architecture**:
- Input layer: 6 units
- Hidden layers: 2 fully connected layers with 128 units each, ReLU activation
- Output layer: $|\mathcal{A}| = 5$ units with softmax activation (policy head)
- Value head: Single linear unit for value function $V^i(s)$ (shared backbone)

**Rationale**: Simple feedforward network sufficient for low-dimensional state space; weight sharing between policy and value heads improves sample efficiency.

---

## 10. Baseline Agents and Comparative Evaluation

### 10.1 Necessity of Baselines

Rigorous evaluation of learned MARL policies requires comparison against well-defined baselines representing:
1. **Null Hypotheses**: Random behavior (lower bound on performance)
2. **Domain Knowledge**: Rule-based strategies encoding human expertise
3. **Degenerate Strategies**: Extreme policies (e.g., always aggressive, always conservative)

### 10.2 Baseline Agent Specifications

#### 10.2.1 Random Agent

**Policy**: $\pi_{\text{rand}}(a | s) = \frac{1}{|\mathcal{A}|} \quad \forall a \in \mathcal{A}$

**Purpose**: Establish performance floor; verify that learning signal exists.

**Expected Behavior**: High failure rate due to uncontrolled risk accumulation.

#### 10.2.2 Greedy Agent (Always GIVE_LOAN)

**Policy**: $\pi_{\text{greedy}}(a | s) = \mathbb{1}[a = \text{GIVE\_LOAN}]$

**Purpose**: Test pure profit-maximization without risk management.

**Expected Behavior**: Rapid capital growth followed by inevitable default as risk exceeds $R_{\text{max}}$.

**Hypothesis**: Should outperform random agent initially but fail earlier.

#### 10.2.3 Conservative Agent (Risk Minimization)

**Policy**: 
$$
\pi_{\text{cons}}(a | s) = \begin{cases}
\mathbb{1}[a = \text{INSURE}] & \text{if } R^i > 15 \\
\mathbb{1}[a = \text{INVEST}] & \text{if } L^i > 20 \\
\mathbb{1}[a = \text{REJECT}] & \text{otherwise}
\end{cases}
$$

**Purpose**: Establish high survival rate baseline with moderate capital accumulation.

**Expected Behavior**: Low default probability but suboptimal capital growth.

#### 10.2.4 Adaptive Rule-Based Agent

**Policy** (pseudocode logic):
```
if R^i > 30: action = INSURE
elif R^j > 25 and R^j > 0: action = UNDERCUT
elif L^i > 25 and R^i < 20: action = GIVE_LOAN
elif L^i > 15: action = INVEST
else: action = REJECT
```

**Purpose**: Represents sophisticated heuristic combining aggression, defense, and sabotage.

**Expected Behavior**: Competitive performance; potential benchmark for evaluating learned policies.

### 10.3 Evaluation Protocol

**Round-Robin Tournament**: Each agent plays against all others (including itself):
- Random vs Random
- Random vs Greedy
- Random vs Conservative
- Random vs Adaptive
- Random vs Learned
- (All pairwise combinations)

**Metrics**:
- Win rate (% episodes where agent survives and opponent fails)
- Average terminal capital (conditional on survival)
- Survival rate (% episodes without default/bankruptcy)
- Average timesteps to failure (for failed episodes)

**Statistical Significance**: Run $N = 1000$ episodes per matchup; report means and 95% confidence intervals.

---

## 11. Experimental Design and Methodology

### 11.1 Training Protocol

**Phase 1: Self-Play Training**
- Duration: $10^6$ environment steps (approximately 25,000 episodes)
- Agents: Two PPO agents with shared architecture, independent parameter updates
- Evaluation: Every 50,000 steps, freeze policies and evaluate against baselines

**Phase 2: Fine-Tuning Against Diverse Opponents**
- Opponent pool: Random, Greedy, Conservative, Adaptive, previous checkpoint policies
- Sample opponent uniformly at episode start
- Duration: Additional $2 \times 10^5$ steps

**Hyperparameters** (see Section 9.4)

### 11.2 Evaluation Metrics

#### 11.2.1 Performance Metrics

1. **Expected Terminal Capital**:
$$
\mathbb{E}_{\pi^A, \pi^B} [C_T^i \mid \text{survival}]
$$

2. **Survival Rate**:
$$
P(\text{agent } i \text{ does not DEFAULT or go BANKRUPT})
$$

3. **Risk-Adjusted Performance**:
$$
\frac{\mathbb{E}[C_T^i] - C_0^i}{\mathbb{E}[R_T^i] + 1}
$$

#### 11.2.2 Strategic Behavior Metrics

4. **Action Frequency Distribution**:
$$
f_a^i = \frac{1}{T} \sum_{t=0}^{T-1} \mathbb{1}[a_t^i = a] \quad \forall a \in \mathcal{A}
$$

5. **UNDERCUT Initiation Rate**: Fraction of episodes where agent uses UNDERCUT at least once

6. **Risk Ceiling**: Maximum risk score reached before defaulting (for failed episodes)

#### 11.2.3 Convergence Metrics

7. **Policy Entropy**: Measure of exploration vs exploitation:
$$
H(\pi^i | s) = -\sum_{a \in \mathcal{A}} \pi^i(a | s) \log \pi^i(a | s)
$$

8. **Inter-Policy Distance**: KL divergence between policies at successive checkpoints:
$$
D_{\text{KL}}(\pi^i_{\text{new}} \| \pi^i_{\text{old}})
$$

### 11.3 Ablation Studies

**Reward Function Sensitivity**:
- Vary risk penalty $\lambda \in \{0, 0.5, 1.0, 2.0\}$
- Measure impact on survival rate and aggression (GIVE_LOAN frequency)

**Time Horizon**:
- Test $T_{\text{max}} \in \{20, 40, 60\}$
- Hypothesis: Longer horizons favor risk-aware strategies

**Asymmetric Initial Conditions**:
- Endow Agent A with $C_0^A = 70$, Agent B with $C_0^B = 30$
- Test whether learned strategies adapt to power imbalances

### 11.4 Reproducibility Requirements

- **Random Seed Control**: Fix seeds for initialization, episode sampling, and training
- **Environment Determinism**: Verify transition function reproducibility via unit tests
- **Hyperparameter Logging**: Record all training hyperparameters, network architectures, and optimizer settings
- **Open Source Implementation**: Publish code repository with documented setup instructions

---

## 12. Game-Theoretic Analysis and Emergent Behavior

### 12.1 Connection to Classical Game Theory

#### 12.1.1 Repeated Prisoner's Dilemma Analogy

Consider simplified two-action scenario:
- **Cooperate (C)**: Play INVEST (safe, moderate return)
- **Defect (D)**: Play GIVE_LOAN (risky, high return if opponent doesn't UNDERCUT)

**Payoff Structure** (approximate):
- Both INVEST: $(+10, +10)$ per turn, no risk
- One GIVE_LOAN, one INVEST: $(+15, +10)$ but risk to first agent
- Both GIVE_LOAN: $(+15, +15)$ but mutual risk accumulation
- UNDERCUT as defection: Damages opponent but incurs self-cost

**Repetition**: Finite $T_{\text{max}} = 40$ repetitions enable conditional strategies (e.g., Tit-for-Tat).

**Key Difference**: CREDIT WAR includes state-dependence (risk accumulation) absent in classical PD.

#### 12.1.2 Nash Equilibrium Considerations

**Pure Strategy Nash Equilibrium**: Unlikely to exist due to:
- Continuous state-dependence
- Asymmetric information (although fully observable, agents don't know future opponent actions)
- Complex multi-step dynamics

**Mixed Strategy Equilibrium**: More plausible; agents randomize over actions to prevent exploitation.

**Subgame Perfect Equilibrium**: In finite horizon, backward induction applies, but computational intractability limits analytical solution.

**Empirical Approach**: Use self-play to approximate equilibrium; analyze learned policy distributions.

### 12.2 Emergent Strategic Patterns (Hypothesized)

#### 12.2.1 Risk Cycling

**Pattern**: Agents alternate between risk accumulation (GIVE_LOAN) and risk reduction (INSURE), creating periodic oscillations in $R_t^i$.

**Mechanism**: As risk approaches threshold, agents shift to defensive actions; once safe, resume profit-seeking.

**Analogy**: Business cycles in macroeconomics.

#### 12.2.2 Mutual De-Escalation

**Pattern**: After initial aggressive competition (GIVE_LOAN, UNDERCUT), agents converge to mutual INVEST or REJECT.

**Mechanism**: Learning that aggressive actions are mutually destructive; implicit coordination emerges.

**Analogy**: Collusive behavior in oligopoly (tacit collusion).

#### 12.2.3 Predator-Prey Dynamics

**Pattern**: One agent adopts aggressive strategy (GIVE_LOAN), opponent responds with UNDERCUT, leading to failure of first agent.

**Mechanism**: Exploiting opponent's risk-taking through sabotage.

**Analogy**: Regulatory arbitrage or competitive undercutting in financial markets.

#### 12.2.4 Defensive Stalemate

**Pattern**: Both agents persistently play REJECT or INSURE, achieving high survival but minimal capital growth.

**Mechanism**: Learned risk aversion dominates profit motive due to failure penalties.

**Analogy**: Credit crunch or liquidity trap during financial crisis.

### 12.3 Conditions for Systemic Stability vs. Collapse

**Stability Indicators**:
- Low UNDERCUT frequency
- Balanced risk scores ($R_t^i < 25$ for both agents)
- Positive capital growth for both agents

**Collapse Indicators**:
- Escalating UNDERCUT frequency (retaliatory cycles)
- Risk scores approaching thresholds
- One agent significantly outperforming other (power imbalance)

**Research Question**: Under what reward structures and learning dynamics do agents converge to stable equilibria vs. mutually destructive competition?

---

## 13. Extensions and Increased Realism

### 13.1 Regulatory Constraints

**Extension**: Introduce capital adequacy requirements (e.g., Basel III analogues).

**Mechanism**: Add constraint $C_t^i \geq \kappa \cdot L_t^i$ where $\kappa = 0.08$ (8% capital ratio).

**Effect**: Limits liquidity available for loans; forces capital accumulation.

**Research Question**: Do regulatory constraints stabilize system or merely shift risk to other dimensions?

### 13.2 Macroeconomic Regime Shifts

**Extension**: Introduce deterministic regime changes at fixed timesteps (e.g., $t = 20$: "recession" reduces loan returns by 30%).

**Mechanism**: Modify reward function and transition dynamics based on regime $\omega_t \in \{\text{Expansion}, \text{Recession}\}$.

**Effect**: Tests agent adaptability to structural breaks.

**Research Question**: Can agents learn regime-conditional policies?

### 13.3 Asymmetric Information

**Extension**: Agent A observes only $(L_t^A, R_t^A, C_t^A, L_t^B)$; risk score $R_t^B$ is hidden.

**Mechanism**: Convert to Partially Observable Markov Game (POMG).

**Effect**: Increases strategic uncertainty; agents must infer opponent risk.

**Research Question**: How does partial observability affect emergence of trust and cooperation?

### 13.4 Multi-Agent Generalization ($N > 2$)

**Extension**: Expand to 3-5 agents in a market.

**Mechanism**: UNDERCUT can target any opponent; systemic failure occurs if majority default.

**Effect**: Introduces coordination challenges and free-rider problems.

**Research Question**: Do learned strategies in $N=2$ setting generalize to $N > 2$?

### 13.5 Continuous Action Spaces

**Extension**: Replace discrete actions with continuous loan amounts, investment sizes, etc.

**Mechanism**: Actions become $a_t^i \in \mathbb{R}^k$ where each dimension controls intensity of action type.

**Effect**: Richer strategy space but increased learning difficulty.

**Algorithm**: Replace softmax policy with Gaussian policy for PPO.

---

## 14. Research Questions and Hypotheses

### 14.1 Core Research Questions

**RQ1**: *Strategy Convergence*
> Do self-play agents converge to stable, risk-aware strategies, or do policies exhibit persistent oscillation or chaotic dynamics?

**Hypothesis**: For sparse reward structures ($\lambda > 1$), agents converge to mixed strategies balancing INVEST and INSURE; for dense rewards ($\lambda < 0.5$), persistent oscillation due to adversarial cycling.

**RQ2**: *Systemic Fragility*
> Under what conditions does competitive interaction lead to systemic collapse (mutual default) vs. stable equilibria?

**Hypothesis**: High UNDERCUT frequency correlates with systemic collapse; environments with survival bonuses encourage mutual de-escalation.

**RQ3**: *Emergent Coordination*
> Can equilibrium-like behavior emerge in the absence of explicit communication or coordination mechanisms?

**Hypothesis**: Repeated interaction enables implicit coordination; agents learn to signal intentions via action sequences (e.g., repeated REJECT signals non-aggression).

**RQ4**: *Reward Structure Sensitivity*
> How do alternative reward formulations (absolute vs. relative performance, risk penalties) shape learned strategies?

**Hypothesis**: Relative rewards increase zero-sum competition (higher UNDERCUT frequency); absolute rewards with high risk penalties produce conservative strategies.

**RQ5**: *Generalization to Baselines*
> Do learned policies trained via self-play generalize to novel opponents (baseline agents)?

**Hypothesis**: Policies overfit to self-play dynamics; fine-tuning against diverse opponents improves robustness.

### 14.2 Testable Predictions

**Prediction 1**: Agents trained with $\lambda = 2.0$ will exhibit $>80\%$ survival rate but $<30\%$ lower terminal capital than agents trained with $\lambda = 0.5$.

**Prediction 2**: UNDERCUT frequency will be $<10\%$ in stable equilibria but $>40\%$ in episodes ending in mutual default.

**Prediction 3**: Learned policies will outperform all baseline agents in capital accumulation by $>50\%$ while maintaining comparable survival rates.

**Prediction 4**: Policy entropy will decrease over training, indicating transition from exploration to exploitation, with final entropy $H(\pi) < 1.0$ (compared to initial $H(\pi) \approx 1.6$ for uniform distribution over 5 actions).

---

## 15. Academic Positioning and Contribution

### 15.1 Position within MARL Literature

CREDIT WAR occupies a niche at the intersection of:

1. **Canonical MARL Benchmarks**: Extends beyond matrix games (too simple) and complex simulations (too opaque) by providing intermediate complexity.

2. **Self-Play and Emergent Behavior**: Joins environments like Poker, Go, and multi-agent particle environments where strategic complexity arises from interaction, not environmental stochasticity.

3. **Economic and Financial Applications**: Fills gap in MARL research focused on stylized financial systems (most work uses either real data or complex ABMs; few offer controlled experimental platforms).

### 15.2 Contributions to Agent-Based Economics

**Methodological Contribution**: Demonstrates viability of deterministic micro-worlds for studying systemic risk and competitive dynamics, complementing existing ABM literature (e.g., Delli Gatti et al., 2010; Thurner et al., 2012).

**Theoretical Contribution**: Provides tractable setting for testing hypotheses about:
- Endogenous risk accumulation
- Competitive externalities in banking
- Emergence of cooperation in repeated strategic interactions

**Empirical Contribution**: Generates synthetic data on strategic behavior under varying institutional and reward structures, enabling comparative institutional analysis.

### 15.3 Suitability for Academic Research

#### 15.3.1 Master's Thesis Applications

**Scope**: 
- Implement CREDIT WAR environment
- Train baseline MARL algorithms (PPO, A3C)
- Compare performance against rule-based agents
- Analyze emergent strategies via action frequency distributions

**Research Questions**: RQ1, RQ4 (manageable scope)

**Estimated Effort**: 3-6 months (implementation + experiments)

#### 15.3.2 Doctoral Research Applications

**Scope**:
- Comprehensive algorithmic comparison (PPO, SAC, MADDPG, etc.)
- Theoretical analysis (Nash equilibrium characterization, stability analysis)
- Extensions (regulatory constraints, asymmetric information, $N > 2$ agents)
- Publication-ready experimental evaluation with robustness checks

**Research Questions**: All RQs (RQ1-RQ5)

**Estimated Effort**: 1-2 years (as part of broader dissertation)

#### 15.3.3 Peer-Reviewed Publication Targets

**Tier 1 Venues**:
- *Autonomous Agents and Multi-Agent Systems* (AAMAS)
- *International Conference on Machine Learning* (ICML) - MARL workshop
- *Conference on Neural Information Processing Systems* (NeurIPS)

**Tier 2 Venues**:
- *Journal of Artificial Intelligence Research* (JAIR)
- *IEEE Transactions on Neural Networks and Learning Systems*
- *Journal of Economic Dynamics and Control* (computational economics special issues)

**Positioning**: Frame as methodological contribution providing new benchmark environment for MARL research with economic interpretation.

### 15.4 Limitations and Future Work

**Acknowledged Limitations**:
1. **Two-Agent Restriction**: Real financial markets involve many institutions
2. **Determinism**: Removes stochastic shocks central to real crises
3. **Simplified Action Space**: Real banking decisions are high-dimensional and continuous
4. **Absence of Information Asymmetry**: Full observability unrealistic

**Future Directions**:
- Extend to $N > 2$ agents with network structure (contagion models)
- Introduce stochastic shocks while maintaining deterministic agent dynamics
- Incorporate learning from historical data (hybrid approach)
- Develop theoretical characterization of equilibria via dynamical systems analysis

---

## Conclusion

CREDIT WAR provides a **research-grade simulation platform** for studying strategic interaction, systemic risk, and emergent coordination in multi-agent financial systems. By combining deterministic dynamics, low-dimensional state space, and rich strategic action space, the environment enables controlled experimentation infeasible with real-world data or complex ABMs.

The formal specification provided in this document establishes a foundation for:
- Rigorous empirical evaluation of MARL algorithms
- Hypothesis testing in agent-based economics
- Comparative institutional analysis of financial market designs

We position CREDIT WAR as a **micro-world** for multi-agent learning research, analogous to GridWorld in single-agent RL or matrix games in classical game theory, but with sufficient complexity to capture fundamental trade-offs in competitive banking: risk vs. return, aggression vs. stability, survival vs. profit.

Future research should explore:
1. Theoretical characterization of equilibria and convergence properties
2. Empirical validation of emergent strategies against economic theory predictions
3. Extension to multi-agent settings and richer action spaces
4. Development of human-interpretable learned policies for policy analysis

By maintaining determinism, transparency, and formal rigor, CREDIT WAR serves as an exemplar for how stylized environments can advance both computational and economic understanding of complex adaptive systems.

---

## References

*Note: This is a design specification document. A full research paper would include comprehensive literature review. Key reference areas include:*

- **Multi-Agent RL**: Hernandez-Leal et al. (2019), Zhang et al. (2021)
- **Agent-Based Financial Modeling**: Thurner et al. (2012), Gatti et al. (2010)
- **Game Theory**: Fudenberg & Tirole (1991), Mas-Colell et al. (1995)
- **Systemic Risk**: Battiston et al. (2012), Acemoglu et al. (2015)
- **Self-Play in RL**: Silver et al. (2017), Bansal et al. (2018)

---

**Document Version**: 1.0  
**Date**: January 2026  
**Status**: Formal Research Design Specification  
**Intended Use**: Academic Research Foundation for MARL/ABM Studies

