"""
Example script demonstrating CREDIT WAR API usage

This script shows how to:
1. Create and configure the environment
2. Implement a custom agent
3. Run manual simulations
4. Collect and analyze metrics
"""

from credit_war import CreditWarEnv, Action
from credit_war.agents import BaseAgent, RandomAgent, RuleBasedAgent
from credit_war.simulation import SimulationRunner
from credit_war.state import AgentState
from typing import List


# ============================================================
# EXAMPLE 1: Manual Step-by-Step Execution
# ============================================================

def example_manual_execution():
    """Demonstrate manual control of environment."""
    print("=" * 60)
    print("EXAMPLE 1: Manual Step-by-Step Execution")
    print("=" * 60)
    
    env = CreditWarEnv(seed=42)
    state = env.reset()
    
    print(f"\nInitial State:")
    print(f"  Agent A: L={state.agent_a.liquidity}, R={state.agent_a.risk}, C={state.agent_a.capital}")
    print(f"  Agent B: L={state.agent_b.liquidity}, R={state.agent_b.risk}, C={state.agent_b.capital}")
    
    # Execute a few steps
    actions = [
        (Action.GIVE_LOAN, Action.INVEST, "A aggressive, B safe"),
        (Action.INVEST, Action.GIVE_LOAN, "A safe, B aggressive"),
        (Action.INSURE, Action.REJECT, "A mitigates risk, B defensive"),
        (Action.UNDERCUT, Action.UNDERCUT, "Both attack!"),
    ]
    
    for action_a, action_b, description in actions:
        print(f"\nTurn {state.turn}: {description}")
        print(f"  Actions: A={action_a.name}, B={action_b.name}")
        
        state, reward_a, reward_b, done, info = env.step(action_a, action_b)
        
        print(f"  Result: A: L={state.agent_a.liquidity}, R={state.agent_a.risk}, C={state.agent_a.capital}")
        print(f"          B: L={state.agent_b.liquidity}, R={state.agent_b.risk}, C={state.agent_b.capital}")
        print(f"  Rewards: A={reward_a}, B={reward_b}")
        
        if done:
            print(f"\n  Episode ended: {info['outcome']}")
            break


# ============================================================
# EXAMPLE 2: Custom Agent Implementation
# ============================================================

class MyCustomAgent(BaseAgent):
    """
    Example custom agent with simple heuristic:
    - Build capital early via loans
    - Switch to defense when risk is high
    - Attack opponent if they're vulnerable
    """
    
    RISK_LIMIT = 25
    ATTACK_THRESHOLD = 20
    
    def select_action(
        self,
        own_state: AgentState,
        opponent_state: AgentState,
        valid_actions: List[Action]
    ) -> Action:
        # Priority 1: Risk management
        if own_state.risk > self.RISK_LIMIT and Action.INSURE in valid_actions:
            return Action.INSURE
        
        # Priority 2: Attack vulnerable opponent
        if (opponent_state.risk > self.ATTACK_THRESHOLD and 
            Action.UNDERCUT in valid_actions):
            return Action.UNDERCUT
        
        # Priority 3: Aggressive growth if risk is manageable
        if own_state.risk < 15 and Action.GIVE_LOAN in valid_actions:
            return Action.GIVE_LOAN
        
        # Priority 4: Safe investment
        if Action.INVEST in valid_actions:
            return Action.INVEST
        
        # Default: Conservative
        if Action.REJECT in valid_actions:
            return Action.REJECT
        
        # Fallback
        return valid_actions[0]


def example_custom_agent():
    """Demonstrate custom agent usage."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Custom Agent Implementation")
    print("=" * 60)
    
    env = CreditWarEnv(seed=100)
    my_agent = MyCustomAgent(name="MyAgent", seed=100)
    opponent = RuleBasedAgent(name="Opponent", seed=200)
    
    runner = SimulationRunner(env)
    metrics = runner.run_tournament(my_agent, opponent, num_episodes=20)
    
    print(f"\nCustom Agent vs Rule-Based Agent (20 episodes):")
    print(f"  MyAgent Wins: {metrics.agent_a_wins} ({metrics.agent_a_wins/20*100:.0f}%)")
    print(f"  Opponent Wins: {metrics.agent_b_wins} ({metrics.agent_b_wins/20*100:.0f}%)")
    print(f"  Draws: {metrics.draws}")
    print(f"\n  MyAgent Survival Rate: {metrics.agent_a_survival_rate*100:.0f}%")
    print(f"  MyAgent Avg Capital: {metrics.agent_a_avg_capital:.1f}")
    print(f"  MyAgent Avg Reward: {metrics.agent_a_avg_reward:.3f}")
    
    print(f"\n  MyAgent Action Distribution:")
    for action, freq in sorted(metrics.agent_a_action_freq.items()):
        print(f"    {action}: {freq*100:.1f}%")


# ============================================================
# EXAMPLE 3: Round-Robin Tournament
# ============================================================

def example_tournament():
    """Run a round-robin tournament between multiple agents."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Round-Robin Tournament")
    print("=" * 60)
    
    from credit_war.agents import RandomAgent, GreedyAgent, ConservativeAgent
    
    agents = [
        RandomAgent(name="Random", seed=1),
        GreedyAgent(name="Greedy", seed=2),
        ConservativeAgent(name="Conservative", seed=3),
        RuleBasedAgent(name="RuleBased", seed=4),
    ]
    
    env = CreditWarEnv(seed=500)
    runner = SimulationRunner(env)
    
    results = {}
    
    # Run all pairwise matchups
    for i, agent_a in enumerate(agents):
        for j, agent_b in enumerate(agents):
            if i <= j:  # Avoid duplicate matches and self-play
                continue
            
            metrics = runner.run_tournament(agent_a, agent_b, num_episodes=10)
            
            matchup = f"{agent_a.name} vs {agent_b.name}"
            results[matchup] = {
                'a_wins': metrics.agent_a_wins,
                'b_wins': metrics.agent_b_wins,
                'draws': metrics.draws,
                'a_survival': metrics.agent_a_survival_rate,
                'b_survival': metrics.agent_b_survival_rate,
            }
    
    # Display results
    print("\nTournament Results (10 episodes per matchup):")
    print("-" * 60)
    for matchup, data in results.items():
        print(f"\n{matchup}:")
        print(f"  Wins: {data['a_wins']} - {data['b_wins']} (Draws: {data['draws']})")
        print(f"  Survival: {data['a_survival']*100:.0f}% - {data['b_survival']*100:.0f}%")


# ============================================================
# EXAMPLE 4: Analyzing Action Patterns
# ============================================================

def example_action_analysis():
    """Analyze how agents adapt their strategies."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Action Pattern Analysis")
    print("=" * 60)
    
    env = CreditWarEnv(seed=999)
    agent_a = RuleBasedAgent(name="RuleA", seed=999)
    agent_b = RuleBasedAgent(name="RuleB", seed=1000)
    
    state = env.reset()
    done = False
    
    action_sequence_a = []
    action_sequence_b = []
    risk_history_a = [state.agent_a.risk]
    risk_history_b = [state.agent_b.risk]
    
    while not done:
        valid_a = env.get_valid_actions(state.agent_a)
        valid_b = env.get_valid_actions(state.agent_b)
        
        action_a = agent_a.select_action(state.agent_a, state.agent_b, valid_a)
        action_b = agent_b.select_action(state.agent_b, state.agent_a, valid_b)
        
        action_sequence_a.append(action_a.name)
        action_sequence_b.append(action_b.name)
        
        state, _, _, done, info = env.step(action_a, action_b)
        
        risk_history_a.append(state.agent_a.risk)
        risk_history_b.append(state.agent_b.risk)
    
    print(f"\nEpisode completed in {state.turn} turns")
    print(f"Outcome: {info['outcome']}")
    
    print(f"\nAgent A Action Sequence (first 10 turns):")
    print("  " + " -> ".join(action_sequence_a[:10]))
    
    print(f"\nAgent B Action Sequence (first 10 turns):")
    print("  " + " -> ".join(action_sequence_b[:10]))
    
    print(f"\nRisk Evolution:")
    print(f"  Agent A: {risk_history_a[:10]} ...")
    print(f"  Agent B: {risk_history_b[:10]} ...")


# ============================================================
# Main Execution
# ============================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("CREDIT WAR: Example Usage Demonstrations")
    print("=" * 60)
    
    example_manual_execution()
    example_custom_agent()
    example_tournament()
    example_action_analysis()
    
    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)
