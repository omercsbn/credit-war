"""
Command-Line Interface for CREDIT WAR

Provides easy access to running simulations and tournaments.
"""

import argparse
import sys
from typing import Dict, Type

from credit_war.env import CreditWarEnv
from credit_war.agents import (
    BaseAgent,
    RandomAgent,
    GreedyAgent,
    ConservativeAgent,
    RuleBasedAgent,
    AggressorAgent
)
from credit_war.simulation import SimulationRunner


# Registry of available agents
AGENT_REGISTRY: Dict[str, Type[BaseAgent]] = {
    "random": RandomAgent,
    "greedy": GreedyAgent,
    "conservative": ConservativeAgent,
    "rulebased": RuleBasedAgent,
    "aggressor": AggressorAgent,
}


def create_agent(agent_type: str, name: str, seed: int) -> BaseAgent:
    """
    Create agent instance from type string.
    
    Args:
        agent_type: Agent type identifier
        name: Agent name
        seed: Random seed
        
    Returns:
        Agent instance
    """
    if agent_type not in AGENT_REGISTRY:
        raise ValueError(f"Unknown agent type: {agent_type}. Available: {list(AGENT_REGISTRY.keys())}")
    
    agent_class = AGENT_REGISTRY[agent_type]
    return agent_class(name=name, seed=seed)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="CREDIT WAR: Multi-Agent Financial Risk Simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available agent types:
  random       - Uniform random action selection
  greedy       - Always issues loans (aggressive)
  conservative - Risk-minimizing strategy
  rulebased    - Adaptive heuristic agent
  aggressor    - Opponent modeling with UNDERCUT focus (adversarial)

Example usage:
  python -m credit_war.cli --agent-a random --agent-b greedy --episodes 100
  python -m credit_war.cli --agent-a aggressor --agent-b rulebased --episodes 1000 --verbose
        """
    )
    
    parser.add_argument(
        "--agent-a",
        type=str,
        required=True,
        choices=list(AGENT_REGISTRY.keys()),
        help="Agent A type"
    )
    
    parser.add_argument(
        "--agent-b",
        type=str,
        required=True,
        choices=list(AGENT_REGISTRY.keys()),
        help="Agent B type"
    )
    
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of episodes to run (default: 100)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed episode information"
    )
    
    args = parser.parse_args()
    
    # Create environment
    env = CreditWarEnv(seed=args.seed)
    
    # Create agents
    agent_a = create_agent(args.agent_a, name="Agent_A", seed=args.seed)
    agent_b = create_agent(args.agent_b, name="Agent_B", seed=args.seed + 1)
    
    # Run tournament
    print(f"Running CREDIT WAR Tournament")
    print(f"Agent A: {agent_a}")
    print(f"Agent B: {agent_b}")
    print(f"Episodes: {args.episodes}")
    print(f"Seed: {args.seed}")
    print("-" * 60)
    
    runner = SimulationRunner(env)
    metrics = runner.run_tournament(agent_a, agent_b, args.episodes, verbose=args.verbose)
    
    # Print results
    print("\n" + "=" * 60)
    print("TOURNAMENT RESULTS")
    print("=" * 60)
    
    print(f"\nTotal Episodes: {metrics.total_episodes}")
    print(f"Average Episode Length: {metrics.avg_turns:.2f} turns")
    
    print("\n--- OUTCOMES ---")
    print(f"Agent A Wins: {metrics.agent_a_wins} ({metrics.agent_a_wins / metrics.total_episodes * 100:.1f}%)")
    print(f"Agent B Wins: {metrics.agent_b_wins} ({metrics.agent_b_wins / metrics.total_episodes * 100:.1f}%)")
    print(f"Draws: {metrics.draws} ({metrics.draws / metrics.total_episodes * 100:.1f}%)")
    
    print("\n--- SURVIVAL RATES ---")
    print(f"Agent A: {metrics.agent_a_survival_rate * 100:.1f}%")
    print(f"Agent B: {metrics.agent_b_survival_rate * 100:.1f}%")
    
    print("\n--- AVERAGE CAPITAL ---")
    print(f"Agent A: {metrics.agent_a_avg_capital:.2f}")
    print(f"Agent B: {metrics.agent_b_avg_capital:.2f}")
    
    print("\n--- AVERAGE REWARDS ---")
    print(f"Agent A: {metrics.agent_a_avg_reward:.3f}")
    print(f"Agent B: {metrics.agent_b_avg_reward:.3f}")
    
    print("\n--- ACTION FREQUENCIES ---")
    print("Agent A:")
    for action, freq in sorted(metrics.agent_a_action_freq.items()):
        print(f"  {action}: {freq * 100:.1f}%")
    
    print("Agent B:")
    for action, freq in sorted(metrics.agent_b_action_freq.items()):
        print(f"  {action}: {freq * 100:.1f}%")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
