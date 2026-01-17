"""
Evaluation Script for Trained PPO Agents

Bu script, eğitilmiş PPO ajanlarını farklı rakiplere karşı test eder.

Kullanım:
    python evaluate_ppo.py --model models/ppo_rulebased_final.zip --episodes 100
"""

import argparse
import os
from typing import Dict, List

try:
    from stable_baselines3 import PPO
except ImportError:
    print("ERROR: stable-baselines3 not installed!")
    print("Install with: pip install stable-baselines3")
    exit(1)

from credit_war.env import CreditWarEnv
from credit_war.agents import (
    RandomAgent,
    GreedyAgent,
    ConservativeAgent,
    RuleBasedAgent,
    AggressorAgent,
)
from credit_war.agents.ppo_agent import PPOAgent
from credit_war.simulation import SimulationRunner


# Opponent registry
OPPONENT_REGISTRY = {
    "random": RandomAgent,
    "greedy": GreedyAgent,
    "conservative": ConservativeAgent,
    "rulebased": RuleBasedAgent,
    "aggressor": AggressorAgent,
}


def evaluate_ppo(
    model_path: str,
    vec_normalize_path: str = None,
    opponents: List[str] = None,
    episodes: int = 100,
    seed: int = 42,
    deterministic: bool = True
):
    """
    Evaluate trained PPO agent against multiple opponents.
    
    Args:
        model_path: Path to trained PPO model (.zip)
        vec_normalize_path: Path to VecNormalize stats (.pkl)
        opponents: List of opponent types (None = all)
        episodes: Number of episodes per opponent
        seed: Random seed
        deterministic: Use deterministic actions
    """
    # Default: test against all opponents
    if opponents is None:
        opponents = list(OPPONENT_REGISTRY.keys())
    
    print(f"\n{'='*70}")
    print(f"Evaluating PPO Agent: {os.path.basename(model_path)}")
    print(f"{'='*70}\n")
    
    # Load PPO agent
    ppo_agent = PPOAgent(
        model_path=model_path,
        vec_normalize_path=vec_normalize_path,
        name="PPO_Agent",
        deterministic=deterministic
    )
    
    # Create environment
    env = CreditWarEnv(seed=seed)
    runner = SimulationRunner(env)
    
    # Results storage
    all_results = {}
    
    # Test against each opponent
    for opponent_name in opponents:
        print(f"\n{'─'*70}")
        print(f"Testing against {opponent_name.upper()}")
        print(f"{'─'*70}")
        
        # Create opponent
        opponent_class = OPPONENT_REGISTRY[opponent_name]
        opponent = opponent_class(name=f"Opponent_{opponent_name}", seed=seed + 100)
        
        # Run tournament
        metrics = runner.run_tournament(
            agent_a=ppo_agent,
            agent_b=opponent,
            n_episodes=episodes,
            verbose=False
        )
        
        # Store results
        all_results[opponent_name] = {
            "win_rate": metrics.agent_a_wins / episodes * 100,
            "loss_rate": metrics.agent_b_wins / episodes * 100,
            "draw_rate": metrics.draws / episodes * 100,
            "avg_reward": metrics.avg_reward_a,
            "avg_capital": metrics.avg_capital_a,
            "survival_rate": metrics.agent_a_survivals / episodes * 100,
        }
        
        # Print results
        print(f"\nResults ({episodes} episodes):")
        print(f"  Win Rate:      {all_results[opponent_name]['win_rate']:.1f}%")
        print(f"  Loss Rate:     {all_results[opponent_name]['loss_rate']:.1f}%")
        print(f"  Draw Rate:     {all_results[opponent_name]['draw_rate']:.1f}%")
        print(f"  Avg Reward:    {all_results[opponent_name]['avg_reward']:+.3f}")
        print(f"  Avg Capital:   {all_results[opponent_name]['avg_capital']:.1f}")
        print(f"  Survival:      {all_results[opponent_name]['survival_rate']:.1f}%")
    
    # Summary
    print(f"\n{'='*70}")
    print(f"OVERALL SUMMARY")
    print(f"{'='*70}\n")
    
    print(f"{'Opponent':<15} {'Win%':<10} {'Loss%':<10} {'Draw%':<10} {'Avg Reward':<12}")
    print(f"{'-'*70}")
    
    for opponent_name in opponents:
        results = all_results[opponent_name]
        print(f"{opponent_name:<15} "
              f"{results['win_rate']:<10.1f} "
              f"{results['loss_rate']:<10.1f} "
              f"{results['draw_rate']:<10.1f} "
              f"{results['avg_reward']:<+12.3f}")
    
    # Overall stats
    avg_win_rate = sum(r['win_rate'] for r in all_results.values()) / len(all_results)
    avg_reward = sum(r['avg_reward'] for r in all_results.values()) / len(all_results)
    
    print(f"{'-'*70}")
    print(f"{'AVERAGE':<15} {avg_win_rate:<10.1f} {'':<10} {'':<10} {avg_reward:<+12.3f}")
    print(f"\n{'='*70}\n")
    
    return all_results


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate trained PPO agent for CREDIT WAR"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained PPO model (.zip)"
    )
    parser.add_argument(
        "--vec-normalize",
        type=str,
        default=None,
        help="Path to VecNormalize stats (.pkl)"
    )
    parser.add_argument(
        "--opponents",
        type=str,
        nargs="+",
        default=None,
        choices=list(OPPONENT_REGISTRY.keys()),
        help="Opponent types to test against (default: all)"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of episodes per opponent (default: 100)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic actions instead of deterministic"
    )
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"ERROR: Model file not found: {args.model}")
        exit(1)
    
    # Evaluate
    evaluate_ppo(
        model_path=args.model,
        vec_normalize_path=args.vec_normalize,
        opponents=args.opponents,
        episodes=args.episodes,
        seed=args.seed,
        deterministic=not args.stochastic
    )


if __name__ == "__main__":
    main()
