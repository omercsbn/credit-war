"""
Comprehensive Evaluation Script for All Game Theory Agents

Evaluates all 10 trained game theory agents against multiple opponents.
"""

import argparse
import os
import glob
from typing import Dict, List
import numpy as np

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
except ImportError:
    print("ERROR: stable-baselines3 not installed!")
    exit(1)

try:
    from rich.console import Console
    from rich.table import Table
    from rich.progress import track
except ImportError:
    print("ERROR: rich not installed!")
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


def find_trained_models(base_dir: str = "./models/game_theory") -> Dict[str, Dict]:
    """
    Find all trained game theory models.
    
    Returns:
        Dict mapping agent_key to model paths
    """
    models = {}
    
    if not os.path.exists(base_dir):
        return models
    
    for agent_dir in os.listdir(base_dir):
        agent_path = os.path.join(base_dir, agent_dir)
        if not os.path.isdir(agent_path):
            continue
        
        # Find final model
        model_pattern = os.path.join(agent_path, f"{agent_dir}_*_final.zip")
        model_files = glob.glob(model_pattern)
        
        if not model_files:
            continue
        
        model_path = model_files[0]
        vecnormalize_path = model_path.replace(".zip", "_vecnormalize.pkl")
        
        if not os.path.exists(vecnormalize_path):
            vecnormalize_path = None
        
        models[agent_dir] = {
            "model_path": model_path,
            "vecnormalize_path": vecnormalize_path
        }
    
    return models


def evaluate_agent(
    agent_key: str,
    model_paths: Dict,
    opponents: List[str],
    episodes: int,
    seed: int,
    console: Console
) -> Dict[str, Dict]:
    """
    Evaluate a single game theory agent against all opponents.
    
    Returns:
        Dict mapping opponent_name to results
    """
    console.print(f"\n[bold cyan]Evaluating: {agent_key}[/bold cyan]")
    
    # Load PPO agent
    ppo_agent = PPOAgent(
        model_path=model_paths["model_path"],
        vec_normalize_path=model_paths["vecnormalize_path"],
        name=f"PPO_{agent_key}",
        deterministic=True
    )
    
    # Create environment
    env = CreditWarEnv(seed=seed)
    runner = SimulationRunner(env)
    
    # Results storage
    results = {}
    
    # Test against each opponent
    for opponent_name in track(opponents, description=f"Testing {agent_key}"):
        # Create opponent
        opponent_class = OPPONENT_REGISTRY[opponent_name]
        opponent = opponent_class(name=f"Opponent_{opponent_name}", seed=seed + 100)
        
        # Run tournament
        metrics = runner.run_tournament(
            agent_a=ppo_agent,
            agent_b=opponent,
            num_episodes=episodes,
            verbose=False
        )
        
        # Store results
        results[opponent_name] = {
            "win_rate": metrics.agent_a_wins / episodes * 100,
            "loss_rate": metrics.agent_b_wins / episodes * 100,
            "draw_rate": metrics.draws / episodes * 100,
            "avg_reward": metrics.agent_a_avg_reward,
            "avg_capital": metrics.agent_a_avg_capital,
            "survival_rate": metrics.agent_a_survival_rate * 100,
        }
    
    return results


def main():
    """Main evaluation loop."""
    parser = argparse.ArgumentParser(
        description="Evaluate all trained game theory agents"
    )
    
    parser.add_argument(
        "--models-dir",
        type=str,
        default="./models/game_theory",
        help="Directory containing trained models (default: ./models/game_theory)"
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
    
    args = parser.parse_args()
    
    # Determine opponents
    if args.opponents is None:
        opponents = list(OPPONENT_REGISTRY.keys())
    else:
        opponents = args.opponents
    
    console = Console()
    
    # Find trained models
    console.print("[yellow]Scanning for trained models...[/yellow]")
    trained_models = find_trained_models(args.models_dir)
    
    if not trained_models:
        console.print(f"[red]No trained models found in {args.models_dir}[/red]")
        console.print("[yellow]Run training first: python train_game_theory_agents.py[/yellow]")
        return
    
    console.print(f"[green]Found {len(trained_models)} trained models[/green]\n")
    
    # Evaluation summary
    console.print("[bold magenta]="*70 + "[/bold magenta]")
    console.print("[bold magenta]GAME THEORY AGENTS - EVALUATION[/bold magenta]")
    console.print("[bold magenta]="*70 + "[/bold magenta]")
    console.print(f"Agents: {len(trained_models)}")
    console.print(f"Opponents: {len(opponents)}")
    console.print(f"Episodes per matchup: {args.episodes}")
    console.print(f"Total episodes: {len(trained_models) * len(opponents) * args.episodes}\n")
    
    # Evaluate each agent
    all_results = {}
    
    for agent_key, model_paths in trained_models.items():
        results = evaluate_agent(
            agent_key=agent_key,
            model_paths=model_paths,
            opponents=opponents,
            episodes=args.episodes,
            seed=args.seed,
            console=console
        )
        all_results[agent_key] = results
    
    # Print comprehensive summary
    console.print(f"\n[bold green]="*70 + "[/bold green]")
    console.print("[bold green]EVALUATION COMPLETE[/bold green]")
    console.print(f"[bold green]="*70 + "[/bold green]\n")
    
    # Table for each opponent
    for opponent_name in opponents:
        table = Table(title=f"Performance vs {opponent_name.upper()}")
        table.add_column("Agent", style="cyan")
        table.add_column("Win%", justify="right", style="green")
        table.add_column("Loss%", justify="right", style="red")
        table.add_column("Draw%", justify="right", style="yellow")
        table.add_column("Avg Reward", justify="right", style="white")
        
        for agent_key in sorted(all_results.keys()):
            results = all_results[agent_key][opponent_name]
            table.add_row(
                agent_key,
                f"{results['win_rate']:.1f}",
                f"{results['loss_rate']:.1f}",
                f"{results['draw_rate']:.1f}",
                f"{results['avg_reward']:+.3f}"
            )
        
        console.print(table)
        console.print()
    
    # Overall rankings
    console.print("[bold]OVERALL RANKINGS (by average win rate)[/bold]\n")
    
    # Calculate average win rate across all opponents
    rankings = []
    for agent_key, opponent_results in all_results.items():
        avg_win_rate = np.mean([r["win_rate"] for r in opponent_results.values()])
        avg_reward = np.mean([r["avg_reward"] for r in opponent_results.values()])
        rankings.append((agent_key, avg_win_rate, avg_reward))
    
    rankings.sort(key=lambda x: x[1], reverse=True)
    
    rank_table = Table(title="Agent Rankings")
    rank_table.add_column("Rank", justify="right", style="bold")
    rank_table.add_column("Agent", style="cyan")
    rank_table.add_column("Avg Win%", justify="right", style="green")
    rank_table.add_column("Avg Reward", justify="right", style="yellow")
    
    for i, (agent_key, win_rate, reward) in enumerate(rankings, 1):
        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else ""
        rank_table.add_row(
            f"{i} {medal}",
            agent_key,
            f"{win_rate:.1f}",
            f"{reward:+.3f}"
        )
    
    console.print(rank_table)
    console.print()
    
    # Best matchups
    console.print("[bold]NOTABLE MATCHUPS[/bold]\n")
    
    # Find best and worst performance
    all_matchups = []
    for agent_key, opponent_results in all_results.items():
        for opponent_name, results in opponent_results.items():
            all_matchups.append((
                agent_key,
                opponent_name,
                results["win_rate"],
                results["avg_reward"]
            ))
    
    all_matchups.sort(key=lambda x: x[2], reverse=True)
    
    console.print("[green]Top 5 Dominant Matchups:[/green]")
    for agent, opponent, win_rate, reward in all_matchups[:5]:
        console.print(f"  • {agent} vs {opponent}: {win_rate:.1f}% wins ({reward:+.3f} reward)")
    
    console.print("\n[red]Top 5 Struggling Matchups:[/red]")
    for agent, opponent, win_rate, reward in all_matchups[-5:]:
        console.print(f"  • {agent} vs {opponent}: {win_rate:.1f}% wins ({reward:+.3f} reward)")
    
    console.print()


if __name__ == "__main__":
    main()
