"""
Battle Royale Tournament - All Game Theory Agents vs Each Other

Round-robin turnuva: Her ajan diğer 9 ajana karşı oynar.
Sonuç: Kazananlar matrisi (heatmap), ELO ratings, detaylı istatistikler.
"""

import argparse
import os
import json
import glob
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

try:
    from stable_baselines3 import PPO
except ImportError:
    print("ERROR: stable-baselines3 not installed!")
    exit(1)

try:
    from rich.console import Console
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
except ImportError:
    print("ERROR: rich not installed!")
    exit(1)

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:
    print("ERROR: matplotlib/seaborn not installed!")
    print("Install with: pip install matplotlib seaborn")
    exit(1)

from credit_war.env import CreditWarEnv
from credit_war.agents.ppo_agent import PPOAgent
from credit_war.simulation import SimulationRunner


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


def load_agent(agent_key: str, model_paths: Dict) -> PPOAgent:
    """Load a trained PPO agent."""
    return PPOAgent(
        model_path=model_paths["model_path"],
        vec_normalize_path=model_paths["vecnormalize_path"],
        name=agent_key,
        deterministic=True
    )


def run_matchup(
    agent_a: PPOAgent,
    agent_b: PPOAgent,
    episodes: int,
    seed: int,
    env: CreditWarEnv,
    runner: SimulationRunner
) -> Dict:
    """
    Run a single matchup between two agents.
    
    Returns:
        Dict with win_rate, loss_rate, draw_rate, avg_reward
    """
    metrics = runner.run_tournament(
        agent_a=agent_a,
        agent_b=agent_b,
        num_episodes=episodes,
        verbose=False
    )
    
    return {
        "wins": metrics.agent_a_wins,
        "losses": metrics.agent_b_wins,
        "draws": metrics.draws,
        "win_rate": metrics.agent_a_wins / episodes * 100,
        "loss_rate": metrics.agent_b_wins / episodes * 100,
        "draw_rate": metrics.draws / episodes * 100,
        "avg_reward": metrics.agent_a_avg_reward,
    }


def calculate_elo_ratings(
    matchup_results: Dict[Tuple[str, str], Dict],
    initial_elo: float = 1500.0,
    k_factor: float = 32.0
) -> Dict[str, float]:
    """
    Calculate ELO ratings based on tournament results.
    
    Args:
        matchup_results: Dict mapping (agent_a, agent_b) to results
        initial_elo: Starting ELO for all agents
        k_factor: ELO update rate
    
    Returns:
        Dict mapping agent_key to final ELO rating
    """
    # Initialize ELO ratings
    agents = set()
    for (agent_a, agent_b) in matchup_results.keys():
        agents.add(agent_a)
        agents.add(agent_b)
    
    elo_ratings = {agent: initial_elo for agent in agents}
    
    # Update ELO based on all matchups
    for (agent_a, agent_b), results in matchup_results.items():
        # Expected scores
        expected_a = 1 / (1 + 10 ** ((elo_ratings[agent_b] - elo_ratings[agent_a]) / 400))
        expected_b = 1 / (1 + 10 ** ((elo_ratings[agent_a] - elo_ratings[agent_b]) / 400))
        
        # Actual scores (win=1, draw=0.5, loss=0)
        total_games = results["wins"] + results["losses"] + results["draws"]
        actual_a = (results["wins"] + 0.5 * results["draws"]) / total_games
        actual_b = (results["losses"] + 0.5 * results["draws"]) / total_games
        
        # Update ELO
        elo_ratings[agent_a] += k_factor * (actual_a - expected_a)
        elo_ratings[agent_b] += k_factor * (actual_b - expected_b)
    
    return elo_ratings


def create_heatmap(
    matchup_results: Dict[Tuple[str, str], Dict],
    agents: List[str],
    output_path: str,
    metric: str = "win_rate"
):
    """
    Create a heatmap of tournament results.
    
    Args:
        matchup_results: Tournament results
        agents: List of agent keys
        output_path: Path to save heatmap image
        metric: Which metric to plot (win_rate, avg_reward)
    """
    # Create matrix
    n = len(agents)
    matrix = np.zeros((n, n))
    
    for i, agent_a in enumerate(agents):
        for j, agent_b in enumerate(agents):
            if i == j:
                matrix[i, j] = np.nan  # Self-play (not computed)
            elif (agent_a, agent_b) in matchup_results:
                matrix[i, j] = matchup_results[(agent_a, agent_b)][metric]
            else:
                matrix[i, j] = np.nan
    
    # Create heatmap
    plt.figure(figsize=(14, 12))
    
    if metric == "win_rate":
        cmap = "RdYlGn"
        vmin, vmax = 0, 100
        fmt = ".1f"
        title = "Win Rate (%) - Agent A vs Agent B"
        cbar_label = "Win Rate (%)"
    elif metric == "avg_reward":
        cmap = "RdYlGn"
        vmin, vmax = -1, 1
        fmt = ".2f"
        title = "Average Reward - Agent A vs Agent B"
        cbar_label = "Average Reward"
    else:
        cmap = "viridis"
        vmin, vmax = None, None
        fmt = ".2f"
        title = f"{metric} - Agent A vs Agent B"
        cbar_label = metric
    
    sns.heatmap(
        matrix,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        xticklabels=agents,
        yticklabels=agents,
        cbar_kws={'label': cbar_label},
        linewidths=0.5,
        linecolor='gray',
        square=True
    )
    
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.xlabel("Agent B (Opponent)", fontsize=12, fontweight='bold')
    plt.ylabel("Agent A (Player)", fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    """Main tournament loop."""
    parser = argparse.ArgumentParser(
        description="Battle Royale Tournament - All agents vs all agents"
    )
    
    parser.add_argument(
        "--models-dir",
        type=str,
        default="./models/game_theory",
        help="Directory containing trained models (default: ./models/game_theory)"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of episodes per matchup (default: 100)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./tournament_results",
        help="Directory to save results (default: ./tournament_results)"
    )
    
    args = parser.parse_args()
    
    console = Console()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find trained models
    console.print("[yellow]Scanning for trained models...[/yellow]")
    trained_models = find_trained_models(args.models_dir)
    
    if len(trained_models) < 2:
        console.print(f"[red]Need at least 2 trained models, found {len(trained_models)}[/red]")
        console.print("[yellow]Run training first: python train_game_theory_agents.py[/yellow]")
        return
    
    agents = sorted(trained_models.keys())
    console.print(f"[green]Found {len(agents)} trained models[/green]\n")
    
    # Calculate total matchups
    total_matchups = len(agents) * (len(agents) - 1)  # N*(N-1) for directed matchups
    
    # Tournament header
    console.print("[bold magenta]"+"="*70+"[/bold magenta]")
    console.print("[bold magenta]BATTLE ROYALE TOURNAMENT[/bold magenta]")
    console.print("[bold magenta]"+"="*70+"[/bold magenta]")
    console.print(f"Agents: {len(agents)}")
    console.print(f"Total matchups: {total_matchups} ({len(agents)}x{len(agents)-1})")
    console.print(f"Episodes per matchup: {args.episodes}")
    console.print(f"Total episodes: {total_matchups * args.episodes:,}\n")
    
    # List agents
    console.print("[bold cyan]Participants:[/bold cyan]")
    for i, agent in enumerate(agents, 1):
        console.print(f"  {i:2d}. {agent}")
    console.print()
    
    # Load all agents
    console.print("[yellow]Loading agents...[/yellow]")
    loaded_agents = {}
    for agent_key in agents:
        loaded_agents[agent_key] = load_agent(agent_key, trained_models[agent_key])
    console.print("[green]✓ All agents loaded[/green]\n")
    
    # Create environment
    env = CreditWarEnv(seed=args.seed)
    runner = SimulationRunner(env)
    
    # Run tournament
    console.print("[bold cyan]Running tournament...[/bold cyan]\n")
    
    matchup_results = {}
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console
    ) as progress:
        
        task = progress.add_task(
            f"[cyan]Battle Royale Progress",
            total=total_matchups
        )
        
        for agent_a_key in agents:
            for agent_b_key in agents:
                if agent_a_key == agent_b_key:
                    continue  # Skip self-play
                
                agent_a = loaded_agents[agent_a_key]
                agent_b = loaded_agents[agent_b_key]
                
                progress.update(
                    task,
                    description=f"[cyan]{agent_a_key} vs {agent_b_key}"
                )
                
                results = run_matchup(
                    agent_a=agent_a,
                    agent_b=agent_b,
                    episodes=args.episodes,
                    seed=args.seed,
                    env=env,
                    runner=runner
                )
                
                matchup_results[(agent_a_key, agent_b_key)] = results
                
                progress.advance(task)
    
    console.print("\n[green]✓ Tournament complete![/green]\n")
    
    # Calculate ELO ratings
    console.print("[yellow]Calculating ELO ratings...[/yellow]")
    elo_ratings = calculate_elo_ratings(matchup_results)
    console.print("[green]✓ ELO ratings calculated[/green]\n")
    
    # Save raw results
    results_file = os.path.join(args.output_dir, "tournament_results.json")
    with open(results_file, 'w') as f:
        # Convert tuple keys to strings for JSON
        json_results = {
            f"{a}_vs_{b}": results
            for (a, b), results in matchup_results.items()
        }
        json.dump(json_results, f, indent=2)
    console.print(f"[green]✓ Results saved: {results_file}[/green]\n")
    
    # Create win rate matrix
    console.print("[yellow]Generating win rate heatmap...[/yellow]")
    heatmap_path = os.path.join(args.output_dir, "win_rate_heatmap.png")
    create_heatmap(matchup_results, agents, heatmap_path, metric="win_rate")
    console.print(f"[green]✓ Heatmap saved: {heatmap_path}[/green]\n")
    
    # Create reward matrix
    console.print("[yellow]Generating reward heatmap...[/yellow]")
    reward_heatmap_path = os.path.join(args.output_dir, "reward_heatmap.png")
    create_heatmap(matchup_results, agents, reward_heatmap_path, metric="avg_reward")
    console.print(f"[green]✓ Reward heatmap saved: {reward_heatmap_path}[/green]\n")
    
    # Print ELO rankings
    console.print("[bold green]"+"="*70+"[/bold green]")
    console.print("[bold green]TOURNAMENT RESULTS - ELO RANKINGS[/bold green]")
    console.print("[bold green]"+"="*70+"[/bold green]\n")
    
    sorted_elo = sorted(elo_ratings.items(), key=lambda x: x[1], reverse=True)
    
    elo_table = Table(title="ELO Rankings")
    elo_table.add_column("Rank", justify="right", style="bold")
    elo_table.add_column("Agent", style="cyan")
    elo_table.add_column("ELO Rating", justify="right", style="yellow")
    elo_table.add_column("Performance", style="green")
    
    for i, (agent, elo) in enumerate(sorted_elo, 1):
        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else ""
        performance = "Elite" if elo > 1600 else "Strong" if elo > 1500 else "Average" if elo > 1400 else "Weak"
        elo_table.add_row(
            f"{i} {medal}",
            agent,
            f"{elo:.1f}",
            performance
        )
    
    console.print(elo_table)
    console.print()
    
    # Overall statistics
    console.print("[bold cyan]OVERALL STATISTICS[/bold cyan]\n")
    
    # Calculate overall win rates
    overall_stats = {}
    for agent in agents:
        wins = 0
        losses = 0
        draws = 0
        total_reward = 0
        matchups = 0
        
        for (a, b), results in matchup_results.items():
            if a == agent:
                wins += results["wins"]
                losses += results["losses"]
                draws += results["draws"]
                total_reward += results["avg_reward"]
                matchups += 1
        
        total_games = wins + losses + draws
        overall_stats[agent] = {
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "total_games": total_games,
            "win_rate": wins / total_games * 100 if total_games > 0 else 0,
            "avg_reward": total_reward / matchups if matchups > 0 else 0,
        }
    
    stats_table = Table(title="Overall Performance")
    stats_table.add_column("Agent", style="cyan")
    stats_table.add_column("Win Rate", justify="right", style="green")
    stats_table.add_column("W-L-D", justify="right", style="white")
    stats_table.add_column("Avg Reward", justify="right", style="yellow")
    stats_table.add_column("ELO", justify="right", style="magenta")
    
    for agent in sorted(agents, key=lambda x: overall_stats[x]["win_rate"], reverse=True):
        stats = overall_stats[agent]
        stats_table.add_row(
            agent,
            f"{stats['win_rate']:.1f}%",
            f"{stats['wins']}-{stats['losses']}-{stats['draws']}",
            f"{stats['avg_reward']:+.3f}",
            f"{elo_ratings[agent]:.1f}"
        )
    
    console.print(stats_table)
    console.print()
    
    # Best and worst matchups
    console.print("[bold yellow]NOTABLE MATCHUPS[/bold yellow]\n")
    
    # Sort by win rate
    sorted_matchups = sorted(
        matchup_results.items(),
        key=lambda x: x[1]["win_rate"],
        reverse=True
    )
    
    console.print("[green]Top 10 Dominant Performances:[/green]")
    for (agent_a, agent_b), results in sorted_matchups[:10]:
        console.print(
            f"  • {agent_a} vs {agent_b}: "
            f"{results['win_rate']:.1f}% wins "
            f"({results['wins']}-{results['losses']}-{results['draws']}, "
            f"reward: {results['avg_reward']:+.3f})"
        )
    
    console.print("\n[red]Top 10 Struggling Performances:[/red]")
    for (agent_a, agent_b), results in sorted_matchups[-10:]:
        console.print(
            f"  • {agent_a} vs {agent_b}: "
            f"{results['win_rate']:.1f}% wins "
            f"({results['wins']}-{results['losses']}-{results['draws']}, "
            f"reward: {results['avg_reward']:+.3f})"
        )
    
    console.print()
    
    # Head-to-head comparison matrix (CSV)
    console.print("[yellow]Generating CSV tables...[/yellow]")
    
    # Win rate matrix
    win_matrix = pd.DataFrame(
        index=agents,
        columns=agents,
        dtype=float
    )
    
    for agent_a in agents:
        for agent_b in agents:
            if agent_a == agent_b:
                win_matrix.loc[agent_a, agent_b] = np.nan
            elif (agent_a, agent_b) in matchup_results:
                win_matrix.loc[agent_a, agent_b] = matchup_results[(agent_a, agent_b)]["win_rate"]
    
    win_matrix_path = os.path.join(args.output_dir, "win_rate_matrix.csv")
    win_matrix.to_csv(win_matrix_path)
    console.print(f"[green]✓ Win rate matrix: {win_matrix_path}[/green]")
    
    # Reward matrix
    reward_matrix = pd.DataFrame(
        index=agents,
        columns=agents,
        dtype=float
    )
    
    for agent_a in agents:
        for agent_b in agents:
            if agent_a == agent_b:
                reward_matrix.loc[agent_a, agent_b] = np.nan
            elif (agent_a, agent_b) in matchup_results:
                reward_matrix.loc[agent_a, agent_b] = matchup_results[(agent_a, agent_b)]["avg_reward"]
    
    reward_matrix_path = os.path.join(args.output_dir, "reward_matrix.csv")
    reward_matrix.to_csv(reward_matrix_path)
    console.print(f"[green]✓ Reward matrix: {reward_matrix_path}[/green]")
    
    # ELO ratings CSV
    elo_df = pd.DataFrame(
        sorted_elo,
        columns=["Agent", "ELO"]
    )
    elo_path = os.path.join(args.output_dir, "elo_ratings.csv")
    elo_df.to_csv(elo_path, index=False)
    console.print(f"[green]✓ ELO ratings: {elo_path}[/green]\n")
    
    # Final summary
    console.print("[bold magenta]"+"="*70+"[/bold magenta]")
    console.print("[bold magenta]BATTLE ROYALE COMPLETE[/bold magenta]")
    console.print("[bold magenta]"+"="*70+"[/bold magenta]\n")
    
    console.print(f"[cyan]Total matchups: {total_matchups}[/cyan]")
    console.print(f"[cyan]Total episodes: {total_matchups * args.episodes:,}[/cyan]")
    console.print(f"[cyan]Results saved to: {args.output_dir}[/cyan]\n")
    
    console.print("[bold yellow]Key Files:[/bold yellow]")
    console.print(f"  📊 Win Rate Heatmap: {heatmap_path}")
    console.print(f"  📈 Reward Heatmap: {reward_heatmap_path}")
    console.print(f"  📋 Win Rate Matrix (CSV): {win_matrix_path}")
    console.print(f"  📋 Reward Matrix (CSV): {reward_matrix_path}")
    console.print(f"  🏆 ELO Rankings (CSV): {elo_path}")
    console.print(f"  📄 Raw Results (JSON): {results_file}\n")


if __name__ == "__main__":
    main()
