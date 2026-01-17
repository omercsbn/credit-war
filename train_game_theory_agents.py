"""
Batch Training Script for 10 Game Theory-Based PPO Agents

Her oyun teorisi karakterini sırayla eğitir.
"""

import argparse
import os
import time
from typing import Dict

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from stable_baselines3.common.callbacks import (
        CheckpointCallback,
        EvalCallback,
        CallbackList,
    )
except ImportError:
    print("ERROR: stable-baselines3 not installed!")
    print("Install with: pip install stable-baselines3")
    exit(1)

try:
    from tqdm.rich import tqdm
    from rich.console import Console
    from rich.table import Table
except ImportError:
    print("ERROR: tqdm/rich not installed!")
    print("Install with: pip install tqdm rich")
    exit(1)

from credit_war.gym_wrapper import CreditWarGymEnv
from credit_war.agents import RuleBasedAgent
from credit_war.game_theory_rewards import REWARD_SHAPER_REGISTRY, get_reward_shaper


# Agent configurations
GAME_THEORY_AGENTS = {
    "nash": {
        "name": "Nash Equilibrium Banker",
        "description": "Seeks mixed strategies and Nash equilibrium",
        "reward_shaper": "nash",
    },
    "titfortat": {
        "name": "Tit-for-Tat Banker",
        "description": "Mirrors opponent behavior, forgives once",
        "reward_shaper": "titfortat",
    },
    "grimtrigger": {
        "name": "Grim Trigger Bank",
        "description": "Cooperates until betrayal, then permanent punishment",
        "reward_shaper": "grimtrigger",
    },
    "bayesian": {
        "name": "Adaptive Bayesian Bank",
        "description": "Updates beliefs and exploits learned patterns",
        "reward_shaper": "bayesian",
    },
    "predator": {
        "name": "Predator Bank",
        "description": "Hunts weak opponents aggressively",
        "reward_shaper": "predator",
    },
    "minimax": {
        "name": "Risk-Averse Regulator Bank",
        "description": "Minimizes worst-case loss, survival first",
        "reward_shaper": "minimax",
    },
    "switcher": {
        "name": "Opportunistic Switcher",
        "description": "Alternates regimes unpredictably",
        "reward_shaper": "switcher",
    },
    "evolutionary": {
        "name": "Evolutionary Learner",
        "description": "Imitates success, discards failure",
        "reward_shaper": "evolutionary",
    },
    "zerosum": {
        "name": "Zero-Sum Warrior",
        "description": "Maximizes relative advantage over opponent",
        "reward_shaper": "zerosum",
    },
    "metalearner": {
        "name": "Meta-Learner Bank",
        "description": "Classifies opponents and switches strategies",
        "reward_shaper": "metalearner",
    },
}


def make_env(opponent, reward_shaper_name: str, seed: int):
    """Create environment with reward shaping."""
    def _init():
        reward_shaper = get_reward_shaper(reward_shaper_name)
        env = CreditWarGymEnv(
            opponent=opponent,
            seed=seed,
            max_episode_steps=1000,
            reward_shaper=reward_shaper
        )
        return env
    return _init


def train_agent(
    agent_key: str,
    config: Dict,
    opponent_name: str,
    timesteps: int,
    seed: int,
    console: Console,
    eval_freq: int = 5000,
    save_freq: int = 10000,
):
    """
    Train a single game theory agent.
    
    Args:
        agent_key: Agent identifier (e.g., "nash")
        config: Agent configuration dict
        opponent_name: Opponent type to train against
        timesteps: Total training timesteps
        seed: Random seed
        console: Rich console for logging
        eval_freq: Evaluation frequency
        save_freq: Checkpoint save frequency
    
    Returns:
        Training statistics dict
    """
    console.print(f"\n[bold cyan]{'='*70}[/bold cyan]")
    console.print(f"[bold cyan]Training: {config['name']} ({agent_key})[/bold cyan]")
    console.print(f"[cyan]Strategy: {config['description']}[/cyan]")
    console.print(f"[cyan]Opponent: {opponent_name}[/cyan]")
    console.print(f"[cyan]Timesteps: {timesteps:,}[/cyan]")
    console.print(f"[bold cyan]{'='*70}[/bold cyan]\n")
    
    start_time = time.time()
    
    # Create opponent
    if opponent_name == "rulebased":
        opponent = RuleBasedAgent(name=f"Opponent_{opponent_name}", seed=seed + 100)
    else:
        raise ValueError(f"Unknown opponent: {opponent_name}")
    
    # Create vectorized environment with reward shaping
    env = DummyVecEnv([make_env(opponent, config["reward_shaper"], seed)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    
    # Create eval environment
    eval_env = DummyVecEnv([make_env(opponent, config["reward_shaper"], seed + 1)])
    eval_env = VecNormalize(
        eval_env, norm_obs=True, norm_reward=True, clip_obs=10.0, training=False
    )
    
    # Output directories
    models_dir = f"./models/game_theory/{agent_key}"
    logs_dir = f"./logs/game_theory/{agent_key}"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    
    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=models_dir,
        name_prefix=f"{agent_key}_{opponent_name}",
        save_vecnormalize=True,
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=models_dir,
        log_path=logs_dir,
        eval_freq=eval_freq,
        deterministic=True,
        render=False,
        n_eval_episodes=10,
    )
    
    callbacks = CallbackList([checkpoint_callback, eval_callback])
    
    # Create PPO model
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,  # Encourage exploration
        verbose=0,
        tensorboard_log=logs_dir,
        policy_kwargs=dict(net_arch=[256, 256]),
        seed=seed,
    )
    
    console.print(f"[yellow]Training in progress...[/yellow]")
    
    # Train
    model.learn(
        total_timesteps=timesteps,
        callback=callbacks,
        progress_bar=True,
    )
    
    # Save final model
    final_model_path = f"{models_dir}/{agent_key}_{opponent_name}_final"
    model.save(final_model_path)
    env.save(f"{final_model_path}_vecnormalize.pkl")
    
    elapsed_time = time.time() - start_time
    
    console.print(f"\n[green]✓ Training complete![/green]")
    console.print(f"  Time: {elapsed_time:.1f}s ({elapsed_time/60:.1f} min)")
    console.print(f"  Model: {final_model_path}.zip")
    console.print(f"  VecNormalize: {final_model_path}_vecnormalize.pkl\n")
    
    return {
        "agent_key": agent_key,
        "name": config["name"],
        "timesteps": timesteps,
        "elapsed_time": elapsed_time,
        "model_path": f"{final_model_path}.zip",
        "vecnormalize_path": f"{final_model_path}_vecnormalize.pkl",
    }


def main():
    """Main training loop."""
    parser = argparse.ArgumentParser(
        description="Train all 10 game theory-based PPO agents"
    )
    
    parser.add_argument(
        "--agents",
        type=str,
        nargs="+",
        default=None,
        choices=list(GAME_THEORY_AGENTS.keys()) + ["all"],
        help="Which agents to train (default: all)"
    )
    parser.add_argument(
        "--opponent",
        type=str,
        default="rulebased",
        choices=["rulebased"],
        help="Opponent type (default: rulebased)"
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=50000,
        help="Training timesteps per agent (default: 50000)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=5000,
        help="Evaluation frequency (default: 5000)"
    )
    parser.add_argument(
        "--save-freq",
        type=int,
        default=10000,
        help="Checkpoint save frequency (default: 10000)"
    )
    
    args = parser.parse_args()
    
    # Determine which agents to train
    if args.agents is None or "all" in args.agents:
        agents_to_train = list(GAME_THEORY_AGENTS.keys())
    else:
        agents_to_train = args.agents
    
    console = Console()
    
    # Print training summary
    console.print(f"\n[bold magenta]{'='*70}[/bold magenta]")
    console.print("[bold magenta]GAME THEORY AGENTS - BATCH TRAINING[/bold magenta]")
    console.print(f"[bold magenta]{'='*70}[/bold magenta]")
    console.print(f"Agents: {len(agents_to_train)}")
    console.print(f"Timesteps per agent: {args.timesteps:,}")
    console.print(f"Total timesteps: {args.timesteps * len(agents_to_train):,}")
    console.print(f"Opponent: {args.opponent}")
    console.print(f"Seed: {args.seed}\n")
    
    # Training loop
    results = []
    total_start_time = time.time()
    
    for i, agent_key in enumerate(agents_to_train, 1):
        config = GAME_THEORY_AGENTS[agent_key]
        
        console.print(f"[bold white]Agent {i}/{len(agents_to_train)}[/bold white]")
        
        result = train_agent(
            agent_key=agent_key,
            config=config,
            opponent_name=args.opponent,
            timesteps=args.timesteps,
            seed=args.seed + i,  # Different seed per agent
            console=console,
            eval_freq=args.eval_freq,
            save_freq=args.save_freq,
        )
        
        results.append(result)
    
    total_elapsed_time = time.time() - total_start_time
    
    # Print final summary
    console.print(f"\n[bold green]{'='*70}[/bold green]")
    console.print("[bold green]TRAINING COMPLETE - ALL AGENTS[/bold green]")
    console.print(f"[bold green]{'='*70}[/bold green]\n")
    
    table = Table(title="Training Summary")
    table.add_column("Agent", style="cyan")
    table.add_column("Name", style="white")
    table.add_column("Timesteps", justify="right", style="yellow")
    table.add_column("Time (min)", justify="right", style="green")
    
    for result in results:
        table.add_row(
            result["agent_key"],
            result["name"],
            f"{result['timesteps']:,}",
            f"{result['elapsed_time']/60:.1f}"
        )
    
    console.print(table)
    console.print(f"\n[bold]Total Time: {total_elapsed_time/60:.1f} minutes ({total_elapsed_time/3600:.1f} hours)[/bold]")
    console.print(f"\n[cyan]Models saved to: ./models/game_theory/[/cyan]")
    console.print(f"[cyan]Logs saved to: ./logs/game_theory/[/cyan]\n")
    
    # Print next steps
    console.print("[bold yellow]Next Steps:[/bold yellow]")
    console.print("1. Evaluate agents:")
    console.print("   python evaluate_game_theory_agents.py")
    console.print("2. View TensorBoard logs:")
    console.print("   tensorboard --logdir ./logs/game_theory/")
    console.print("3. Run tournament:")
    console.print("   python tournament_all_agents.py\n")


if __name__ == "__main__":
    main()
