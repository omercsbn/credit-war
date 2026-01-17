"""
PPO Training Script for CREDIT WAR

Bu script, Stable-Baselines3 kullanarak PPO (Proximal Policy Optimization)
algoritması ile öğrenen bir ajan eğitir.

Kullanım:
    python train_ppo.py --opponent rulebased --timesteps 1000000
"""

import argparse
import os
from typing import Dict, Any
import numpy as np

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import (
        BaseCallback,
        CheckpointCallback,
        EvalCallback
    )
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
except ImportError:
    print("ERROR: stable-baselines3 not installed!")
    print("Install with: pip install stable-baselines3[extra]")
    exit(1)

from credit_war.gym_wrapper import CreditWarGymEnv
from credit_war.agents import (
    RandomAgent,
    GreedyAgent,
    ConservativeAgent,
    RuleBasedAgent,
    AggressorAgent
)


# Opponent registry
OPPONENT_REGISTRY = {
    "random": RandomAgent,
    "greedy": GreedyAgent,
    "conservative": ConservativeAgent,
    "rulebased": RuleBasedAgent,
    "aggressor": AggressorAgent,
}


class TensorboardCallback(BaseCallback):
    """
    Custom callback for logging additional metrics to Tensorboard.
    """
    
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_wins = []
        self.episode_losses = []
        self.episode_draws = []
    
    def _on_step(self) -> bool:
        """
        Called at each step.
        """
        # Check if episode is done
        for idx, done in enumerate(self.locals.get("dones", [])):
            if done:
                # Extract episode info
                info = self.locals["infos"][idx]
                outcome = info.get("outcome", "unknown")
                
                # Track outcomes
                if outcome == "agent_a_win":
                    self.episode_wins.append(1)
                    self.episode_losses.append(0)
                    self.episode_draws.append(0)
                elif outcome == "agent_b_win":
                    self.episode_wins.append(0)
                    self.episode_losses.append(1)
                    self.episode_draws.append(0)
                else:  # draw
                    self.episode_wins.append(0)
                    self.episode_losses.append(0)
                    self.episode_draws.append(1)
                
                # Log to tensorboard every 10 episodes
                if len(self.episode_wins) % 10 == 0:
                    win_rate = np.mean(self.episode_wins[-10:])
                    loss_rate = np.mean(self.episode_losses[-10:])
                    draw_rate = np.mean(self.episode_draws[-10:])
                    
                    self.logger.record("rollout/win_rate", win_rate)
                    self.logger.record("rollout/loss_rate", loss_rate)
                    self.logger.record("rollout/draw_rate", draw_rate)
        
        return True


def make_env(opponent_name: str, seed: int = 0):
    """
    Create a single CreditWar environment with specified opponent.
    
    Args:
        opponent_name: Opponent agent type
        seed: Random seed
    
    Returns:
        Monitored environment
    """
    # Create opponent
    opponent_class = OPPONENT_REGISTRY[opponent_name]
    opponent = opponent_class(name=f"Opponent_{opponent_name}", seed=seed)
    
    # Create environment
    env = CreditWarGymEnv(opponent=opponent, seed=seed)
    
    # Wrap with Monitor for episode statistics
    env = Monitor(env)
    
    return env


def train_ppo(
    opponent: str = "rulebased",
    total_timesteps: int = 1_000_000,
    learning_rate: float = 3e-4,
    n_steps: int = 2048,
    batch_size: int = 64,
    n_epochs: int = 10,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    ent_coef: float = 0.01,
    vf_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    tensorboard_log: str = "./tensorboard_logs/",
    model_save_path: str = "./models/",
    seed: int = 42,
):
    """
    Train PPO agent against specified opponent.
    
    Args:
        opponent: Opponent agent type
        total_timesteps: Total training timesteps
        learning_rate: Learning rate for optimizer
        n_steps: Number of steps per rollout
        batch_size: Minibatch size
        n_epochs: Number of optimization epochs per rollout
        gamma: Discount factor
        gae_lambda: GAE lambda parameter
        clip_range: PPO clip range
        ent_coef: Entropy coefficient for exploration
        vf_coef: Value function coefficient
        max_grad_norm: Gradient clipping
        tensorboard_log: Tensorboard log directory
        model_save_path: Model checkpoint directory
        seed: Random seed
    """
    print(f"\n{'='*60}")
    print(f"Training PPO Agent against {opponent.upper()}")
    print(f"{'='*60}\n")
    
    # Create directories
    os.makedirs(tensorboard_log, exist_ok=True)
    os.makedirs(model_save_path, exist_ok=True)
    
    # Create training environment
    env = make_env(opponent, seed=seed)
    env = DummyVecEnv([lambda: env])
    
    # Normalize observations (critical for stable training)
    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=gamma
    )
    
    # Create evaluation environment (separate from training)
    eval_env = make_env(opponent, seed=seed + 1000)
    eval_env = DummyVecEnv([lambda: eval_env])
    eval_env = VecNormalize(
        eval_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=gamma,
        training=False  # Don't update stats during evaluation
    )
    
    # PPO model
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        verbose=1,
        tensorboard_log=tensorboard_log,
        seed=seed,
        policy_kwargs={
            "net_arch": [dict(pi=[256, 256], vf=[256, 256])]  # 2-layer network
        }
    )
    
    print(f"\nModel Architecture:")
    print(f"  Policy Network: [12 -> 256 -> 256 -> 5]")
    print(f"  Value Network:  [12 -> 256 -> 256 -> 1]")
    print(f"  Total Parameters: ~{sum(p.numel() for p in model.policy.parameters()):,}\n")
    
    # Callbacks
    callbacks = []
    
    # Checkpoint callback (save every 100k steps)
    checkpoint_callback = CheckpointCallback(
        save_freq=100_000,
        save_path=model_save_path,
        name_prefix=f"ppo_{opponent}",
        save_replay_buffer=False,
        save_vecnormalize=True
    )
    callbacks.append(checkpoint_callback)
    
    # Evaluation callback (evaluate every 50k steps)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=model_save_path,
        log_path=model_save_path,
        eval_freq=50_000,
        n_eval_episodes=20,
        deterministic=True,
        render=False
    )
    callbacks.append(eval_callback)
    
    # Custom tensorboard callback
    tensorboard_callback = TensorboardCallback(verbose=0)
    callbacks.append(tensorboard_callback)
    
    # Train!
    print(f"Starting training for {total_timesteps:,} timesteps...")
    print(f"Expected episodes: ~{total_timesteps // 40:,} (assuming ~40 steps/episode)")
    print(f"Tensorboard: tensorboard --logdir {tensorboard_log}\n")
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        tb_log_name=f"PPO_{opponent}",
        progress_bar=True
    )
    
    # Save final model
    final_path = os.path.join(model_save_path, f"ppo_{opponent}_final.zip")
    model.save(final_path)
    env.save(os.path.join(model_save_path, f"vec_normalize_{opponent}_final.pkl"))
    
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"Final model saved to: {final_path}")
    print(f"{'='*60}\n")
    
    return model, env


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Train PPO agent for CREDIT WAR",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--opponent",
        type=str,
        default="rulebased",
        choices=list(OPPONENT_REGISTRY.keys()),
        help="Opponent agent type (default: rulebased)"
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=1_000_000,
        help="Total training timesteps (default: 1M)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Learning rate (default: 3e-4)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--tensorboard-log",
        type=str,
        default="./tensorboard_logs/",
        help="Tensorboard log directory"
    )
    parser.add_argument(
        "--model-save-path",
        type=str,
        default="./models/",
        help="Model checkpoint directory"
    )
    
    args = parser.parse_args()
    
    # Train
    train_ppo(
        opponent=args.opponent,
        total_timesteps=args.timesteps,
        learning_rate=args.lr,
        tensorboard_log=args.tensorboard_log,
        model_save_path=args.model_save_path,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
