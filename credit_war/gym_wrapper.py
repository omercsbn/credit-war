"""
Gymnasium Wrapper for CREDIT WAR Environment

Bu wrapper, CreditWarEnv'i Stable-Baselines3 ile uyumlu hale getirir.
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from credit_war.env import CreditWarEnv
from credit_war.actions import Action
from credit_war.agents import BaseAgent
from credit_war.game_theory_rewards import RewardShaper


class CreditWarGymEnv(gym.Env):
    """
    Gymnasium-compatible wrapper for CreditWarEnv.
    
    Bu wrapper, tek bir öğrenen ajan (Agent A) perspektifinden ortamı sunar.
    Agent B, sabit bir rakip ajan (fixed opponent) olarak çalışır.
    
    Observation Space: Box(12,) - [own_state(6) + opponent_state(6)]
    - own: [liquidity, risk, capital, P1, P2, P3]
    - opponent: [liquidity, risk, capital, P1, P2, P3]
    
    Action Space: Discrete(5) - [GIVE_LOAN, REJECT, INVEST, INSURE, UNDERCUT]
    
    Reward: Sparse terminal reward
    - Win: +1
    - Loss: -1
    - Draw: 0
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(
        self, 
        opponent: BaseAgent,
        seed: Optional[int] = None,
        max_episode_steps: int = 1000,
        reward_shaper: Optional[RewardShaper] = None
    ):
        """
        Args:
            opponent: Rakip ajan (Agent B)
            seed: Random seed
            max_episode_steps: Maksimum episode uzunluğu
            reward_shaper: Optional reward shaping strategy (for game theory agents)
        """
        super().__init__()
        
        self.env = CreditWarEnv(seed=seed)
        self.opponent = opponent
        self.max_episode_steps = max_episode_steps
        self.current_step = 0
        self.reward_shaper = reward_shaper
        
        # Store previous state for reward shaping
        self.state_before_action = None
        self.last_agent_action = None
        self.last_opponent_action = None
        
        # Observation space: [own(6) + opponent(6)] = 12 boyutlu
        # Her değişken için makul bounds
        self.observation_space = spaces.Box(
            low=np.array([
                0.0,    # own_liquidity (≥0, clamped)
                0.0,    # own_risk (≥0, clamped)
                -100.0, # own_capital (unbounded, can go negative)
                0.0,    # own_P1 (≥0)
                0.0,    # own_P2 (≥0)
                0.0,    # own_P3 (≥0)
                0.0,    # opp_liquidity
                0.0,    # opp_risk
                -100.0, # opp_capital
                0.0,    # opp_P1
                0.0,    # opp_P2
                0.0,    # opp_P3
            ]),
            high=np.array([
                200.0,  # own_liquidity (practical upper bound)
                100.0,  # own_risk (practical upper bound)
                500.0,  # own_capital (practical upper bound)
                100.0,  # own_P1
                100.0,  # own_P2
                100.0,  # own_P3
                200.0,  # opp_liquidity
                100.0,  # opp_risk
                500.0,  # opp_capital
                100.0,  # opp_P1
                100.0,  # opp_P2
                100.0,  # opp_P3
            ]),
            dtype=np.float32
        )
        
        # Action space: 5 discrete actions
        self.action_space = spaces.Discrete(5)
    
    def _get_obs(self) -> np.ndarray:
        """
        GlobalState'i flat numpy array'e çevir.
        
        Returns:
            12-boyutlu observation vector
        """
        state = self.env.state
        
        obs = np.array([
            # Agent A (learner)
            state.agent_a.liquidity,
            state.agent_a.risk,
            state.agent_a.capital,
            state.agent_a.pending_inflows[0],  # P1
            state.agent_a.pending_inflows[1],  # P2
            state.agent_a.pending_inflows[2],  # P3
            # Agent B (opponent)
            state.agent_b.liquidity,
            state.agent_b.risk,
            state.agent_b.capital,
            state.agent_b.pending_inflows[0],
            state.agent_b.pending_inflows[1],
            state.agent_b.pending_inflows[2],
        ], dtype=np.float32)
        
        return obs
    
    def _get_info(self) -> Dict[str, Any]:
        """
        Ekstra episode bilgileri.
        """
        state = self.env.state
        return {
            "turn": state.turn,
            "agent_a_capital": state.agent_a.capital,
            "agent_b_capital": state.agent_b.capital,
            "agent_a_risk": state.agent_a.risk,
            "agent_b_risk": state.agent_b.risk,
        }
    
    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset environment to initial state.
        
        Returns:
            observation: 12-boyutlu numpy array
            info: Episode bilgileri
        """
        super().reset(seed=seed)
        
        self.env.reset()
        self.current_step = 0
        
        # Reset reward shaper
        if self.reward_shaper:
            self.reward_shaper.reset()
        
        self.state_before_action = None
        self.last_agent_action = None
        self.last_opponent_action = None
        
        obs = self._get_obs()
        info = self._get_info()
        
        return obs, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Args:
            action: Learner'ın seçtiği aksiyon (0-4 arası integer)
        
        Returns:
            observation: Yeni state (12-boyutlu)
            reward: Scalar reward (sparse, sadece terminal state'de ≠0)
            terminated: Episode doğal olarak bitti mi (win/loss/draw)
            truncated: Episode max_steps'e ulaştı mı
            info: Ekstra bilgiler
        """
        self.current_step += 1
        
        # Store state before action for reward shaping
        state_before = self.env.state
        
        # Action integer'ı Action enum'a çevir
        # Gymnasium 0-based, Action enum 1-based (auto() starts at 1)
        action_a = Action(int(action) + 1)
        
        # Opponent'ın aksiyonunu al
        state = self.env.state
        valid_b = self.env.get_valid_actions(state.agent_b)
        action_b = self.opponent.select_action(
            own_state=state.agent_b,
            opponent_state=state.agent_a,
            valid_actions=valid_b
        )
        
        # Environment step
        new_state, reward_a, reward_b, done, info = self.env.step(action_a, action_b)
        
        # Apply reward shaping if configured
        if self.reward_shaper:
            reward_a = self.reward_shaper.compute_reward(
                action=action_a,
                opponent_action=action_b,
                state_before=state_before,
                state_after=new_state,
                agent_id=0,  # We are agent A
                base_reward=float(reward_a)
            )
        
        # Observation
        obs = self._get_obs()
        
        # Reward (Agent A'nın reward'ı)
        reward = float(reward_a)
        
        # Terminated (doğal bitiş)
        terminated = done and self.current_step < self.max_episode_steps
        
        # Truncated (time limit)
        truncated = self.current_step >= self.max_episode_steps and not terminated
        
        # Info
        step_info = self._get_info()
        step_info["outcome"] = info.get("outcome", "ongoing")
        
        return obs, reward, terminated, truncated, step_info
    
    def render(self):
        """
        Human-readable rendering (isteğe bağlı).
        """
        state = self.env.state
        print(f"\n--- Turn {state.turn} ---")
        print(f"Agent A (Learner): L={state.agent_a.liquidity:.1f}, "
              f"R={state.agent_a.risk:.1f}, C={state.agent_a.capital:.1f}")
        print(f"Agent B (Opponent): L={state.agent_b.liquidity:.1f}, "
              f"R={state.agent_b.risk:.1f}, C={state.agent_b.capital:.1f}")
    
    def close(self):
        """Cleanup resources."""
        pass
