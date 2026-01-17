"""
PPO Agent - Trained Reinforcement Learning Agent

Bu ajan, Stable-Baselines3 ile eğitilmiş bir PPO modelini kullanır.
"""

from typing import List, Optional
import numpy as np

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import VecNormalize
except ImportError:
    PPO = None
    VecNormalize = None

from credit_war.actions import Action
from credit_war.state import AgentState
from .base import BaseAgent


class PPOAgent(BaseAgent):
    """
    Reinforcement Learning agent using PPO (Proximal Policy Optimization).
    
    Bu ajan, önceden eğitilmiş bir PPO modelini yükler ve
    öğrendiği stratejiyi oynarken uygular.
    
    Eğitim için: train_ppo.py scriptini kullanın
    
    Örnek:
        model_path = "models/ppo_rulebased_final.zip"
        agent = PPOAgent(model_path=model_path, name="PPO_Agent")
        action = agent.select_action(own_state, opponent_state, valid_actions)
    """
    
    def __init__(
        self, 
        model_path: str,
        vec_normalize_path: Optional[str] = None,
        name: str = "PPOAgent",
        seed: int = None,
        deterministic: bool = True
    ):
        """
        Args:
            model_path: Eğitilmiş PPO model dosyası (.zip)
            vec_normalize_path: VecNormalize stats dosyası (.pkl) - isteğe bağlı
            name: Agent ismi
            seed: Random seed (kullanılmaz ama uyumluluk için)
            deterministic: Deterministik aksiyon seçimi (True: argmax, False: sample)
        """
        super().__init__(name, seed if seed is not None else 0)
        
        if PPO is None:
            raise ImportError(
                "stable-baselines3 not installed! "
                "Install with: pip install stable-baselines3"
            )
        
        # Load model
        self.model = PPO.load(model_path)
        self.deterministic = deterministic
        
        # Load VecNormalize stats (if available)
        self.vec_normalize = None
        if vec_normalize_path:
            try:
                self.vec_normalize = VecNormalize.load(vec_normalize_path, venv=None)
                print(f"Loaded VecNormalize stats from {vec_normalize_path}")
            except Exception as e:
                print(f"Warning: Could not load VecNormalize stats: {e}")
    
    def _state_to_obs(self, own_state: AgentState, opponent_state: AgentState) -> np.ndarray:
        """
        AgentState'leri Gym observation formatına çevir.
        
        Args:
            own_state: Kendi durumumuz
            opponent_state: Rakibin durumu
        
        Returns:
            12-boyutlu observation array
        """
        obs = np.array([
            # Own state (6)
            own_state.liquidity,
            own_state.risk,
            own_state.capital,
            own_state.pending_inflows[0],
            own_state.pending_inflows[1],
            own_state.pending_inflows[2],
            # Opponent state (6)
            opponent_state.liquidity,
            opponent_state.risk,
            opponent_state.capital,
            opponent_state.pending_inflows[0],
            opponent_state.pending_inflows[1],
            opponent_state.pending_inflows[2],
        ], dtype=np.float32)
        
        return obs
    
    def select_action(
        self,
        own_state: AgentState,
        opponent_state: AgentState,
        valid_actions: List[Action]
    ) -> Action:
        """
        PPO modelini kullanarak aksiyon seç.
        
        Args:
            own_state: Kendi durumumuz
            opponent_state: Rakibin durumu
            valid_actions: Geçerli aksiyonlar (şu anda kullanılmıyor - model tüm aksiyonları dener)
        
        Returns:
            Seçilen aksiyon
        """
        # State'i observation'a çevir
        obs = self._state_to_obs(own_state, opponent_state)
        
        # Normalize (if stats available)
        if self.vec_normalize:
            obs = self.vec_normalize.normalize_obs(obs)
        
        # Model prediction
        action, _states = self.model.predict(obs, deterministic=self.deterministic)
        
        # Action integer'ı Action enum'a çevir
        # Gymnasium 0-based, Action enum 1-based (auto() starts at 1)
        selected_action = Action(int(action) + 1)
        
        # Eğer seçilen aksiyon geçerli değilse, ilk geçerli aksiyonu seç
        # (Bu durum nadiren olur - model eğitim sırasında invalid aksiyonları öğrenir)
        if selected_action not in valid_actions:
            # Fallback to first valid action
            selected_action = valid_actions[0]
        
        return selected_action
    
    def __repr__(self) -> str:
        return f"PPOAgent(name='{self.name}', deterministic={self.deterministic})"
