"""
Aggressor Agent - Opponent Modeling için Saldırgan Strateji

Bu ajan, rakibin zayıflıklarını tespit ederek saldırgan bir şekilde UNDERCUT hamleleri yapar.
MARL ve Computational Economics araştırmaları için kritik bir rakip ajan.

Strateji:
1. Rakibin risk seviyesi yüksek ve sermayesi düşükse -> UNDERCUT (Baltalama)
2. Kendi risk seviyesi yüksekse -> INSURE (Korunma)
3. Likidite sıkıntısı varsa -> INVEST (Sermaye artırma)
4. Diğer durumlarda -> GIVE_LOAN (Agresif büyüme)

Bu ajan, diğer ajanları (özellikle Greedy) elemeye odaklanarak
ortamı zorlu (adversarial) bir rekabet ortamına dönüştürür.
"""

from typing import Dict, List

from credit_war.actions import Action
from credit_war.state import AgentState
from .base import BaseAgent


class AggressorAgent(BaseAgent):
    """
    Rakip odaklı saldırgan ajan (Opponent Modeling).
    
    Bu ajan, rakibin zayıf anlarını tespit eder ve UNDERCUT ile
    rakibi oyundan çıkarmaya çalışır. MARL araştırmalarında
    "adversarial behavior" incelemek için ideal rakip.
    
    Kritik Parametreler:
    - UNDERCUT_RISK_THRESHOLD: Rakibin bu risk seviyesinin üzerinde olması gerekir
    - UNDERCUT_CAPITAL_THRESHOLD: Rakibin bu sermaye seviyesinin altında olması tercih edilir
    - SELF_RISK_THRESHOLD: Kendi riskimiz bu seviyenin üzerindeyse önce korunuruz
    - LIQUIDITY_THRESHOLD: Likidite bu seviyenin altındaysa sermaye artırırız
    """
    
    # Stratejik Eşikler (Hyperparameters for MARL Research)
    UNDERCUT_RISK_THRESHOLD = 5      # Rakip en az bu kadar risk taşımalı
    UNDERCUT_CAPITAL_THRESHOLD = 30  # Rakip bu sermayenin altındaysa daha kolay elenir
    SELF_RISK_THRESHOLD = 25         # Kendi riskimiz bu seviyenin üzerindeyse INSURE yap
    LIQUIDITY_THRESHOLD = 5          # Likidite bu seviyenin altındaysa INVEST yap
    AGGRESSIVE_LOAN_THRESHOLD = 20   # Sermaye bu seviyenin üzerindeyse agresif kredi ver
    
    def __init__(self, name: str = "Aggressor", seed: int = None):
        """
        Args:
            name: Ajanın ismi (loglar ve raporlar için)
            seed: Random seed (kullanılmaz ama CLI uyumluluğu için)
        """
        super().__init__(name, seed if seed is not None else 0)
    
    def select_action(
        self, 
        own_state: AgentState, 
        opponent_state: AgentState, 
        valid_actions: List[Action]
    ) -> Action:
        """
        Rakibin durumunu analiz ederek en saldırgan hamleyi seç.
        
        Karar Ağacı:
        1. UNDERCUT fırsatı var mı? (Rakip zayıf mı?)
        2. Kendimiz risk altında mıyız? (INSURE gerekli mi?)
        3. Likidite sıkıntımız var mı? (INVEST gerekli mi?)
        4. Varsayılan: GIVE_LOAN (Agresif büyüme)
        
        Args:
            own_state: Kendi durumumuz
            opponent_state: Rakibin durumu (UNDERCUT için kritik!)
            valid_actions: Mevcut geçerli aksiyonlar
            
        Returns:
            Seçilen aksiyon
        """
        # Fırsat Analizi: Rakip UNDERCUT için uygun mu?
        opponent_is_vulnerable = (
            opponent_state.risk >= self.UNDERCUT_RISK_THRESHOLD and
            opponent_state.capital < self.UNDERCUT_CAPITAL_THRESHOLD
        )
        
        # Kendi durumumuzu değerlendir
        self_at_risk = own_state.risk >= self.SELF_RISK_THRESHOLD
        low_liquidity = own_state.liquidity < self.LIQUIDITY_THRESHOLD
        strong_capital = own_state.capital >= self.AGGRESSIVE_LOAN_THRESHOLD
        
        # Karar 1: UNDERCUT Fırsatı (En Yüksek Öncelik)
        # Rakip zayıfsa ve biz çok riskli değilsek, saldır!
        if (Action.UNDERCUT in valid_actions and 
            opponent_is_vulnerable and 
            not self_at_risk):
            return Action.UNDERCUT
        
        # Karar 2: Kendimizi Koru
        # Risk seviyemiz yüksekse, önce risk azalt
        if Action.INSURE in valid_actions and self_at_risk:
            return Action.INSURE
        
        # Karar 3: Likidite Yönetimi
        # Likidite düşükse, sermayeyi likiditeye çevir
        if Action.INVEST in valid_actions and low_liquidity:
            return Action.INVEST
        
        # Karar 4: Agresif Büyüme (Varsayılan Strateji)
        # Sermayemiz güçlüyse, kredi vererek riski rakibe yükle ve büyü
        if Action.GIVE_LOAN in valid_actions and strong_capital:
            return Action.GIVE_LOAN
        
        # Karar 5: Güvenli Büyüme
        # GIVE_LOAN çok riskli/maliyetliyse, INVEST ile sermaye artır
        if Action.INVEST in valid_actions:
            return Action.INVEST
        
        # Fallback: İlk geçerli aksiyonu seç
        # (Normalde buraya gelmemeli, ama güvenlik için)
        return valid_actions[0]
    
    def __repr__(self) -> str:
        return f"AggressorAgent(name='{self.name}')"
