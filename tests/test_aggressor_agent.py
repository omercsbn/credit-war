"""
Tests for AggressorAgent - Opponent Modeling Behavior

Bu testler, AggressorAgent'ın rakip durumunu analiz ederek
doğru stratejik kararlar aldığını doğrular.
"""

import pytest
from credit_war.actions import Action
from credit_war.state import AgentState
from credit_war.agents import AggressorAgent


class TestAggressorAgent:
    """AggressorAgent davranış testleri"""
    
    @pytest.fixture
    def agent(self) -> AggressorAgent:
        """Test için AggressorAgent instance'ı"""
        return AggressorAgent(name="TestAggressor")
    
    def test_undercut_vulnerable_opponent(self, agent: AggressorAgent):
        """
        Test: Rakip zayıfken (yüksek risk, düşük sermaye) UNDERCUT yap
        """
        own_state = AgentState(
            liquidity=20.0,
            risk=10.0,  # Düşük risk - saldırabilir
            capital=50.0,
            pending_inflows=[0.0, 0.0, 0.0]
        )
        opponent_state = AgentState(
            liquidity=10.0,
            risk=15.0,  # Yüksek risk (>= 5)
            capital=20.0,  # Düşük sermaye (< 30)
            pending_inflows=[0.0, 0.0, 0.0]
        )
        valid_actions = [Action.GIVE_LOAN, Action.REJECT, Action.INVEST, 
                        Action.INSURE, Action.UNDERCUT]
        
        action = agent.select_action(own_state, opponent_state, valid_actions)
        
        # Rakip zayıf -> UNDERCUT yapmalı
        assert action == Action.UNDERCUT
    
    def test_self_preservation_priority(self, agent: AggressorAgent):
        """
        Test: Kendi risk seviyesi yüksekse, saldırı yerine INSURE yap
        """
        own_state = AgentState(
            liquidity=20.0,
            risk=30.0,  # Yüksek risk (>= 25) - korunma öncelikli
            capital=50.0,
            pending_inflows=[0.0, 0.0, 0.0]
        )
        opponent_state = AgentState(
            liquidity=10.0,
            risk=15.0,  # Rakip de zayıf ama önce kendimizi koruruz
            capital=20.0,
            pending_inflows=[0.0, 0.0, 0.0]
        )
        valid_actions = [Action.GIVE_LOAN, Action.REJECT, Action.INVEST, 
                        Action.INSURE, Action.UNDERCUT]
        
        action = agent.select_action(own_state, opponent_state, valid_actions)
        
        # Kendi riskimiz yüksek -> önce INSURE
        assert action == Action.INSURE
    
    def test_no_undercut_when_opponent_strong(self, agent: AggressorAgent):
        """
        Test: Rakip güçlüyken (düşük risk veya yüksek sermaye) UNDERCUT yapma
        """
        own_state = AgentState(
            liquidity=20.0,
            risk=10.0,
            capital=50.0,
            pending_inflows=[0.0, 0.0, 0.0]
        )
        opponent_state = AgentState(
            liquidity=30.0,
            risk=2.0,  # Düşük risk (< 5)
            capital=60.0,  # Yüksek sermaye (>= 30)
            pending_inflows=[0.0, 0.0, 0.0]
        )
        valid_actions = [Action.GIVE_LOAN, Action.REJECT, Action.INVEST, 
                        Action.INSURE, Action.UNDERCUT]
        
        action = agent.select_action(own_state, opponent_state, valid_actions)
        
        # Rakip güçlü -> UNDERCUT yapmamalı
        assert action != Action.UNDERCUT
        # Sermayemiz yüksek -> GIVE_LOAN yapmalı (agresif büyüme)
        assert action == Action.GIVE_LOAN
    
    def test_liquidity_management(self, agent: AggressorAgent):
        """
        Test: Likidite düşükken INVEST yap (rakip zayıf değilse)
        """
        own_state = AgentState(
            liquidity=3.0,  # Düşük likidite (< 5)
            risk=10.0,
            capital=50.0,
            pending_inflows=[0.0, 0.0, 0.0]
        )
        opponent_state = AgentState(
            liquidity=20.0,
            risk=2.0,  # Rakip GÜÇLÜ (düşük risk < 5) - UNDERCUT fırsatı yok
            capital=40.0,  # Rakip GÜÇLÜ (yüksek sermaye >= 30)
            pending_inflows=[0.0, 0.0, 0.0]
        )
        valid_actions = [Action.GIVE_LOAN, Action.REJECT, Action.INVEST, 
                        Action.INSURE, Action.UNDERCUT]
        
        action = agent.select_action(own_state, opponent_state, valid_actions)
        
        # Rakip güçlü (UNDERCUT fırsatı yok) + Likidite düşük -> INVEST
        assert action == Action.INVEST
    
    def test_aggressive_loan_when_strong(self, agent: AggressorAgent):
        """
        Test: Sermaye güçlü ve rakip zayıf değilse GIVE_LOAN yap
        """
        own_state = AgentState(
            liquidity=20.0,
            risk=10.0,
            capital=60.0,  # Güçlü sermaye (>= 20)
            pending_inflows=[0.0, 0.0, 0.0]
        )
        opponent_state = AgentState(
            liquidity=20.0,
            risk=2.0,  # Rakip zayıf değil (risk < 5)
            capital=50.0,
            pending_inflows=[0.0, 0.0, 0.0]
        )
        valid_actions = [Action.GIVE_LOAN, Action.REJECT, Action.INVEST, 
                        Action.INSURE, Action.UNDERCUT]
        
        action = agent.select_action(own_state, opponent_state, valid_actions)
        
        # Sermaye güçlü ve saldırı fırsatı yok -> agresif büyüme
        assert action == Action.GIVE_LOAN
    
    def test_fallback_to_invest(self, agent: AggressorAgent):
        """
        Test: GIVE_LOAN mevcut değilse INVEST yap
        """
        own_state = AgentState(
            liquidity=20.0,
            risk=10.0,
            capital=60.0,
            pending_inflows=[0.0, 0.0, 0.0]
        )
        opponent_state = AgentState(
            liquidity=20.0,
            risk=2.0,
            capital=50.0,
            pending_inflows=[0.0, 0.0, 0.0]
        )
        # GIVE_LOAN mevcut değil
        valid_actions = [Action.REJECT, Action.INVEST, Action.INSURE]
        
        action = agent.select_action(own_state, opponent_state, valid_actions)
        
        # GIVE_LOAN yok -> INVEST yapmalı (güvenli büyüme)
        assert action == Action.INVEST
    
    def test_undercut_not_available(self, agent: AggressorAgent):
        """
        Test: UNDERCUT mevcut değilse alternatif strateji kullan
        """
        own_state = AgentState(
            liquidity=20.0,
            risk=10.0,
            capital=50.0,
            pending_inflows=[0.0, 0.0, 0.0]
        )
        opponent_state = AgentState(
            liquidity=10.0,
            risk=15.0,  # Zayıf rakip
            capital=20.0,
            pending_inflows=[0.0, 0.0, 0.0]
        )
        # UNDERCUT mevcut değil (likidite yetersiz veya diğer kısıtlar)
        valid_actions = [Action.GIVE_LOAN, Action.REJECT, Action.INVEST, Action.INSURE]
        
        action = agent.select_action(own_state, opponent_state, valid_actions)
        
        # UNDERCUT yok -> başka strateji seç (GIVE_LOAN veya INVEST)
        assert action in [Action.GIVE_LOAN, Action.INVEST]
        # Sermaye yüksek -> GIVE_LOAN olmalı
        assert action == Action.GIVE_LOAN
