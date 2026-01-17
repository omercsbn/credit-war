"""
AggressorAgent Integration Test
Tüm agent kombinasyonlarını test ederek AggressorAgent'ın doğru çalıştığını doğrula
"""

from credit_war import CreditWarEnv, Action
from credit_war.agents import (
    RandomAgent,
    GreedyAgent,
    ConservativeAgent,
    RuleBasedAgent,
    AggressorAgent
)

def test_aggressor_vs_all_agents():
    """AggressorAgent'ı tüm diğer agentlara karşı test et"""
    
    env = CreditWarEnv(seed=42)
    aggressor = AggressorAgent(name="Aggressor", seed=42)
    
    opponents = [
        RandomAgent(name="Random", seed=43),
        GreedyAgent(name="Greedy", seed=44),
        ConservativeAgent(name="Conservative", seed=45),
        RuleBasedAgent(name="RuleBased", seed=46),
    ]
    
    results = {}
    
    for opponent in opponents:
        print(f"\n{'='*60}")
        print(f"Testing: Aggressor vs {opponent.name}")
        print(f"{'='*60}")
        
        wins = 0
        draws = 0
        losses = 0
        episodes = 10
        
        for ep in range(episodes):
            state = env.reset()
            done = False
            turn = 0
            
            while not done and turn < 50:
                # Ajanlar karar verir
                valid_a = env.get_valid_actions(state.agent_a)
                valid_b = env.get_valid_actions(state.agent_b)
                
                action_a = aggressor.select_action(
                    state.agent_a,
                    state.agent_b,
                    valid_a
                )
                action_b = opponent.select_action(
                    state.agent_b,
                    state.agent_a,
                    valid_b
                )
                
                # Aksiyonları uygula
                state, r_a, r_b, done, info = env.step(action_a, action_b)
                turn += 1
            
            # Sonucu kaydet
            if r_a > r_b:
                wins += 1
            elif r_a < r_b:
                losses += 1
            else:
                draws += 1
        
        # Sonuçları göster
        win_rate = (wins / episodes) * 100
        results[opponent.name] = win_rate
        
        print(f"Results after {episodes} episodes:")
        print(f"  Wins:   {wins} ({win_rate:.1f}%)")
        print(f"  Losses: {losses} ({(losses/episodes)*100:.1f}%)")
        print(f"  Draws:  {draws} ({(draws/episodes)*100:.1f}%)")
    
    # Genel sonuçlar
    print(f"\n{'='*60}")
    print("OVERALL RESULTS")
    print(f"{'='*60}")
    for opp_name, win_rate in results.items():
        print(f"  vs {opp_name:15s}: {win_rate:5.1f}% win rate")
    
    # Ortalama kazanma oranı
    avg_win_rate = sum(results.values()) / len(results)
    print(f"\n  Average Win Rate: {avg_win_rate:.1f}%")
    print(f"{'='*60}")
    
    # AggressorAgent'ın en az %40 kazanma oranına sahip olduğunu doğrula
    assert avg_win_rate >= 40.0, f"AggressorAgent too weak: {avg_win_rate:.1f}% < 40%"
    print("\n✅ AggressorAgent is competitive! (Average win rate >= 40%)")


if __name__ == "__main__":
    test_aggressor_vs_all_agents()
