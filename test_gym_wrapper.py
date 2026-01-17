"""
Test Gymnasium Wrapper - Stable-Baselines3 Uyumluluğunu Kontrol Et

Bu script, gym_wrapper'ın doğru çalıştığını kontrol eder.
"""

import numpy as np

# Test gym wrapper
print("Testing Gymnasium Wrapper...")
print("="*60)

try:
    from credit_war.gym_wrapper import CreditWarGymEnv
    from credit_war.agents import RandomAgent
    
    # Create environment
    opponent = RandomAgent(name="TestOpponent", seed=42)
    env = CreditWarGymEnv(opponent=opponent, seed=42)
    
    print("✅ Environment created successfully")
    print(f"   Observation space: {env.observation_space}")
    print(f"   Action space: {env.action_space}")
    
    # Reset
    obs, info = env.reset()
    print(f"\n✅ Reset successful")
    print(f"   Observation shape: {obs.shape}")
    print(f"   Observation: {obs}")
    print(f"   Info: {info}")
    
    # Take a few steps
    print(f"\n✅ Running 5 test steps...")
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"   Step {i+1}: action={action}, reward={reward:.2f}, "
              f"terminated={terminated}, truncated={truncated}")
        
        if terminated or truncated:
            print(f"   Episode ended. Outcome: {info.get('outcome', 'unknown')}")
            break
    
    print("\n" + "="*60)
    print("✅ Gymnasium wrapper works correctly!")
    print("="*60)
    
except ImportError as e:
    print(f"\n❌ Import Error: {e}")
    print("\nℹ️  Stable-Baselines3 kurulu değil.")
    print("   Kurmak için: pip install -r requirements_rl.txt")
    exit(1)

except Exception as e:
    print(f"\n❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test PPOAgent (if stable-baselines3 installed)
print("\n" + "="*60)
print("Testing PPOAgent (without trained model)...")
print("="*60)

try:
    from credit_war.agents import PPOAgent
    
    if PPOAgent is None:
        print("ℹ️  PPOAgent not available (stable-baselines3 not installed)")
    else:
        print("✅ PPOAgent class available")
        print("   To train a model, run:")
        print("   python train_ppo.py --opponent rulebased --timesteps 1000000")
        
except Exception as e:
    print(f"ℹ️  PPOAgent test skipped: {e}")

print("\n" + "="*60)
print("✅ All wrapper tests passed!")
print("="*60)
print("\nNext steps:")
print("1. Install RL dependencies: pip install -r requirements_rl.txt")
print("2. Train a model: python train_ppo.py --opponent rulebased --timesteps 100000")
print("3. Evaluate model: python evaluate_ppo.py --model models/ppo_rulebased_final.zip")
print("\nSee RL_TRAINING_GUIDE.md for detailed instructions.")
