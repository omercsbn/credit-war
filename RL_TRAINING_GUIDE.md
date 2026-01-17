# CREDIT WAR - Reinforcement Learning Training Guide

Bu rehber, CREDIT WAR ortamında PPO (Proximal Policy Optimization) algoritması ile ajan eğitmeyi açıklar.

---

## 📦 Kurulum

### 1. Temel Bağımlılıkları Yükleyin

```bash
# Ana projeyi yükleyin
pip install -e .

# RL kütüphanelerini yükleyin
pip install -r requirements_rl.txt
```

### 2. GPU Desteği (İsteğe Bağlı)

Daha hızlı eğitim için PyTorch GPU versiyonunu yükleyin:

```bash
# CUDA 11.8 için
pip install torch --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1 için  
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

---

## 🎯 Hızlı Başlangıç

### Basit Eğitim (RuleBased Rakibe Karşı)

```bash
python train_ppo.py --opponent rulebased --timesteps 1000000
```

Bu komut:
- ✅ **1 Milyon timestep** eğitim çalıştırır (~25,000 episode)
- ✅ **RuleBasedAgent** rakibine karşı öğrenir
- ✅ Model checkpoint'lerini `./models/` klasörüne kaydeder
- ✅ Tensorboard loglarını `./tensorboard_logs/` klasörüne yazar

### Eğitimi İzleme (Tensorboard)

```bash
tensorboard --logdir tensorboard_logs/
```

Tarayıcınızda `http://localhost:6006` adresini açın.

**İzlenecek Metrikler:**
- `rollout/win_rate` - Kazanma oranı (hedef: >50%)
- `rollout/ep_rew_mean` - Ortalama reward
- `train/policy_loss` - Policy network loss
- `train/value_loss` - Value network loss

---

## 🧪 Farklı Rakiplere Karşı Eğitim

### Kolay Rakip (Random)
```bash
python train_ppo.py --opponent random --timesteps 500000
```
- **Avantaj**: Hızlı öğrenme, yüksek win rate
- **Dezavantaj**: Zayıf stratejiler öğrenir

### Orta Zorluk (Conservative)
```bash
python train_ppo.py --opponent conservative --timesteps 1000000
```
- **Avantaj**: Dengeli öğrenme
- **Dezavantaj**: Risk yönetimi üzerine odaklanır

### Zor Rakip (Aggressor)
```bash
python train_ppo.py --opponent aggressor --timesteps 2000000
```
- **Avantaj**: Adversarial davranış öğrenir
- **Dezavantaj**: Uzun eğitim süresi, düşük başlangıç win rate

### En Zor Rakip (RuleBased)
```bash
python train_ppo.py --opponent rulebased --timesteps 2000000
```
- **Avantaj**: En güçlü stratejileri öğrenir
- **Dezavantaj**: Uzun yakınsama süresi

---

## 🎛️ Hyperparameter Tuning

### Daha Hızlı Öğrenme (Yüksek Learning Rate)
```bash
python train_ppo.py --opponent rulebased --lr 5e-4 --timesteps 1000000
```

### Daha Kararlı Eğitim (Düşük Learning Rate)
```bash
python train_ppo.py --opponent aggressor --lr 1e-4 --timesteps 2000000
```

### Custom Seed
```bash
python train_ppo.py --opponent rulebased --seed 999 --timesteps 1000000
```

---

## 📊 Model Değerlendirme

### Tek Rakibe Karşı Test

```bash
python evaluate_ppo.py --model models/ppo_rulebased_final.zip --episodes 100 --opponents rulebased
```

### Tüm Rakiplere Karşı Test

```bash
python evaluate_ppo.py --model models/ppo_rulebased_final.zip --episodes 100
```

**Beklenen Çıktı:**

```
======================================================================
OVERALL SUMMARY
======================================================================

Opponent        Win%       Loss%      Draw%      Avg Reward  
----------------------------------------------------------------------
random          95.0       5.0        0.0        +0.900      
greedy          30.0       0.0        70.0       +0.300      
conservative    80.0       10.0       10.0       +0.700      
rulebased       60.0       20.0       20.0       +0.400      
aggressor       55.0       25.0       20.0       +0.300      
----------------------------------------------------------------------
AVERAGE         64.0                             +0.520      
======================================================================
```

---

## 🔬 Gelişmiş Kullanım

### 1. Curriculum Learning (Kademeli Zorluk)

```bash
# Adım 1: Random rakibe karşı temel öğren
python train_ppo.py --opponent random --timesteps 500000

# Adım 2: Conservative'e karşı risk yönetimi öğren
python train_ppo.py --opponent conservative --timesteps 1000000

# Adım 3: RuleBased'e karşı ileri strateji öğren
python train_ppo.py --opponent rulebased --timesteps 2000000
```

### 2. Self-Play (Kendi Kendine Oynama)

Self-play için kod:

```python
from credit_war.gym_wrapper import CreditWarGymEnv
from credit_war.agents.ppo_agent import PPOAgent

# İlk modeli eğit
# ... (train_ppo.py ile)

# Eğitilmiş modeli rakip olarak kullan
trained_opponent = PPOAgent(
    model_path="models/ppo_rulebased_final.zip",
    name="PPO_Opponent"
)

# Yeni model bu rakibe karşı öğrenir
env = CreditWarGymEnv(opponent=trained_opponent, seed=42)
# ... (SB3 PPO training)
```

### 3. Multi-Agent RL (MARL)

Her iki ajan da aynı anda öğrenir (gelecekteki çalışma).

---

## 📈 Beklenen Öğrenme Eğrisi

### Phase 1: Random Exploration (0-100k steps)
- Win rate: 20-40%
- Model rastgele aksiyonlar dener
- **Aksiyon**: Sabırlı olun, loss yüksek olabilir

### Phase 2: Basic Strategy (100k-500k steps)
- Win rate: 40-60%
- GIVE_LOAN ve REJECT arasında denge öğrenir
- **Aksiyon**: Learning rate düşürmek için iyi zaman

### Phase 3: Advanced Tactics (500k-1M steps)
- Win rate: 60-75%
- UNDERCUT ve INSURE stratejilerini öğrenir
- **Aksiyon**: Checkpoint'leri kaydedin

### Phase 4: Mastery (1M+ steps)
- Win rate: 75%+
- Rakip modelleme ve uzun vadeli planlama
- **Aksiyon**: Farklı rakiplere karşı test edin

---

## 🐛 Troubleshooting

### Problem: Win Rate Artmıyor

**Çözüm 1**: Learning rate azalt
```bash
python train_ppo.py --lr 1e-4 --timesteps 2000000
```

**Çözüm 2**: Daha fazla timestep
```bash
python train_ppo.py --timesteps 5000000
```

**Çözüm 3**: Daha kolay rakip seç
```bash
python train_ppo.py --opponent random --timesteps 500000
```

### Problem: Training Çok Yavaş

**Çözüm 1**: GPU kullan (eğer mevcut)
```bash
# PyTorch CUDA yükle
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

**Çözüm 2**: Batch size artır
```python
# train_ppo.py içinde
model = PPO(..., batch_size=128, n_steps=4096)
```

### Problem: Policy Collapse (Hep Aynı Aksiyon)

**Çözüm**: Entropy coefficient artır
```python
# train_ppo.py içinde
model = PPO(..., ent_coef=0.05)  # 0.01'den 0.05'e
```

---

## 📚 Algoritma Detayları

### PPO Hyperparameters

| Parameter | Default | Açıklama |
|-----------|---------|----------|
| `learning_rate` | 3e-4 | Optimizer learning rate |
| `n_steps` | 2048 | Rollout buffer size |
| `batch_size` | 64 | Minibatch size |
| `n_epochs` | 10 | Optimization epochs per rollout |
| `gamma` | 0.99 | Discount factor |
| `gae_lambda` | 0.95 | GAE lambda |
| `clip_range` | 0.2 | PPO clip epsilon |
| `ent_coef` | 0.01 | Entropy coefficient |
| `vf_coef` | 0.5 | Value function coefficient |

### Network Architecture

**Policy Network (Actor):**
```
Input (12) → Dense(256) → ReLU → Dense(256) → ReLU → Output(5) → Softmax
```

**Value Network (Critic):**
```
Input (12) → Dense(256) → ReLU → Dense(256) → ReLU → Output(1)
```

**Total Parameters:** ~135,000

---

## 🎓 Akademik Kullanım

### Tez İçin Önerilen Deney Seti

```bash
# Deney 1: Baseline (RuleBased rakip)
python train_ppo.py --opponent rulebased --timesteps 2000000 --seed 42

# Deney 2: Adversarial (Aggressor rakip)
python train_ppo.py --opponent aggressor --timesteps 2000000 --seed 42

# Deney 3: Multi-seed (reproducibility)
for seed in 42 123 999; do
    python train_ppo.py --opponent rulebased --timesteps 1000000 --seed $seed
done

# Deney 4: Curriculum Learning
python train_ppo.py --opponent random --timesteps 500000 --seed 42
# (sonra modeli fine-tune et)
```

### Metrikler ve Raporlama

Training sırasında kayıt edilen metrikler:
- `rollout/win_rate` - Kazanma oranı
- `rollout/loss_rate` - Kaybetme oranı
- `rollout/draw_rate` - Beraberlik oranı
- `rollout/ep_len_mean` - Ortalama episode uzunluğu
- `train/learning_rate` - Güncel learning rate
- `train/entropy_loss` - Policy entropy

Tensorboard ile dışa aktarma:
```bash
tensorboard --logdir tensorboard_logs/ --logdir_spec=PPO:tensorboard_logs/PPO_rulebased_1
```

---

## 🚀 İleri Adımlar

### 1. Multi-Agent PPO
Her iki ajan da aynı anda öğrenir (karmaşık!)

### 2. DQN Algoritması
Discrete action space için alternatif

### 3. Partial Observability
Rakibin state'ini gizle (daha zor!)

### 4. Transfer Learning
Bir rakibe karşı öğrendiğini başka rakibe transfer et

---

## 📖 Referanslar

- **PPO Algorithm**: [Schulman et al., 2017](https://arxiv.org/abs/1707.06347)
- **Stable-Baselines3**: [Documentation](https://stable-baselines3.readthedocs.io/)
- **Gymnasium**: [Documentation](https://gymnasium.farama.org/)

---

**Başarılar! 🎉**

Sorularınız için: CREDIT WAR GitHub Issues
