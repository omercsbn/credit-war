"""
Interactive CREDIT WAR Demo - Human vs AI

Streamlit web app: Kullanıcı AI ajanlarına karşı canlı oynar.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from typing import List, Dict, Tuple
import os
import glob

from credit_war.env import CreditWarEnv
from credit_war.actions import Action
from credit_war.agents import (
    RandomAgent,
    GreedyAgent,
    ConservativeAgent,
    RuleBasedAgent,
    AggressorAgent,
)
from credit_war.agents.ppo_agent import PPOAgent


# Page config
st.set_page_config(
    page_title="CREDIT WAR - Human vs AI",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stButton>button {
        width: 100%;
        height: 60px;
        font-size: 16px;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .win-banner {
        background-color: #d4edda;
        color: #155724;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
    }
    .lose-banner {
        background-color: #f8d7da;
        color: #721c24;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
    }
    .draw-banner {
        background-color: #fff3cd;
        color: #856404;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


# Action descriptions
ACTION_INFO = {
    Action.GIVE_LOAN: {
        "name": "🏦 GIVE LOAN",
        "description": "Issue risky loan for high return (High risk, high reward)",
        "emoji": "📈",
        "color": "#ff6b6b"
    },
    Action.REJECT: {
        "name": "🚫 REJECT",
        "description": "Decline loan; reduce risk (Safe, defensive)",
        "emoji": "🛡️",
        "color": "#4ecdc4"
    },
    Action.INVEST: {
        "name": "💰 INVEST",
        "description": "Safe investment; moderate growth (Balanced)",
        "emoji": "📊",
        "color": "#95e1d3"
    },
    Action.INSURE: {
        "name": "🔒 INSURE",
        "description": "Purchase risk mitigation (Protect yourself)",
        "emoji": "🛡️",
        "color": "#f38181"
    },
    Action.UNDERCUT: {
        "name": "⚔️ UNDERCUT",
        "description": "Sabotage opponent's portfolio (Aggressive attack)",
        "emoji": "💥",
        "color": "#aa96da"
    }
}


def find_trained_models(base_dir: str = "./models/game_theory") -> Dict[str, Dict]:
    """Find all trained game theory models."""
    models = {}
    
    if not os.path.exists(base_dir):
        return models
    
    for agent_dir in os.listdir(base_dir):
        agent_path = os.path.join(base_dir, agent_dir)
        if not os.path.isdir(agent_path):
            continue
        
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


def load_agent(agent_type: str, models_dict: Dict):
    """Load an agent based on type."""
    if agent_type == "Random":
        return RandomAgent(name="AI_Random", seed=42)
    elif agent_type == "Greedy":
        return GreedyAgent(name="AI_Greedy", seed=42)
    elif agent_type == "Conservative":
        return ConservativeAgent(name="AI_Conservative", seed=42)
    elif agent_type == "RuleBased":
        return RuleBasedAgent(name="AI_RuleBased", seed=42)
    elif agent_type == "Aggressor":
        return AggressorAgent(name="AI_Aggressor", seed=42)
    else:
        # PPO agents
        agent_key = agent_type.lower().replace(" ", "").replace("-", "")
        if agent_key in models_dict:
            return PPOAgent(
                model_path=models_dict[agent_key]["model_path"],
                vec_normalize_path=models_dict[agent_key]["vecnormalize_path"],
                name=f"AI_{agent_type}",
                deterministic=True
            )
    
    return RandomAgent(name="AI_Default", seed=42)


def initialize_game():
    """Initialize game state."""
    st.session_state.env = CreditWarEnv(seed=42)
    st.session_state.game_started = True
    st.session_state.game_over = False
    st.session_state.turn_history = []
    st.session_state.action_history = {"human": [], "ai": []}
    st.session_state.winner = None
    st.session_state.outcome_message = ""


def display_agent_state(agent_state, title: str, is_opponent: bool = False):
    """Display agent state metrics."""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "💰 Capital",
            f"${agent_state.capital:.1f}",
            delta=None
        )
    
    with col2:
        risk_color = "🔴" if agent_state.risk >= 7 else "🟡" if agent_state.risk >= 5 else "🟢"
        st.metric(
            f"{risk_color} Risk",
            f"{agent_state.risk:.1f}",
            delta=None
        )
    
    with col3:
        st.metric(
            "💧 Liquidity",
            f"{agent_state.liquidity:.1f}",
            delta=None
        )
    
    # Pending inflows
    st.write(f"**📅 Pending Inflows:** P1: {agent_state.pending_inflows[0]:.1f} | "
             f"P2: {agent_state.pending_inflows[1]:.1f} | "
             f"P3: {agent_state.pending_inflows[2]:.1f}")


def create_capital_chart(history: List[Dict]):
    """Create capital history chart."""
    if not history:
        return None
    
    turns = [h["turn"] for h in history]
    human_capital = [h["human_capital"] for h in history]
    ai_capital = [h["ai_capital"] for h in history]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=turns,
        y=human_capital,
        mode='lines+markers',
        name='You (Human)',
        line=dict(color='#4ecdc4', width=3),
        marker=dict(size=8)
    ))
    
    fig.add_trace(go.Scatter(
        x=turns,
        y=ai_capital,
        mode='lines+markers',
        name='AI Opponent',
        line=dict(color='#ff6b6b', width=3),
        marker=dict(size=8)
    ))
    
    fig.update_layout(
        title="Capital Over Time",
        xaxis_title="Turn",
        yaxis_title="Capital ($)",
        hovermode='x unified',
        height=400
    )
    
    return fig


def main():
    """Main Streamlit app."""
    
    # Title
    st.title("🏦 CREDIT WAR - Human vs AI")
    st.markdown("**Test your banking strategy against AI opponents!**")
    
    # Sidebar - Agent selection
    st.sidebar.header("⚙️ Game Settings")
    
    # Find trained models
    trained_models = find_trained_models()
    
    # Available opponents
    rule_based_opponents = [
        "Random",
        "Greedy", 
        "Conservative",
        "RuleBased",
        "Aggressor"
    ]
    
    ppo_opponents = [
        "Nash Equilibrium",
        "Tit-for-Tat",
        "Grim Trigger",
        "Bayesian",
        "Predator",
        "Minimax",
        "Switcher",
        "Evolutionary",
        "Zero-Sum",
        "Meta-Learner"
    ]
    
    # Only show PPO agents if models exist
    if trained_models:
        all_opponents = rule_based_opponents + ["---"] + ppo_opponents
    else:
        all_opponents = rule_based_opponents
    
    opponent_choice = st.sidebar.selectbox(
        "Select AI Opponent",
        all_opponents,
        help="Choose which AI strategy to play against"
    )
    
    if opponent_choice == "---":
        st.sidebar.info("⬇️ Trained PPO agents below")
        st.stop()
    
    # Opponent info
    st.sidebar.markdown("---")
    st.sidebar.subheader(f"🤖 Opponent: {opponent_choice}")
    
    opponent_descriptions = {
        "Random": "Makes random decisions",
        "Greedy": "Always takes highest immediate reward",
        "Conservative": "Minimizes risk at all costs",
        "RuleBased": "Balanced rule-based strategy",
        "Aggressor": "Attacks when opponent is weak",
        "Nash Equilibrium": "Seeks mixed strategies and balance",
        "Tit-for-Tat": "Mirrors your behavior, forgives once",
        "Grim Trigger": "Cooperates until betrayal, then permanent punishment",
        "Bayesian": "Updates beliefs and exploits patterns (STRONGEST!)",
        "Predator": "Hunts weak opponents aggressively",
        "Minimax": "Minimizes worst-case loss",
        "Switcher": "Alternates between regimes unpredictably",
        "Evolutionary": "Imitates successful strategies",
        "Zero-Sum": "Maximizes relative advantage",
        "Meta-Learner": "Classifies opponents and adapts"
    }
    
    st.sidebar.info(opponent_descriptions.get(opponent_choice, "Unknown opponent"))
    
    # Start/Restart button
    st.sidebar.markdown("---")
    if st.sidebar.button("🎮 Start New Game", type="primary"):
        initialize_game()
        st.session_state.opponent = load_agent(opponent_choice, trained_models)
        st.rerun()
    
    # Check if game started
    if not hasattr(st.session_state, 'game_started') or not st.session_state.game_started:
        st.info("👈 Select an opponent and click **Start New Game** to begin!")
        
        # Show game rules
        st.markdown("---")
        st.subheader("📜 How to Play")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **🎯 Goal:**
            - Maximize your capital
            - Manage risk carefully
            - Outlast your opponent
            
            **💡 Strategy Tips:**
            - High risk = potential failure
            - Liquidity = flexibility
            - Timing is everything
            """)
        
        with col2:
            st.markdown("""
            **🎬 Actions:**
            - **GIVE LOAN**: High risk, high reward
            - **REJECT**: Reduce risk, play safe
            - **INVEST**: Moderate growth
            - **INSURE**: Risk mitigation
            - **UNDERCUT**: Attack opponent
            """)
        
        st.stop()
    
    # Game is running
    env = st.session_state.env
    state = env.state
    
    # Display game state
    if not st.session_state.game_over:
        st.markdown(f"### 🎮 Turn {state.turn}")
        
        # Display both players
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 👤 You (Agent A)")
            display_agent_state(state.agent_a, "Your State")
        
        with col2:
            st.markdown(f"#### 🤖 {opponent_choice} (Agent B)")
            display_agent_state(state.agent_b, "AI State", is_opponent=True)
        
        st.markdown("---")
        
        # Action selection
        st.subheader("🎯 Choose Your Action")
        
        # Get valid actions for human
        valid_actions = env.get_valid_actions(state.agent_a)
        
        cols = st.columns(5)
        
        for idx, action in enumerate([Action.GIVE_LOAN, Action.REJECT, Action.INVEST, Action.INSURE, Action.UNDERCUT]):
            with cols[idx]:
                info = ACTION_INFO[action]
                
                disabled = action not in valid_actions
                
                if st.button(
                    f"{info['emoji']} {action.name}",
                    key=f"action_{action.name}",
                    disabled=disabled,
                    help=info['description']
                ):
                    # Human action selected
                    human_action = action
                    
                    # AI makes decision
                    ai_valid_actions = env.get_valid_actions(state.agent_b)
                    ai_action = st.session_state.opponent.select_action(
                        own_state=state.agent_b,
                        opponent_state=state.agent_a,
                        valid_actions=ai_valid_actions
                    )
                    
                    # Execute turn
                    new_state, reward_a, reward_b, done, info = env.step(human_action, ai_action)
                    
                    # Record history
                    st.session_state.turn_history.append({
                        "turn": state.turn,
                        "human_action": human_action.name,
                        "ai_action": ai_action.name,
                        "human_capital": new_state.agent_a.capital,
                        "ai_capital": new_state.agent_b.capital,
                        "human_risk": new_state.agent_a.risk,
                        "ai_risk": new_state.agent_b.risk
                    })
                    
                    st.session_state.action_history["human"].append(human_action.name)
                    st.session_state.action_history["ai"].append(ai_action.name)
                    
                    # Check if game over
                    if done:
                        st.session_state.game_over = True
                        outcome = info.get("outcome", "draw")
                        
                        if outcome == "agent_a_wins":
                            st.session_state.winner = "human"
                            st.session_state.outcome_message = "🎉 Congratulations! You WIN!"
                        elif outcome == "agent_b_wins":
                            st.session_state.winner = "ai"
                            st.session_state.outcome_message = f"😞 {opponent_choice} WINS!"
                        else:
                            st.session_state.winner = "draw"
                            st.session_state.outcome_message = "🤝 It's a DRAW!"
                    
                    st.rerun()
                
                if disabled:
                    st.caption("❌ Invalid")
    
    else:
        # Game over - show results
        st.markdown("---")
        
        if st.session_state.winner == "human":
            st.markdown(f'<div class="win-banner">🎉 {st.session_state.outcome_message}</div>', 
                       unsafe_allow_html=True)
        elif st.session_state.winner == "ai":
            st.markdown(f'<div class="lose-banner">😞 {st.session_state.outcome_message}</div>', 
                       unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="draw-banner">🤝 {st.session_state.outcome_message}</div>', 
                       unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Final stats
        col1, col2, col3 = st.columns(3)
        
        final_turn = st.session_state.turn_history[-1]
        
        with col1:
            st.metric("🏁 Total Turns", len(st.session_state.turn_history))
        
        with col2:
            st.metric("💰 Your Final Capital", f"${final_turn['human_capital']:.1f}")
        
        with col3:
            st.metric("🤖 AI Final Capital", f"${final_turn['ai_capital']:.1f}")
        
        # Capital chart
        st.subheader("📈 Capital History")
        fig = create_capital_chart(st.session_state.turn_history)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        
        # Action history
        st.subheader("📋 Action History")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Your Actions:**")
            action_counts = pd.Series(st.session_state.action_history["human"]).value_counts()
            st.bar_chart(action_counts)
        
        with col2:
            st.markdown(f"**{opponent_choice}'s Actions:**")
            ai_action_counts = pd.Series(st.session_state.action_history["ai"]).value_counts()
            st.bar_chart(ai_action_counts)
        
        # Turn-by-turn details
        with st.expander("📊 Detailed Turn History"):
            df = pd.DataFrame(st.session_state.turn_history)
            st.dataframe(df, use_container_width=True)
    
    # Sidebar stats during game
    if hasattr(st.session_state, 'turn_history') and st.session_state.turn_history:
        st.sidebar.markdown("---")
        st.sidebar.subheader("📊 Game Stats")
        st.sidebar.metric("Turns Played", len(st.session_state.turn_history))
        
        if len(st.session_state.turn_history) > 0:
            latest = st.session_state.turn_history[-1]
            st.sidebar.metric("Your Capital", f"${latest['human_capital']:.1f}")
            st.sidebar.metric("Your Risk", f"{latest['human_risk']:.1f}")


if __name__ == "__main__":
    main()
