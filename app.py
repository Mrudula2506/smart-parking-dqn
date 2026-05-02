import streamlit as st
import numpy as np
import torch
import time
import os
import sys
import math

# Ensure imports work from the project root
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from env.parking_env import SmartParkingEnv
from agent.dqn_network import DQN
from config.config import ENVIRONMENT_CONFIG

# --- UI Configuration ---
st.set_page_config(page_title="Smart Parking RL", layout="wide", page_icon="🚗")
st.title("🚗 Smart Parking Slot Allocation - RL Agent")
st.markdown("Watch the trained Deep Q-Network allocate parking spots dynamically!")

# --- Sidebar Controls ---
st.sidebar.header("Environment Settings")
simulation_speed = st.sidebar.slider("Simulation Speed (s/step)", 0.1, 2.0, 0.5)
depart_prob = st.sidebar.slider("Departure Probability (Turnover)", 0.0, 0.5, 0.1, help="Higher = cars leave more often (Low Congestion). Lower = Peak Hour Congestion.")
max_steps = st.sidebar.number_input("Max Cars Entering (Episode length)", 10, 500, 50)

if st.sidebar.button("Run Simulation"):
    # Load Model
    model_path = os.path.join(os.path.dirname(__file__), 'training', 'saved_models', 'dqn_parking.pth')
    if not os.path.exists(model_path):
        st.error(f"Trained model not found at `{model_path}`. Please run `python training/train.py` first.")
        st.stop()

    # Initialize Config & Environment
    config = ENVIRONMENT_CONFIG.copy()
    config['depart_prob'] = depart_prob
    config['max_steps'] = int(max_steps)
    
    env = SmartParkingEnv(config)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    policy_net = DQN(state_dim, action_dim).to(device)
    policy_net.load_state_dict(torch.load(model_path, map_location=device))
    policy_net.eval()

    # Layout Placeholders
    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.subheader("Parking Lot Grid")
        grid_placeholder = st.empty()
        
    with col2:
        st.subheader("🧠 Intelligence Metrics")
        metrics_placeholder = st.empty()
        st.markdown("""
        ---
        **How to know if the Agent is "Smart"?**
        1. **High Accuracy (~95-100%):** A random agent crashes into already parked cars. Our RL agent learns to skip occupied slots and achieves high valid placement rates.
        2. **Low Average Distance:** The RL agent learns that parking closer to `Gate` (P0) gets a higher reward. Notice how cars cluster near the gate!
        3. **Congestion Balancing:** It balances distance with spacing cars out (to avoid bumper-to-bumper congestion penalties).
        """)

    st.markdown("### Activity Log")
    log_placeholder = st.empty()

    # Simulation Loop
    state = env.reset()
    done = False
    total_reward = 0
    step = 0
    logs = []
    
    # Intelligence Tracking
    valid_placements = 0
    invalid_placements = 0
    total_distance_driven = 0.0

    while not done:
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        
        with torch.no_grad():
            action = policy_net(state_tensor).max(1)[1].item()
            
        next_state, reward, done, info = env.step(action)
        
        state = next_state
        total_reward += reward
        step += 1
        
        grid_size = config['grid_size']
        grid_2d = state.reshape((grid_size, grid_size))
        
        # 1. Update Grid Visualization (Using HTML/CSS for a proper graphical UI)
        html_grid = f"<h4>Grid View</h4><div style='display: grid; grid-template-columns: repeat({grid_size}, 1fr); gap: 8px; width: 100%; max-width: 400px; padding: 10px; background-color: #1e1e1e; border-radius: 10px;'>"
        for r in range(grid_size):
            for c in range(grid_size):
                is_gate = (r, c) == tuple(config['entrance_pos'])
                is_occupied = grid_2d[r, c] == 1
                
                # Base styling
                bg_color = "#2b313e"  # Empty
                border = "2px dashed #4CAF50"
                icon = "🅿️"
                label_color = "#888"
                
                if is_gate:
                    border = "2px solid #2196F3"
                    bg_color = "#15202b"
                    icon = "🚙" if is_occupied else "⛩️"
                    label_color = "#2196F3"
                elif is_occupied:
                    bg_color = "#3e2b2b"
                    border = "2px solid #F44336"
                    icon = "🚘"
                    label_color = "#F44336"
                
                slot_id = r * grid_size + c
                label = "Gate" if is_gate else f"P{slot_id}"
                
                html_grid += f"""<div style="background-color: {bg_color}; border: {border}; border-radius: 6px; height: 70px; display: flex; flex-direction: column; align-items: center; justify-content: center; box-shadow: 2px 2px 5px rgba(0,0,0,0.3);"><div style="font-size: 0.7rem; font-weight: bold; color: {label_color}; margin-bottom: 4px;">{label}</div><div style="font-size: 1.5rem;">{icon}</div></div>"""
        html_grid += "</div>"
            
        grid_placeholder.markdown(html_grid, unsafe_allow_html=True)

        # 2. Update Metrics
        occupancy = np.sum(state)
        capacity = state_dim
        occupancy_rate = (occupancy / capacity) * 100
        
        # Calculate intelligence statistics
        row, col = env._get_slot_coord(action)
        dist_to_gate = math.sqrt((row - config['entrance_pos'][0])**2 + (col - config['entrance_pos'][1])**2)
        
        is_valid = info.get('valid_action', False)
        if is_valid:
            valid_placements += 1
            total_distance_driven += dist_to_gate
        else:
            invalid_placements += 1
            
        accuracy = (valid_placements / max(1, valid_placements + invalid_placements)) * 100
        avg_dist = (total_distance_driven / max(1, valid_placements))
        
        metrics_placeholder.markdown(f"""
        **System Performance**
        - **Simulation Step:** {step}/{config['max_steps']}
        - **Parking Occupancy:** {int(occupancy)}/{capacity} ({occupancy_rate:.1f}%)
        - **Total Agent Reward:** {total_reward:.2f}
        
        **Real-Time Brain (Why it's Smart)**
        - **Last Assigned Slot:** `P{action}` (Distance {dist_to_gate:.1f})
        - **Valid Placement?:** {'✅ Yes (Smart Avoidance)' if is_valid else '❌ Collision (Penalty)'}
        - **Overall Setup Accuracy:** <span style="background:#2b2b2b;color:#4CAF50;padding:2px 8px;border-radius:4px;font-weight:bold;">{accuracy:.1f} %</span> *(Random is < 20%)*
        - **Avg Distance from Gate:** <span style="background:#2b2b2b;color:#2196F3;padding:2px 8px;border-radius:4px;font-weight:bold;">{avg_dist:.2f}</span> *(Low represents intent)*
        """, unsafe_allow_html=True)
        
        # 3. Update Logs
        if is_valid:
            logs.insert(0, f"Step {step}: Car assigned to ✨ Slot P{action} (Dist {dist_to_gate:.1f}m). Reward: +{reward:.2f}")
        else:
            logs.insert(0, f"Step {step}: Collision! Agent blindly tried occupied Slot P{action}. Penalty applied.")
            
        log_placeholder.code("\n".join(logs[:15])) # Keep last 15 logs
        
        time.sleep(simulation_speed)

    st.success("Simulation Complete!")
    if 'lot_full' in info:
        st.warning("Parking lot reached 100% capacity!")
