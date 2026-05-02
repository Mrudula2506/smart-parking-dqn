import os
import sys
import torch
import numpy as np
import math

# Add the project root to the python path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.parking_env import SmartParkingEnv
from agent.dqn_agent import DQNAgent
from config.config import ENVIRONMENT_CONFIG, RL_CONFIG

def train():
    print("Initializing Environment and Agent...")
    env = SmartParkingEnv(ENVIRONMENT_CONFIG)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = DQNAgent(state_dim, action_dim, RL_CONFIG)
    
    num_episodes = RL_CONFIG['num_episodes']
    target_update = RL_CONFIG['target_update']
    
    metrics = {
        'rewards': [],
        'losses': [],
        'steps': []
    }
    
    print(f"Starting training for {num_episodes} episodes...")
    
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        total_loss = 0
        steps = 0
        done = False
        
        while not done:
            action = agent.select_action(state, env)
            next_state, reward, done, info = env.step(action)
            
            # Convert values to tensors for the replay buffer
            state_t = torch.tensor(state, dtype=torch.float32, device=agent.device).unsqueeze(0)
            action_t = torch.tensor([[action]], device=agent.device, dtype=torch.long)
            reward_t = torch.tensor([reward], dtype=torch.float32, device=agent.device)
            next_state_t = torch.tensor(next_state, dtype=torch.float32, device=agent.device).unsqueeze(0)
            done_t = torch.tensor([float(done)], dtype=torch.float32, device=agent.device)
            
            # Store the transition in memory
            agent.memory.push(state_t, action_t, reward_t, next_state_t, done_t)
            
            state = next_state
            total_reward += reward
            steps += 1
            
            # Perform one step of optimization
            loss = agent.optimize_model()
            total_loss += loss
            
        # Store metrics
        metrics['rewards'].append(total_reward)
        metrics['losses'].append(total_loss / steps if steps > 0 else 0)
        metrics['steps'].append(steps)
        
        # Update the target network
        if episode % target_update == 0:
            agent.update_target_network()
            
        # Logging progress
        if (episode + 1) % 50 == 0:
            current_eps = RL_CONFIG['eps_end'] + (RL_CONFIG['eps_start'] - RL_CONFIG['eps_end']) * \
                math.exp(-1. * agent.steps_done / RL_CONFIG['eps_decay'])
            avg_loss = metrics['losses'][-1]
            print(f"Episode {episode+1:3d}/{num_episodes} | Reward: {total_reward:7.2f} | Steps: {steps:3d} | Avg Loss: {avg_loss:6.4f} | Epsilon: {current_eps:.3f}")

    print("Training complete.")
    
    # Create directory for saving the model
    os.makedirs(os.path.join(os.path.dirname(__file__), 'saved_models'), exist_ok=True)
    
    # Save the trained policy network
    model_path = os.path.join(os.path.dirname(__file__), 'saved_models', 'dqn_parking.pth')
    agent.save(model_path)
    print(f"Model saved to {model_path}")
    
    # Save training metrics for visualization
    metrics_path = os.path.join(os.path.dirname(__file__), 'metrics.npy')
    np.save(metrics_path, metrics)
    print(f"Metrics saved to {metrics_path}")

if __name__ == "__main__":
    train()