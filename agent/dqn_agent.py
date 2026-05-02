import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import math

from .dqn_network import DQN
from .replay_buffer import ReplayBuffer, Transition

class DQNAgent:
    """
    DQN Agent encapsulating the policy net, target net, memory, and optimization.
    """
    def __init__(self, state_dim, action_dim, config):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Networks
        self.policy_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config['learning_rate'])
        self.memory = ReplayBuffer(config['memory_size'])

        self.steps_done = 0

    def select_action(self, state, env):
        """
        Epsilon-greedy action selection.
        """
        eps_threshold = self.config['eps_end'] + (self.config['eps_start'] - self.config['eps_end']) * \
            math.exp(-1. * self.steps_done / self.config['eps_decay'])
        self.steps_done += 1

        if random.random() > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                return self.policy_net(state_tensor).max(1)[1].item()
        else:
            return env.action_space.sample()

    def optimize_model(self):
        """
        Perform a single step of optimization on the policy network.
        """
        if len(self.memory) < self.config['batch_size']:
            return 0.0 # Not enough samples to optimize yet

        transitions = self.memory.sample(self.config['batch_size'])
        # Transpose the batch
        batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        next_state_batch = torch.cat(batch.next_state)
        done_batch = torch.cat(batch.done)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based on the "older" target_net
        next_state_values = torch.zeros(self.config['batch_size'], device=self.device)
        
        with torch.no_grad():
            next_state_values = self.target_net(next_state_batch).max(1)[0]
            # Mask out terminal states
            next_state_values = next_state_values * (1 - done_batch)

        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.config['gamma']) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()
        
        return loss.item()

    def update_target_network(self):
        """Copy weights from policy net to target net."""
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
    def save(self, filepath):
        """Save the policy network."""
        torch.save(self.policy_net.state_dict(), filepath)
