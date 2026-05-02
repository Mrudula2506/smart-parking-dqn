import numpy as np
import gym
from gym import spaces
import math

class SmartParkingEnv(gym.Env):
    """
    Custom Gym Environment for Smart Parking System using Reinforcement Learning
    Grid-based parking lot mapping.
    """
    def __init__(self, config=None):
        super(SmartParkingEnv, self).__init__()
        
        if config is None:
            config = {}
        self.config = config
        
        # Grid dimensions
        self.grid_size = config.get('grid_size', 5) # 5x5 by default
        self.num_slots = self.grid_size * self.grid_size
        self.max_steps = config.get('max_steps', 200)
        self.depart_prob = config.get('depart_prob', 0.1) # Probability a car leaves
        self.entrance_pos = config.get('entrance_pos', (0, 0)) # Entrance is Top-Left

        # Action Space: Assign current car to one of the available slots
        self.action_space = spaces.Discrete(self.num_slots)

        # State Space: Grid occupancy (0=empty, 1=occupied)
        # Flattened 1D array representing the grid
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(self.num_slots,), dtype=np.float32
        )

        # Internal state
        self.grid = np.zeros(self.num_slots, dtype=np.float32)
        self.current_step = 0

    def reset(self):
        """Reset the environment to an empty grid at the start of a new episode."""
        self.grid = np.zeros(self.num_slots, dtype=np.float32)
        self.current_step = 0
        return self._get_obs()

    def _get_obs(self):
        """Return the current state."""
        return self.grid.copy()

    def _get_slot_coord(self, slot_id):
        """Convert a 1D slot index to 2D grid coordinates."""
        row = slot_id // self.grid_size
        col = slot_id % self.grid_size
        return (row, col)

    def _get_neighbors(self, slot_id):
        """Find adjacent slots to calculate congestion penalty."""
        row, col = self._get_slot_coord(slot_id)
        neighbors = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)] # Up, Down, Left, Right
        for dr, dc in directions:
            r, c = row + dr, col + dc
            if 0 <= r < self.grid_size and 0 <= c < self.grid_size:
                neighbors.append(r * self.grid_size + c)
        return neighbors

    def step(self, action):
        """
        Execute one action (assigning a slot).
        """
        self.current_step += 1
        done = self.current_step >= self.max_steps

        reward = 0
        info = {"valid_action": True}

        # Multi-objective Reward Design
        if self.grid[action] == 1:
            # 1. Invalid Action Penalty: Tried to park in an occupied slot
            reward -= 50.0  # Increased massively to force agent to stop colliding
            info["valid_action"] = False
        else:
            # Successfully parked
            self.grid[action] = 1
            
            # Base fairness/success reward
            reward += 10.0
            
            # 2. Distance Penalty: Encourage parking closer to the entrance
            row, col = self._get_slot_coord(action)
            dist = math.sqrt((row - self.entrance_pos[0])**2 + (col - self.entrance_pos[1])**2)
            reward -= (dist * 1.5)  # Tripled! Agent MUST prioritize closer slots now

            # 3. Congestion Penalty: Discourage parking next to other cars to prevent bottleneck
            neighbors = self._get_neighbors(action)
            congestion = sum([self.grid[n] for n in neighbors])
            reward -= (congestion * 0.2)  # Decreased significantly so it doesn't run away just to be alone

        # 4. Dynamic Parking Demand Simulation (Cars leaving)
        self._simulate_departures()

        # Check if the lot is completely full
        if np.sum(self.grid) == self.num_slots:
            done = True
            info["lot_full"] = True

        return self._get_obs(), reward, done, info

    def _simulate_departures(self):
        """Simulate dynamic demand -> Randomly free up slots as cars leave."""
        for i in range(self.num_slots):
            if self.grid[i] == 1 and np.random.rand() < self.depart_prob:
                self.grid[i] = 0
