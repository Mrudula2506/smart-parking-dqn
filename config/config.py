# Baseline configurations for the environment and the training loop

ENVIRONMENT_CONFIG = {
    "grid_size": 5,          # Scalable grid size (5x5 = 25 slots)
    "max_steps": 200,        # Number of cars entering per episode
    "depart_prob": 0.1,      # Represents parking turnover (dynamic demand simulation)
    "entrance_pos": (0, 0)   # Gate position
}

RL_CONFIG = {
    "batch_size": 64,
    "gamma": 0.99,           # Discount factor
    "eps_start": 1.0,        # Initial exploration rate
    "eps_end": 0.01,         # Final exploration rate
    "eps_decay": 500,        # DECREASED decay speed (so agent explores state space more thoroughly)
    "target_update": 10,     # Update Target Network every x episodes
    "learning_rate": 1e-3,
    "memory_size": 10000,    # Replay buffer size
    "num_episodes": 1500     # TRIPLED: Train longer to guarantee the math converges perfectly
}
