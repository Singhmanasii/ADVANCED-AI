import numpy as np
import random

# Define the environment
grid_size = 5
goal = (4, 4)
obstacles = [(2, 2), (3, 3)]  # Define obstacles

# Q-learning parameters
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 0.1  # Exploration rate
episodes = 1000

# Initialize the Q-table: (grid_size, grid_size, 4 actions)
q_table = np.zeros((grid_size, grid_size, 4))

# Define actions (Up, Down, Left, Right)
actions = {
    0: (-1, 0),  # Up
    1: (1, 0),   # Down
    2: (0, -1),  # Left
    3: (0, 1)    # Right
}

def is_valid_position(pos):
    """Check if the position is valid (not out of bounds or in an obstacle)."""
    x, y = pos
    if x < 0 or x >= grid_size or y < 0 or y >= grid_size:
        return False
    if pos in obstacles:
        return False
    return True

def get_next_state(state, action):
    """Get the next state after taking an action."""
    x, y = state
    dx, dy = actions[action]
    next_state = (x + dx, y + dy)
    return next_state if is_valid_position(next_state) else state  # Stay if move is invalid

def get_reward(state):
    """Return the reward for a given state."""
    if state == goal:
        return 10
    if state in obstacles:
        return -10
    return -1  # Small penalty for every move

# Training the agent
for episode in range(episodes):
    state = (0, 0)  # Start position
    total_reward = 0

    while state != goal:
        # Choose an action (epsilon-greedy strategy)
        if random.uniform(0, 1) < epsilon:
            action = random.choice(list(actions.keys()))  # Explore
        else:
            action = np.argmax(q_table[state[0], state[1]])  # Exploit (choose best known action)

        # Take the action
        next_state = get_next_state(state, action)
        reward = get_reward(next_state)

        # Update Q-value using Q-learning formula
        old_value = q_table[state[0], state[1], action]
        next_max = np.max(q_table[next_state[0], next_state[1]])
        q_table[state[0], state[1], action] = old_value + alpha * (reward + gamma * next_max - old_value)

        state = next_state
        total_reward += reward

    # Print progress every 100 episodes
    if (episode + 1) % 100 == 0:
        print(f"Episode {episode + 1}, Total Reward: {total_reward}")

# Test the trained agent
state = (0, 0)
path = [state]

while state != goal:
    action = np.argmax(q_table[state[0], state[1]])
    state = get_next_state(state, action)
    path.append(state)

print("Optimal Path:", path)
