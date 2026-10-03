# Q4. Display states, actions, rewards, and termination conditions.
import gymnasium as gym

env = gym.make("FrozenLake-v1", is_slippery=False)
state, info = env.reset(seed=1)
print("Initial state:", state)

for i in range(10):
    action = env.action_space.sample()
    next_state, reward, terminated, truncated, info = env.step(action)
    print(f"Step {i+1}: state={state}, action={action}, next_state={next_state}, "
          f"reward={reward}, terminated={terminated}, truncated={truncated}")
    state = next_state
    if terminated or truncated:
        break

env.close()
