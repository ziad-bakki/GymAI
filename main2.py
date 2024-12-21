import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

# Initialize the environment
env = gym.make("CartPole-v1", render_mode="human")

# Wrap the environment with Monitor for evaluation
eval_env = Monitor(env)

# Initialize the DQN model
model = DQN("MlpPolicy", env, learning_rate=0.001, verbose=1)

# Train the model
model.learn(total_timesteps=100000)

# Evaluate the trained model
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward}, Std reward: {std_reward}")

# Test the trained model
obs, _ = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, _ = env.step(action)
    env.render()
    if done or truncated:
        obs, _ = env.reset()

env.close()


# episodes = 10
# for episode in range(1, episodes+1):
#     state = env.reset()
#     done = False
#     score = 0
#     while not done:
#         action = random.choice([0,1])
#         state, reward, done, truncated, _ = env.step(action)
#         score += reward
#         env.render()


#     print(f"Episode {episode}, Score: {score}")

    