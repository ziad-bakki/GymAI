import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

# Initialize the Acrobot environment
env_id = "Acrobot-v1"
env = gym.make(env_id, render_mode="rgb_array")

# Initialize the PPO model
model = PPO(
    "MlpPolicy",  # Multi-Layer Perceptron policy
    env,
    verbose=1,
    learning_rate=0.0003,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
)

# Train the model
print("Training the PPO model...")
model.learn(total_timesteps=100000)

# Save the trained model
model.save("ppo_acrobot_model")

# Evaluate the model
print("Evaluating the trained model...")
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward}, Std reward: {std_reward}")

# Test the trained model with rendering
obs, _ = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    env.render()
    if done or truncated:
        obs, _ = env.reset()

env.close()
