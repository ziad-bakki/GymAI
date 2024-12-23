import os
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy

# Paths for saving/loading the model and VecNormalize statistics
MODEL_PATH = "dqn_cartpole_model.zip"
VECNORM_PATH = "vecnormalize_stats.pkl"
 
# Initialize the environment
def make_env():
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    env = Monitor(env)
    return env

# Create the environment
env = DummyVecEnv([make_env])

# Check if VecNormalize statistics exist
if os.path.exists(VECNORM_PATH):
    env = VecNormalize.load(VECNORM_PATH, env)
else:
    env = VecNormalize(env, norm_obs=True, norm_reward=True)

# Two hidden layers with 128 neurons each
policy_kwargs = dict(
    net_arch=[128, 128]
)

# Load the model if it exists, otherwise create a new one
if os.path.exists(MODEL_PATH):
    print("Loading existing model...")
    model = DQN.load(MODEL_PATH, env=env)
else:
    print("Creating new model...")
    model = DQN("MlpPolicy", env, learning_rate=0.00001, policy_kwargs=policy_kwargs, verbose=1)

# Train the model further
model.learn(total_timesteps=500000)

# Save the model and VecNormalize statistics
model.save(MODEL_PATH)
env.save(VECNORM_PATH)

# Create evaluation environment and load VecNormalize statistics
eval_env = DummyVecEnv([make_env])
eval_env = VecNormalize.load(VECNORM_PATH, eval_env)

# Disable training updates for evaluation
eval_env.training = False
eval_env.norm_reward = False

# Evaluate the trained model
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward}, Std reward: {std_reward}")

# Test the trained model with rendering
test_env = DummyVecEnv([make_env])
test_env = VecNormalize.load(VECNORM_PATH, test_env)
test_env.training = False
test_env.norm_reward = False

obs = test_env.reset()
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = test_env.step(action)
    test_env.render()
    if done[0]:
        obs = test_env.reset()

test_env.close()
