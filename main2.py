import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy

# Initialize the environment
def make_env():
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    env = Monitor(env)
    return env

env = DummyVecEnv([make_env])

env = VecNormalize(env, norm_obs=True, norm_reward=True)

# Two hidden layers with 128 neurons each
policy_kwargs = dict(
    net_arch=[128, 128]
)

# Initialize the DQN model
model = DQN("MlpPolicy", env, learning_rate=0.0001, policy_kwargs=policy_kwargs, verbose=1)

# Train the model
model.learn(total_timesteps=100000)

env.save("vecnormalize_stats.pkl")

eval_env = DummyVecEnv([make_env])
eval_env = VecNormalize.load("vecnormalize_stats.pkl", eval_env)

eval_env.training = False
eval_env.norm_reward = False


# Evaluate the trained model
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward}, Std reward: {std_reward}")


# Test the trained model with rendering
test_env = DummyVecEnv([make_env])
test_env = VecNormalize.load("vecnormalize_stats.pkl", test_env)
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



    