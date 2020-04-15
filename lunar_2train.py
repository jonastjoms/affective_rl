import gym

from stable_baselines import DQN, A2C, SAC
#from stable_baselines.common.policies import MlpPolicy
from stable_baselines.sac.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv


from lunar_lander_human import LunarLander
env = LunarLander()
env = DummyVecEnv([lambda: env])

print("Starting traning")
# Instantiate the agent
model = SAC(MlpPolicy, env, verbose=1)
del model
model = SAC.load("sac2_lunar")
model.set_env(env)
# Train the agent
model.learn(total_timesteps=250000, log_interval=10)
# Save the agent
model.save("sac2_lunar")

print("Finished training")
