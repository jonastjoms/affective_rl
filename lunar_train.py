import gym

from stable_baselines import DQN, A2C, SAC
#from stable_baselines.common.policies import MlpPolicy
from stable_baselines.sac.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv


from lunar_lander_bot import LunarLander
env = LunarLander()
env = DummyVecEnv([lambda: env])

print("Starting traning")
# Instantiate the agent
#model = A2C(MlpPolicy, env, ent_coef=0.1, verbose=1)
#model = DQN('MlpPolicy', env, learning_rate=1e-2, prioritized_replay=True, verbose=1)
#model = SAC(MlpPolicy, env, verbose=1)
# Load trained agent
model = DQN.load("sac_lunar")
model.set_env(env)
# Train the agent
for batch in range(20):
    model.learn(total_timesteps = int(10000))
    print("Batch: ", batch)
# Save the agent
#model.save("a2c_lunar")
#model.save("dqn_lunar")
model.save("sac_lunar")

print("Finished training")
