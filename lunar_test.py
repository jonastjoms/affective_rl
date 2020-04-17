import gym

from stable_baselines import DQN, SAC
from stable_baselines.common.evaluation import evaluate_policy

FPS    = 50
SCALE  = 30.0   # affects how fast-paced the game is, forces should be adjusted as well

MAIN_ENGINE_POWER  = 13.0
SIDE_ENGINE_POWER  =  0.6

INITIAL_RANDOM = 1000.0   # Set 1500 to make game harder

LANDER_POLY =[
    (-14,+17), (-17,0), (-17,-10),
    (+17,-10), (+17,0), (+14,+17)
    ]
LEG_AWAY = 20
LEG_DOWN = 18
LEG_W, LEG_H = 2, 8
LEG_SPRING_TORQUE = 40

SIDE_ENGINE_HEIGHT = 14.0
SIDE_ENGINE_AWAY   = 12.0

VIEWPORT_W = 600
VIEWPORT_H = 400

from lunar_lander_bot import LunarLander
env = LunarLander()

# Load the trained agent
model = DQN.load("dqn_lunar_hover3")

# Evaluate the agent
#mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)

# Enjoy trained agent
obs = env.reset()
while True:
    pos = env.lander.position
    print((pos.x - VIEWPORT_W/SCALE/2) / (VIEWPORT_W/SCALE/2), (pos.y - (env.helipad_y+LEG_DOWN/SCALE)) / (VIEWPORT_H/SCALE/2))
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
