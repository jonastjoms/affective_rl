#!/usr/bin/env python
# SAC stuff:
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import with_statement
import argparse
import numpy as np
import os
import sys
import scipy
import copy
from math import pi, sqrt
from collections import deque
import random
import torch
import time
from torch import optim
from tqdm import tqdm
from hyperparams import OFF_POLICY_BATCH_SIZE as BATCH_SIZE, DISCOUNT, ENTROPY_WEIGHT, HIDDEN_SIZE, LEARNING_RATE, MAX_STEPS, POLYAK_FACTOR, REPLAY_SIZE, UPDATE_INTERVAL, UPDATE_START, SAVE_INTERVAL
from sac_models import Critic, SoftActor, create_target_network, update_target_network
from decimal import Decimal

# Other
import sys, gym, time
from lunar_lander_bot import LunarLander
import pickle
import numpy as np
from stable_baselines import DQN

class RL:

    def __init__(self):
        self.disturb = False
        self.state = np.zeros((1, 15))
        self.device = torch.device('cpu')
        self.timer = 0
        self.use_human_action = True
        self.per_episode_reward = 0

    def get_state(self, obs, probs):
        self.state[0:7] = obs
        self.state[7:15] = probs

def key_press(key, mod):
    global human_agent_action, human_wants_restart, human_sets_pause
    if key==0xff0d: human_wants_restart = True
    if key==32: human_sets_pause = not human_sets_pause
    a = int( key - ord('0') )
    if a <= 0 or a >= ACTIONS: return
    human_agent_action = a

def key_release(key, mod):
    global human_agent_action
    a = int( key - ord('0') )
    if a <= 0 or a >= ACTIONS: return
    if human_agent_action == a:
        human_agent_action = 0

def change_mode():
    if np.random.randint(0, 2) == 0:
        disturb = True
    else:
        disturb = False
    return disturb

def rollout(env, rl_agent, actor, critic_1, critic_2, value_critic, target_value_critic, actor_optimiser, value_critic_optimiser, D, target_entropy, log_alpha, alpha_optimizer):
    global human_agent_action, human_wants_restart, human_sets_pause, im_id
    human_wants_restart = False
    obs = env.reset()
    skip = 0
    total_reward = 0
    total_timesteps = 0
    count = 0
    while 1:
        # Add to in-epidsode counter
        count += 1
        # Add to RL_agent timer

        rl_agent.timer += 1
        if not skip:
            human_action = human_agent_action
            total_timesteps += 1
            skip = SKIP_CONTROL
        else:
            skip -= 1

        # Get the robot action:
        robot_action, _states = model.predict(obs)

        # Execute action in gym
        if rl_agent.use_human_action == True:
            obs, r, done, info = env.step(human_action)
        else:
            obs, r, done, info = env.step(robot_action)

        # Add to reward for RL_agent
        rl_agent.per_episode_reward += r

        # Render environment
        if rl_agent.disturb == True:
            window_still_open = env.render(disturb = True)
        else:
            window_still_open = env.render()

        # Get emotional probabilities
        try:
            dbfile = open('probs.p', 'rb')
            probs = pickle.load(dbfile)
        except:
            pass

        # Set state for RL_agent
        rl_agent.get_state(obs, probs)

        # Every 10 time step execute RL_agent action
        if rl_agent.timer % 10 == 0:
            state = torch.tensor(rl_agent.state).float().to(rl_agent.device)
            # Determine action:
                if step < UPDATE_START:
                    # To improve exploration take actions sampled from a uniform random distribution over actions at the start of training
                    action = torch.tensor([2 * random.random() - 1, 2 * random.random() - 1], device=rl_agent.device).unsqueeze(0)
                else:
                    # Observe state s and select action a ~ mu(a|s)
                    action = actor(state.unsqueeze(0)).sample()

        # Every 50 step, maybe change mode
        if count % 50 == 0:
            rl_agent.disturb = change_mode()
            print(rl_agent.disturb)

        # Check if done etc.
        if window_still_open == False:
            return False
        if done:
            break
        if human_wants_restart:
            break
        while human_sets_pause:
            env.render()
            time.sleep(0.1)
        time.sleep(0.1)

    print("timesteps %i reward %0.2f" % (total_timesteps, total_reward))


if __name__ == '__main__':

    resuming = False
    # Lunar Lander stuff:
    env = LunarLander()
    env.render()
    env.unwrapped.viewer.window.on_key_press = key_press
    env.unwrapped.viewer.window.on_key_release = key_release
    # Load the trained agent
    model = DQN.load("dqn_lunar_hover4")
    if not hasattr(env.action_space, 'n'):
        raise Exception('Keyboard agent only supports discrete action spaces')
    ACTIONS = env.action_space.n
    SKIP_CONTROL = 0    # Use previous control decision SKIP_CONTROL times, that's how you
                        # can test what skip is still usable.
    human_agent_action = 0
    human_wants_restart = False
    human_sets_pause = False
    human_in_control = True

    # SAC stuff:
    rl_agent = RL()
    # SAC initialisations
    action_space = 2
    state_space = 15
    actor = SoftActor(HIDDEN_SIZE).to(rl_agent.device)
    critic_1 = Critic(HIDDEN_SIZE, state_action=True).to(rl_agent.device)
    critic_2 = Critic(HIDDEN_SIZE, state_action=True).to(rl_agent.device)
    value_critic = Critic(HIDDEN_SIZE).to(rl_agent.device)
    if resuming == True:
        print("Loading models")
        checkpoint = torch.load("/home/jonas/catkin_ws/src/exercises/part2/ros/checkpoints/agent.pth")
        actor.load_state_dict(checkpoint['actor_state_dict'])
        critic_1.load_state_dict(checkpoint['critic_1_state_dict'])
        critic_2.load_state_dict(checkpoint['critic_2_state_dict'])
        value_critic.load_state_dict(checkpoint['value_critic_state_dict'])
        UPDATE_START = 0

    target_value_critic = create_target_network(value_critic).to(rl_agent.device)
    actor_optimiser = optim.Adam(actor.parameters(), lr=LEARNING_RATE)
    critics_optimiser = optim.Adam(list(critic_1.parameters()) + list(critic_2.parameters()), lr=LEARNING_RATE)
    value_critic_optimiser = optim.Adam(value_critic.parameters(), lr=LEARNING_RATE)
    D = deque(maxlen=REPLAY_SIZE)
    # Automatic entropy tuning init
    target_entropy = -np.prod(action_space).item()
    log_alpha = torch.zeros(1, requires_grad=True, device=rl_agent.device)
    alpha_optimizer = optim.Adam([log_alpha], lr=LEARNING_RATE)

    # Load models
    if resuming == True:
        target_value_critic.load_state_dict(checkpoint['target_value_critic_state_dict'])
        actor_optimiser.load_state_dict(checkpoint['actor_optimiser_state_dict'])
        critics_optimiser.load_state_dict(checkpoint['critics_optimiser_state_dict'])
        value_critic_optimiser.load_state_dict(checkpoint['value_critic_optimiser_state_dict'])
        alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
        D = pickle.load( open("/home/jonas/catkin_ws/src/exercises/part2/ros/checkpoints/deque.p", "rb" ) )

    pbar = tqdm(range(1, MAX_STEPS + 1), unit_scale=1, smoothing=0)

    for step in pbar:
        window_still_open = rollout(env, rl_agent, actor, critic_1, critic_2, value_critic, target_value_critic, actor_optimiser, value_critic_optimiser, D, target_entropy, log_alpha, alpha_optimizer)
        if window_still_open==False: break
