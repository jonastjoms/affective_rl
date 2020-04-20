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
        self.state = np.zeros((1, 7))[0]
        self.next_state = np.zeros((1, 7))[0]
        self.device = torch.device('cpu')
        self.timer = 0
        self.use_human_action = True
        self.per_step_reward = 0
        self.per_step_probs = np.zeros((1,7))
        self.first_iteration = True
        self.action = torch.empty(1)
        self.first_update = True
        self.lunar_episode_counter = 0
        self.lunar_per_episode_step_counter = 0

    def get_state(self, probs):
        state = np.zeros((1, 7))[0]
        state[:] = probs
        return state

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

def rollout(env, rl_agent):
    global human_agent_action, human_wants_restart, human_sets_pause, steps
    global actor, critic_1, critic_2, value_critic, target_value_critic, actor_optimiser, value_critic_optimiser, D, target_entropy, log_alpha, alpha_optimizer
    human_wants_restart = False
    obs = env.reset()
    skip = 0
    total_reward = 0
    total_timesteps = 0
    while 1:
        # Add to RL_agent timer
        rl_agent.timer += 1
        rl_agent.lunar_per_episode_step_counter += 1

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
        rl_agent.per_step_reward += r

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

        rl_agent.per_step_probs += probs

        # Every 10 time step execute RL_agent
        if rl_agent.timer % 50 == 0:
            print("Last episode's reward: ", rl_agent.per_step_reward)
            if not rl_agent.first_iteration:
                # Get next state for RL_agent
                rl_agent.next_state = rl_agent.get_state(probs*50)
                # Get new action and execute:
                rl_agent.state = rl_agent.next_state
                rl_agent.action = actor(torch.tensor(rl_agent.state).float().to(rl_agent.device).unsqueeze(0)).mean
                if rl_agent.action.detach().numpy()[0] > 0:
                    rl_agent.use_human_action = True
                else:
                    rl_agent.use_human_action = False
            else: # First iteration, no next_state
                # Get state for RL_agent
                rl_agent.state = rl_agent.get_state(probs*50)
                # Get action and execute:
                rl_agent.action = actor(torch.tensor(rl_agent.state).float().to(rl_agent.device).unsqueeze(0)).mean

                if rl_agent.action.detach().numpy()[0] > 0:
                    rl_agent.use_human_action = True
                else:
                    rl_agent.use_human_action = False
                rl_agent.first_iteration = False
            # Reset reward and probs
            rl_agent.per_step_reward = 0
            rl_agent.per_step_probs = np.zeros((1,7))

            if rl_agent.use_human_action == True:
                print("Human action")
            else:
                print("Robot action")
            print("Disturb: ", rl_agent.disturb)

        # Every 50 step, maybe change mode
        if rl_agent.timer % 200 == 0:
            rl_agent.disturb = change_mode()

        # Check if done etc.
        if window_still_open == False:
            return False
        if rl_agent.lunar_per_episode_step_counter > 500:
            done = True
        if done:
            break
        if human_wants_restart:
            break
        while human_sets_pause:
            env.render()
            time.sleep(0.1)
        time.sleep(0.1)

    #print("timesteps %i reward %0.2f" % (total_timesteps, total_reward))


if __name__ == '__main__':

    resuming = True
    # Lunar Lander stuff:
    env = LunarLander()
    env.render()
    env.unwrapped.viewer.window.on_key_press = key_press
    env.unwrapped.viewer.window.on_key_release = key_release
    # Load the trained agent
    model = DQN.load("dqn_lunar_hover_latest")
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
    action_space = 1
    state_space = 15
    actor = SoftActor(HIDDEN_SIZE).to(rl_agent.device)
    critic_1 = Critic(HIDDEN_SIZE, state_action=True).to(rl_agent.device)
    critic_2 = Critic(HIDDEN_SIZE, state_action=True).to(rl_agent.device)
    value_critic = Critic(HIDDEN_SIZE).to(rl_agent.device)
    if resuming == True:
        print("Loading models")
        checkpoint = torch.load("checkpoints/agent_2.pth")
        actor.load_state_dict(checkpoint['actor_state_dict'])
        critic_1.load_state_dict(checkpoint['critic_1_state_dict'])
        critic_2.load_state_dict(checkpoint['critic_2_state_dict'])
        value_critic.load_state_dict(checkpoint['value_critic_state_dict'])

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
        D = pickle.load( open("checkpoints/deque_2.p", "rb" ) )
        print("Models loaded")
        time.sleep(5)

    for step in range(50):
        rl_agent.lunar_episode_counter += 1
        rl_agent.lunar_per_episode_step_counter = 0
        window_still_open = rollout(env, rl_agent)
        if window_still_open == False:
            break
