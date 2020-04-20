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
import os
import glob
import json
import cv2
import numpy as np
import torch
from torchvision.transforms import transforms
from ResidualMaskingNetwork.models import densenet121, resmasking_dropout1
import pickle

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


if __name__ == '__main__':

    resuming = True
    vid = cv2.VideoCapture(0)

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
        checkpoint = torch.load("checkpoints/agent_heavy.pth")
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
        D = pickle.load( open("checkpoints/deque_heavy.p", "rb" ) )
        print("Models loaded")
        time.sleep(0.1)

    for step in range(100):
        ret, frame = vid.read()
        # Get emotional probabilities
        try:
            dbfile = open('probs.p', 'rb')
            probs = pickle.load(dbfile)
        except:
            pass
        # Get next state for RL_agent
        rl_agent.next_state = rl_agent.get_state(probs*50)
        # Get new action and execute:
        rl_agent.state = rl_agent.next_state
        rl_agent.action = actor(torch.tensor(rl_agent.state).float().to(rl_agent.device).unsqueeze(0)).mean
        if rl_agent.action.detach().numpy()[0] > 0:
           caption = "Use human action"
        else:
           caption = "Use robot action"
        font = cv2.FONT_HERSHEY_SIMPLEX
        # org
        org = (400, 100)
        # fontScale
        fontScale = 2
        # Blue color in BGR
        color = (0, 0, 0)
        # Line thickness of 2 px
        thickness = 4
        # Using cv2.putText() method
        cv2.putText(frame, caption, org, font, fontScale, color, thickness, cv2.LINE_AA)
        time.sleep(0.2)
        # Displaying the image
        cv2.imshow('Image', frame)
        # Save img
        filename = 'image'+str(step)+'.png'
        cv2.imwrite('images/'+filename, frame)
        if cv2.waitKey(1) == ord('q'):
            break
