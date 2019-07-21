import numpy as np
import random

import params
import utils
from model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

from utils import DEVICE, Noise, ReplayBuffer


class TDAgent:

    def __init__(self, brain_name, input_size, output_size, agent_count):

        self.best_avg_score = -1
        self.agent_count = agent_count
        print("Instantiating agent for: {}".format(brain_name))
        self.step_count = 0
        self.brain_name = brain_name
        self.state_size = input_size
        self.action_size = output_size
        self.buffer_size = params.BUFFER_SIZE
        self.batch_size = params.BATCH_SIZE
        self.lr_actor = params.LR_ACTOR
        self.lr_critic = params.LR_CRITIC
        self.weight_decay = params.WEIGHT_DECAY

        # Neural Nets
        self.actor_local = Actor(input_size, output_size).to(DEVICE)
        self.critic_local = Critic(input_size, output_size).to(DEVICE)
        self.critic_local2 = Critic(input_size, output_size).to(DEVICE)
        self.actor_target = Actor(input_size, output_size).to(DEVICE)
        self.critic_target = Critic(input_size, output_size).to(DEVICE)
        self.critic_target2 = Critic(input_size, output_size).to(DEVICE)

        # Copy weights
        utils.copy_weights(copy_from=self.actor_target, copy_to=self.actor_local)
        utils.copy_weights(copy_from=self.critic_target, copy_to=self.critic_local)
        utils.copy_weights(copy_from=self.critic_target2, copy_to=self.critic_local2)

        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=self.lr_actor,
                                          weight_decay=self.weight_decay)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=self.lr_critic,
                                           weight_decay=self.weight_decay)
        self.critic_optimizer2 = optim.Adam(self.critic_local2.parameters(), lr=self.lr_critic,
                                           weight_decay=self.weight_decay)

        # Noise
        if params.ADD_OU_NOISE and params.TRAINING_MODE:
            self.OUNoise = Noise(size=output_size)
        else:
            self.OUNoise = None

        # Replay memory
        self.memory = ReplayBuffer()

    def store_transitions(self, states, actions, rewards, next_states, dones):
        # Save experience / reward
        for row in range(self.agent_count):
            self.memory.add(states[row], actions[row], rewards[row], next_states[row], dones[row])

    def step(self):
        self.step_count += 1
        if len(self.memory) >= self.batch_size:
            exp = self.memory.sample()
            self.learn(exp)

    def get_actions(self, state):
        state = torch.from_numpy(state).float().to(DEVICE)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if self.OUNoise:
            action += self.OUNoise.sample()
        return np.clip(action, -1, 1)

    def reset_noise(self):
        if self.OUNoise:
            self.OUNoise.reset()

    def learn(self, exp):
        # Following https://spinningup.openai.com/en/latest/algorithms/ddpg.html
        gamma = params.GAMMA
        states, actions, rewards, next_states, dones = exp

        self.actor_target.eval()
        self.critic_target.eval()
        self.critic_target2.eval()
        with torch.no_grad():
            actions_next = self.actor_target(next_states)
            q_1 = self.critic_target(next_states, actions_next)
            q_2 = self.critic_target2(next_states, actions_next)
            q_n = torch.min(q_1,q_2)
            
        self.actor_target.train()
        self.critic_target.train()
        self.critic_target2.train()

        q_expected = rewards + gamma * q_n * (1 - dones)
        q_predicted = self.critic_local(states, actions)
        q_predicted2 = self.critic_local2(states, actions)
        critic_loss = F.mse_loss(q_predicted, q_expected)
        critic_loss2 = F.mse_loss(q_predicted2, q_expected)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        self.critic_optimizer2.zero_grad()
        critic_loss2.backward()
        self.critic_optimizer2.step()

        if self.step_count % params.POLICY_DELAY == 0:
            act_p = self.actor_local(states)
            actor_loss = -self.critic_local(states, act_p).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            utils.soft_update(self.critic_local, self.critic_target)
            utils.soft_update(self.critic_local2, self.critic_target2)
            utils.soft_update(self.actor_local, self.actor_target)

    def load_model(self, PATH):
        print('Loading saved model...')
        self.actor_local.load_state_dict(torch.load(PATH))
        self.OUNoise = None
        self.actor_local.eval()
        print('Model loaded successfully!')
