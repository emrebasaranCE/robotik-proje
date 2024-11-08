import argparse
import pickle
from collections import namedtuple
from itertools import count

import os
import time
import numpy as np
import matplotlib.pyplot as plt

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from tensorboardX import SummaryWriter


# Parameters
env_name = 'MountainCar-v0'
gamma = 0.99
render = False
seed = 1
log_interval = 10

# Initialize environment and define state/action sizes
env = gym.make(env_name).unwrapped
env.reset(seed=seed)  # Set the seed when resetting the environment
torch.manual_seed(seed)

# Define the number of states and actions based on the environment
num_state = env.observation_space.shape[0]
num_action = env.action_space.n

Transition = namedtuple('Transition', ['state', 'action', 'a_log_prob', 'reward', 'next_state'])

class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(num_state, 128)
        self.action_head = nn.Linear(128, num_action)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        action_prob = F.softmax(self.action_head(x), dim=1)
        return action_prob


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(num_state, 128)
        self.state_value = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        value = self.state_value(x)
        return value


class PPO:
    clip_param = 0.2
    max_grad_norm = 0.5
    ppo_update_time = 1
    buffer_capacity = 1000
    batch_size = 16

    def __init__(self):
        super(PPO, self).__init__()
        self.actor_net = Actor().to(device)  # Move to GPU
        self.critic_net = Critic().to(device)  # Move to GPU
        self.buffer = []
        self.counter = 0
        self.training_step = 0
        self.writer = SummaryWriter('./exp')

        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), 1e-3)
        self.critic_net_optimizer = optim.Adam(self.critic_net.parameters(), 3e-3)
        if not os.path.exists('./param'):
            os.makedirs('./param/net_param')
            os.makedirs('./param/img')

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)  # Move state to GPU
        with torch.no_grad():
            action_prob = self.actor_net(state)
        c = Categorical(action_prob)
        action = c.sample()
        return action.item(), action_prob[:, action.item()].item()

    def get_value(self, state):
        state = torch.from_numpy(state).to(device)  # Move state to GPU
        with torch.no_grad():
            value = self.critic_net(state)
        return value.item()

    def save_param(self):
        torch.save(self.actor_net.state_dict(), f"./param/net_param/actor_net_{int(time.time())}.pkl")
        torch.save(self.critic_net.state_dict(), f"./param/net_param/critic_net_{int(time.time())}.pkl")

    def store_transition(self, transition):
        self.buffer.append(transition)
        self.counter += 1

    def update(self, i_ep):
        state = torch.tensor(np.array([t.state for t in self.buffer]), dtype=torch.float).to(device)  # Move state to GPU
        action = torch.tensor([t.action for t in self.buffer], dtype=torch.long).view(-1, 1).to(device)  # Move to GPU
        reward = [t.reward for t in self.buffer]
        old_action_log_prob = torch.tensor([t.a_log_prob for t in self.buffer], dtype=torch.float).view(-1, 1).to(device)  # Move to GPU

        R = 0
        Gt = []
        for r in reward[::-1]:
            R = r + gamma * R
            Gt.insert(0, R)
        Gt = torch.tensor(Gt, dtype=torch.float).to(device)  # Move to GPU

        for _ in range(self.ppo_update_time):
            for index in BatchSampler(SubsetRandomSampler(range(len(self.buffer))), self.batch_size, False):
                Gt_index = Gt[index].view(-1, 1)
                V = self.critic_net(state[index])
                delta = Gt_index - V
                advantage = delta.detach()

                action_prob = self.actor_net(state[index]).gather(1, action[index])  # new policy

                ratio = action_prob / old_action_log_prob[index]
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantage

                # update actor network
                action_loss = -torch.min(surr1, surr2).mean()  # maximize policy
                self.writer.add_scalar('loss/action_loss', action_loss, global_step=self.training_step)
                self.actor_optimizer.zero_grad()
                action_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                # update critic network
                value_loss = F.mse_loss(Gt_index, V)
                self.writer.add_scalar('loss/value_loss', value_loss, global_step=self.training_step)
                self.critic_net_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.max_grad_norm)
                self.critic_net_optimizer.step()
                self.training_step += 1

        del self.buffer[:]  # clear experience


def main():
    agent = PPO()
    for i_epoch in range(1000):
        print(f"Iteration: {i_epoch}")
        state = env.reset() if isinstance(env.reset(), np.ndarray) else env.reset()[0]
        if render:
            env.render()

        for t in count():
            action, action_prob = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)[:4]
            trans = Transition(state, action, action_prob, reward, next_state)
            if render:
                env.render()
            agent.store_transition(trans)
            state = next_state

            if done:
                if len(agent.buffer) >= agent.batch_size:
                    agent.update(i_epoch)
                agent.writer.add_scalar('Steptime/steptime', t, global_step=i_epoch)
                break


if __name__ == '__main__':
    print("Code started.")
    main()
    print("end")
