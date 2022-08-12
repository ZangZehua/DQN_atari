import math
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from networks.dqn_network import ImageNetwork
from common.memory_buffer import MemoryBuffer


class DQN:
    def __init__(self, device, in_channels, action_space, memory_size,
                 epsilon_max=1, epsilon_min=0.05, epsilon_decay=30000, lr=1e-4, gamma=0.99):
        self.device = device
        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.action_space = action_space
        self.memory = MemoryBuffer(memory_size=memory_size)
        network = ImageNetwork
        self.policy_net = network(in_channels, self.action_space.n).to(self.device)
        self.target_net = network(in_channels, self.action_space.n).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.RMSprop(self.policy_net.parameters(), lr=lr, eps=0.001, alpha=0.95)
        # self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.gamma = gamma

    def epsilon_by_step(self, step):
        epsilon = self.epsilon_min + (self.epsilon_max - self.epsilon_min) * math.exp(-1. * step / self.epsilon_decay)
        return epsilon

    def get_state(self, lazy_frame):
        state = torch.from_numpy(lazy_frame._force().transpose(2, 0, 1)[None] / 255).float()
        return state

    def select_action(self, state, epsilon=None):
        if epsilon is None:
            epsilon = self.epsilon_max
        if random.random() < epsilon:
            action = self.action_space.sample()
        else:
            action = self.policy_net(state.to(self.device)).cpu().detach().numpy().argmax(1)[0]
        return action

    def learn(self, batch_size):
        # sample from memory
        states, actions, rewards, next_states, dones = self.sample_memory(batch_size)

        # learn once
        td_loss = self.compute_td_loss(states, actions, rewards, next_states, dones, gamma=self.gamma)
        self.optimizer.zero_grad()
        td_loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        return td_loss

    def sample_memory(self, batch_size):
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for i in range(batch_size):
            idx = random.randint(0, self.memory.size() - 1)
            frame, action, reward, next_frame, done = self.memory.buffer[idx]
            states.append(self.get_state(frame))
            actions.append(action)
            rewards.append(reward)
            next_states.append(self.get_state(next_frame))
            dones.append(done)
        return torch.cat(states).to(self.device), torch.LongTensor(actions).to(self.device), torch.Tensor(rewards).to(self.device), torch.cat(next_states).to(self.device), torch.BoolTensor(dones).to(self.device)

    def compute_td_loss(self, states, actions, rewards, next_states, dones, gamma=0.99):
        # get q-values for all actions in current states
        predicted_q = self.policy_net(states)
        # select q-values for chosen actions
        predicted_q_actions = predicted_q[range(states.shape[0]), actions]
        # compute q-values for all actions in next states
        predicted_next_q = self.target_net(next_states)
        # compute V*(next_states) using predicted next q-values
        next_state_values = predicted_next_q.max(-1)[0]
        # compute "target q-values" for loss - it's what's inside square parentheses in the above formula.
        target_q_actions = rewards + gamma * next_state_values
        # at the last state we shall use simplified formula: Q(s,a) = r(s,a) since s' doesn't exist
        target_q_actions = torch.where(dones, rewards, target_q_actions)
        # mean squared error loss to minimize

        loss = nn.SmoothL1Loss(predicted_q_actions, target_q_actions.detach())
        return loss


