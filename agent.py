import copy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from model import DDQN
from replay_buffer import ReplayBuffer

class DDQNAgent:
    def __init__(
        self,
        num_actions=6,
        num_frames=8,
        gamma=0.99,
        lr=5e-4,
        batch_size=32,
        buffer_size=50_000,
        min_replay_size=1_000,
        epsilon_start=1.0,
        epsilon_min=0.0001,
        episodes=500,
        buffer_dir="replay_buffer",
    ):
        self.num_actions = num_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.min_replay_size = min_replay_size
        self.episode_count = 0

        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = (epsilon_min / epsilon_start) ** (1.0 / episodes)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {self.device}")

        self.online_net = DDQN(num_frames=num_frames, num_actions=num_actions).to(self.device)
        self.target_net = copy.deepcopy(self.online_net)
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss
        self.replay_buffer = ReplayBuffer(buffer_size, data_dir=buffer_dir)

    # Interaction
    def select_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.num_actions)
        self.online_net.eval()
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.online_net(state_t)
        self.online_net.train()
        return int(q_values.argmax(dim=1).item())

    def store(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

    # Learning
    def train_step(self):
        if len(self.replay_buffer) < self.min_replay_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states_t      = torch.tensor(states).to(self.device)
        actions_t     = torch.tensor(actions).to(self.device)
        rewards_t     = torch.tensor(rewards).to(self.device)
        next_states_t = torch.tensor(next_states).to(self.device)
        dones_t       = torch.tensor(dones).to(self.device)

        # Current Q-values for the actions that were taken
        q_values = self.online_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # DDQN target: online net picks next action, target net scores it
        with torch.no_grad():
            next_actions = self.online_net(next_states_t).argmax(dim=1)
            next_q = self.target_net(next_states_t).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            targets = rewards_t + self.gamma * next_q * (1.0 - dones_t)

        loss = self.loss_fn(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        return loss.detach()

    def end_episode(self):
        if len(self.replay_buffer) >= self.min_replay_size:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.episode_count += 1
        # Hard update every 4 episodes 
        if self.episode_count % 4 == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

    # Checkpointing
    def save(self, path):
        torch.save({
            "online_net":    self.online_net.state_dict(),
            "target_net":    self.target_net.state_dict(),
            "optimizer":     self.optimizer.state_dict(),
            "epsilon":       self.epsilon,
            "episode_count": self.episode_count,
        }, path)
        self.replay_buffer.save()

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(ckpt["online_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.epsilon       = ckpt["epsilon"]
        self.episode_count = ckpt["episode_count"]


class DQNAgent(DDQNAgent):
    """Standard DQN: target network selects AND evaluates the next action."""

    def train_step(self):
        if len(self.replay_buffer) < self.min_replay_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states_t      = torch.tensor(states).to(self.device)
        actions_t     = torch.tensor(actions).to(self.device)
        rewards_t     = torch.tensor(rewards).to(self.device)
        next_states_t = torch.tensor(next_states).to(self.device)
        dones_t       = torch.tensor(dones).to(self.device)

        q_values = self.online_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q  = self.target_net(next_states_t).max(dim=1)[0]
            targets = rewards_t + self.gamma * next_q * (1.0 - dones_t)

        loss = self.loss_fn(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        return loss.detach()
