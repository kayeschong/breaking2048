from games import Env2048

import os
import time
import random
import gym
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple, deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class DQCNN(nn.Module):

    def __init__(self, height, width, n_actions):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=2, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)

        self.fc = nn.Linear(64 * (height-1) * (width-1), out_features=128)
        self.out = nn.Linear(128, n_actions)


    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        x = self.fc(x.view(x.size(0), -1))
        x = self.out(x) # flatten
        return x

Experience = namedtuple(
    'Experience',
    ('state', 'action', 'next_state', 'reward', 'done')
)

class ReplayMemory():
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.push_count = 0
    
    def append(self, experience):
        if len(self.memory) < self.capacity:
            self.memory.append(experience)
        else:
            self.memory[self.push_count % self.capacity] = experience
        self.push_count += 1

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# TODO: Agent interface to interact with GUI
class DQNAgent():
    def __init__(self, model_path="dqcnn.pt", device=None, train=False):
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.model_path=model_path

        self.model = DQCNN(4, 4, 4).to(self.device)
        if os.path.exists(self.model_path):        
            self.model.load_state_dict(torch.load(self.model_path))
            self.model.eval()
        else:
            if train:
                self.train()
            else:
                print("Please train model first or specify weights path")
    
    def train(self):
        # Hyperparams
        BATCH_SIZE = 4096
        MEMORY_SIZE = 100000
        GAMMA = 0.99
        EPS = 0.2
        TARGET_UPDATE = 20
        MA_WINDOW = 100
        LEARNING_RATE = 0.1

        device = self.device

        env = Env2048()

        rewards_window = deque(maxlen=MA_WINDOW)
        memory = ReplayMemory(MEMORY_SIZE)
        loss_fn = nn.MSELoss() # f(inp, target)
        policy_net = DQCNN(env.height, env.width, env.action_space.n).to(device)
        target_net = DQCNN(env.height, env.width, env.action_space.n).to(device)
        target_net.load_state_dict(policy_net.state_dict())
        target_net.eval()
        optimizer = optim.Adam(params=policy_net.parameters(), lr=LEARNING_RATE)
        
        NUM_EPISODES = 500
        possible_values = [0] + [2**i for i in range(1, 13)]

        for episode in range(NUM_EPISODES):
            # print("episode:", episode)
            env.reset()
            done = False
            obs = env.grid.copy()
            # Convert tiles to range 0-11
            obs[obs==0] = 1
            obs = np.log2(obs)
            # # Convert tiles to binary
            # obs = np.stack([env.grid == val for val in possible_values]).astype(int)
            
            curr_state = torch.Tensor(obs).unsqueeze(0).to(device)
            
            rng = np.random.default_rng(seed=None)
            episode_reward = torch.tensor([0])
            while not done:
                with torch.no_grad():
                    if EPS > rng.random():
                        out = torch.rand((1, env.action_space.n)) # explore
                    else:
                        out = policy_net(curr_state) # exploit
                    # # Filter to valid actions
                    out[:, env.get_invalid_moves()] = 0

                    action = out.argmax(dim=1).to(device)
                    obs, reward, done, info = env.step(action.item())
                    # Convert tiles to range 0-11
                    obs[obs==0] = 1
                    obs = np.log2(obs)
                    # # Convert tiles to binary
                    # obs = np.stack([obs == val for val in possible_values]).astype(int)
                    next_state = torch.Tensor(obs).unsqueeze(0).to(device)
                    episode_reward += reward
                    reward = torch.tensor(reward).unsqueeze(0).to(device)
                    memory.append(Experience(curr_state, action, next_state, reward, torch.tensor(done).unsqueeze(0).to(device)))
                    curr_state = next_state
                
                # Start training once min samples reach
                if len(memory) >= BATCH_SIZE:
                    experience_batch = memory.sample(BATCH_SIZE)
                    batch_states, batch_actions, batch_next_states, batch_rewards, batch_done = self.unpack_batch(experience_batch)
                    
                    current_Q = policy_net(batch_states).gather(dim=1, index=batch_actions.unsqueeze(-1))
                    next_Q, max_indexes = target_net(batch_next_states).max(dim=1)
                    # If lose game, we know all future action-values are 0
                    next_Q[batch_done] = 0
                    # Expected current Q based on bellman eq
                    target_Q = batch_rewards + (next_Q * GAMMA)
                    
                    loss = loss_fn(current_Q, target_Q.unsqueeze(1))
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            if episode % TARGET_UPDATE == 0:
                target_net.load_state_dict(policy_net.state_dict())

        torch.save(policy_net.state_dict(), self.model_path)
        self.model.load_state_dict(torch.load(self.model_path))
        self.model.eval()

    def unpack_batch(self, experiences):
        # Convert batch of Experiences to Experience of batches
        batch = Experience(*zip(*experiences))

        batch_states = torch.cat(batch.state)
        batch_actions = torch.cat(batch.action)
        batch_next_states = torch.cat(batch.next_state)
        batch_rewards = torch.cat(batch.reward)
        batch_done = torch.cat(batch.done)

        return batch_states, batch_actions, batch_next_states, batch_rewards, batch_done
    
    def get_move(self, board) -> int:
        obs = board.grid.copy()
        obs[obs==0] = 1
        obs = np.log2(obs)
        # obs = np.stack([env.grid == val for val in possible_values]).astype(int)
        curr_state = torch.Tensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            # DQN pred action-value
            out = self.model(curr_state)
            # Filter valid actions, set to low q-value, take best valid move
            out[:, board.get_invalid_moves()] = -1e6
            # DQN best action-value from valid actions    
            action = out.argmax(dim=1).item()
        
        return action

if __name__ == "__main__":
    env = Env2048()
    model = DQNAgent(train=True)
    print(model.get_move(env))