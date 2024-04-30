#This code is adapted from a YouTube Tutorial by brthor. (Source: https://www.youtube.com/watch?v=tsy1mgB7hB0&t=0s retrieved in April 2024.)

import os
import random
import numpy as np
import torch
from torch import nn
import itertools
import time
import gym

# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPS_START = 1.0
EPS_END = 0.1
EPS_DECAY = int(1e6)

class TransposeImageObs(gym.ObservationWrapper):
    def __init__(self, env, op):
        super().__init__(env)
        assert len(op) == 3, "Op must have 3 dimensions"

        self.op = op

        obs_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            [
                obs_shape[self.op[0]],
                obs_shape[self.op[1]],
                obs_shape[self.op[2]]
            ],
            dtype=self.observation_space.dtype)

    def observation(self, obs):
        return obs.transpose(self.op[0], self.op[1], self.op[2])

#cnn architecture outline in the paper (which was published in the journal nature)
def nature_cnn(observation_space, depths=(32, 64, 64), final_layer=512):
  n_input_channels = observation_space.shape[0]

  cnn = nn.Sequential(
  nn.Conv2d(n_input_channels, depths[0], kernel_size=8, stride=4),
  nn.ReLU(),
  nn.Conv2d(depths[0], depths[1], kernel_size=4, stride=2),
  nn.ReLU(),
  nn.Conv2d(depths[1], depths[2], kernel_size=3, stride=1),
  nn.ReLU(),
  nn.Flatten()
  )

  #compute shape by doing one forward pass
  with torch.no_grad():
    n_flatten = cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]
  out = nn.Sequential(cnn, nn.Linear(n_flatten, final_layer), nn.ReLU())

  return out


class Dueling_Network(nn.Module):
  def __init__(self, env, device):
    super().__init__()

    self.num_actions = env.action_space.n
    self.device = device

    conv_net = nature_cnn(env.observation_space)

    # Shared feature extractor
    self.feature_extractor = conv_net

    # Dueling network branches
    self.value_stream = nn.Sequential(
        nn.Linear(512, 256),  # Adjust this size as needed
        nn.ReLU(),
        nn.Linear(256, 1)
    )

    self.advantage_stream = nn.Sequential(
        nn.Linear(512, 256),  # Adjust this size as needed
        nn.ReLU(),
        nn.Linear(256, self.num_actions)
    )

  #The forward function is required to run any PyTorch network
  def forward(self, x):
        # Forward pass through the shared feature extractor
        features = self.feature_extractor(x)

        # Compute value and advantage streams
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)

        # Combine value and advantage to get Q-values
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))

        return q_values

  def act(self, obs):
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
    q_values = self(obs_t.unsqueeze(0)) #unsqueeze 0 to create a fake batch dimension because pytorch operations expect a batched dimension, we are not using a batched env
    max_q_index = torch.argmax(q_values, dim=1)[0]
    action = max_q_index.detach().item() #turn pytorch tensor into an integer using detach method

    return action

env = gym.make("ALE/Breakout-v5", render_mode="human")
env = TransposeImageObs(env, op=[2, 0, 1])  # Convert to torch order (C, H, W)
# env.reset()

# for i in range(1000):
#     env.step(env.action_space.sample())
#     env.render()

net = Dueling_Network(env, device)
net.load_state_dict(torch.load("/Users/eimss/Downloads/OneDrive_1_4-30-2024/dueling_trained_500", map_location = device))

obs = env.reset()
beginning_episode = True
for step in itertools.count():
    epsilon = np.interp(step, [0, EPS_DECAY], [EPS_START, EPS_END])

    sample = random.random()

    if sample <= epsilon:
        action = env.action_space.sample()
    else:
        action = net.act(obs)

    if beginning_episode:
        action = 1
        beginning_episode = False

    obs, reward, done, _, info = env.step(action)
    env.render()
    time.sleep(0.02)

    if done:
        obs = env.reset()
        beginning_episode = True