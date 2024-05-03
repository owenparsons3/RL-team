# This code is adapted from a YouTube Tutorial by brthor.
# (Source: https://www.youtube.com/watch?v=tsy1mgB7hB0&t=0s retrieved in April 2024.)

#you may need to pip install "gymnasium[atari, accept-rom-license]" before running this code

# #Set up
import torch
from torch import nn
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import FrameStack
from gymnasium.wrappers import GrayScaleObservation
from gymnasium.wrappers import ResizeObservation
from collections import deque
import itertools
import random

# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('./logs/ddqn')

# ##Hyperparameters
# GAMMA = TD target discount rate
# 
# BATCH_SIZE = number of samples from replay buffer when computing gradients
# 
# BUFFER = maximum number of transitions to store before updating old transitions
# 
# REPLAY_MIN = minimum number of transition in the replay buffer before starting training
# 
# EPS_START = start value for epsilon
# 
# EPS_END = end value for epsilon EPS_START and EPS_END
# 
# EPS_DECAY = number of steps between
# 
# TARGET_UPDATE_FREQUENCY = frequency for updating the target parameters
# 
# LR = learning rate

#Hyperparameters
GAMMA = 0.99
BATCH_SIZE = 32
BUFFER = int(1e6)
REPLAY_MIN = 50000
EPS_START = 1.0
EPS_END = 0.1
EPS_DECAY = int(1e6)
TARGET_UPDATE_FREQUENCY = 1000
LR = 2.5e-4

# #Environment
# 


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


class Network(nn.Module):
  def __init__(self, env, device):
    super().__init__()

    self.num_actions = env.action_space.n
    self.device = device

    conv_net = nature_cnn(env.observation_space)

    self.net = nn.Sequential(
        conv_net,
        nn.Linear(512, self.num_actions)
    )

  #The forward function is required to run any PyTorch network
  def forward(self, x):
    return self.net(x)

  def act(self, obs):
    obs_t = torch.as_tensor(np.array(obs), dtype=torch.float32, device=self.device)
    q_values = self(obs_t.unsqueeze(0)) #unsqueeze 0 to create a fake batch dimension because pytorch operations expect a batched dimension, we are not using a batched env
    max_q_index = torch.argmax(q_values, dim=1)[0]
    action = max_q_index.detach().item() #turn pytorch tensor into an integer using detach method

    return action
  
  def act(self, obs):
    obs_t = torch.as_tensor(np.array(obs), dtype=torch.float32, device=self.device)
    q_values = self(obs_t.unsqueeze(0)) #unsqueeze 0 to create a fake batch dimension because pytorch operations expect a batched dimension, we are not using a batched env
    max_q_index = torch.argmax(q_values, dim=1)[0]
    action = max_q_index.detach().item() #turn pytorch tensor into an integer using detach method

    return action

# Pre-processing

# Create a replay buffer that has a max length of BUFFER.
# 
# Also create a reward buffer that stores the rewards for an episode to track training performance.

#env = gym.make("CartPole-v1")
#may have to add preprocessing steps for this environment to work
env = gym.make("ALE/Breakout-v5")
env = ResizeObservation(env, 84)
env = GrayScaleObservation(env)
env = FrameStack(env, 4)

replay_buffer = deque(maxlen=BUFFER)
reward_buffer = deque([0.0], maxlen=100)

#there may be better ways to do this using gym and "monitor"
episode_reward = 0.0

# #Create neural networks

online_network = Network(env, device=device)
target_network = Network(env, device=device)

online_network =online_network.to(device)
target_network = target_network.to(device)

#optimise using the adam optimiser
optimiser = torch.optim.Adam(online_network.parameters(), lr=LR)

#set the target network parameters equal to the online network parameters (because they were initialised separately), this is also part of the algorithm in the paper
target_network.load_state_dict(online_network.state_dict())


# ##Initialise replay buffer
# initliase obs
obs, _ = env.reset()

#get the first set of actions and observations before training using replay_min
for _ in range(REPLAY_MIN):

  #take an action
  action = env.action_space.sample()

  #take a step in the environment based on the action and get teh new observations, reward and whether the episode is over
  #store this information in the replay buffer
  #set observation to the new observation
  new_obs, reward, done, _, info = env.step(action)
  transition = (obs, action, reward, done, new_obs)
  replay_buffer.append(transition)
  obs = new_obs

  #reset the environment if the episode is over
  if done:
    obs, _ = env.reset()

# #Training
obs, _ = env.reset()

for step in itertools.count():

  #Epsilon greedy policy to facilitate exploration
  epsilon = np.interp(step, [0, EPS_DECAY], [EPS_START, EPS_END])

  sample = random.random()

  if sample <= epsilon:
    action = env.action_space.sample()
  else:
    action = online_network.act(obs)

  # Interact with the environment every 4th frame
  # if step % 4 == 0:
  #     # Take a step in the environment based on the action and get the new observations, reward, and whether the episode is over
  #     # Store this information in the replay buffer
  #     # Set observation to the new observation and add set reward to episode reward
  #     new_obs, reward, done, _, info = env.step(action)
  #     transition = (obs, action, reward, done, new_obs)
  #     replay_buffer.append(transition)
  #     obs = new_obs
  # else:
  #     # Repeat the last action on the skipped frames
  #     new_obs, reward, done, _, info = env.step(action)

  # Take a step in the environment based on the action and get the new observations, reward, and whether the episode is over
  # Store this information in the replay buffer
  # Set observation to the new observation and add set reward to episode reward
  new_obs, reward, done, _, info = env.step(action)
  transition = (obs, action, reward, done, new_obs)
  replay_buffer.append(transition)
  obs = new_obs
  
  episode_reward += reward

  #reset the environment if the episode is over
  if done:
    obs, _ = env.reset()

    reward_buffer.append(episode_reward)
    episode_reward = 0.0


  #Start gradient steps
  #####################
  #sample BATCH_SIZE number of samples from the replay buffer
  transitions = random.sample(replay_buffer, BATCH_SIZE)

  #separate transition tuple and use it to create lists for the batch and then convert to np (faster for converting to pytorch tensor)
  obses = np.asarray([np.array(t[0]) for t in transitions]) #this line breaks when converting to array for some reason
  obslist = []
# Print shapes of individual frames
#   for t in transitions:
#       if np.array(t[0], dtype = object).shape != (4, 84, 84):
#           obslist.append(np.array(t[0][0]))
#       else:
#           obslist.append(np.array(t[0]))
#   obses = np.array(obslist)
  actions = np.asarray([t[1] for t in transitions])
  rewards = np.asarray([t[2] for t in transitions])
  #rewards = np.clip(rewards, -1, 1) # preprocessing clipping rewards between -1 and 1 (While applying DQN to different environment settings,
                                    # where reward points are not on the same scale, the training becomes inefficient.)
  dones = np.asarray([t[3] for t in transitions])
  new_obses = np.asarray([t[4] for t in transitions])

  obses_t = torch.as_tensor(obses, dtype=torch.float32, device=device)
  actions_t = torch.as_tensor(actions, dtype=torch.int64, device=device).unsqueeze(-1) #unsqeeze -1 adds dimension at the end
  rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=device).unsqueeze(-1)
  dones_t = torch.as_tensor(dones, dtype=torch.float32, device=device).unsqueeze(-1)
  new_obses_t = torch.as_tensor(new_obses, dtype=torch.float32, device=device)


  #Compute Targets
  ################
  online_q_values = online_network(new_obses_t)
  best_online_q_index = online_q_values.argmax(dim=1, keepdim=True)

  target_q_values = target_network(new_obses_t)

  #we want the highest q-value per observation
  #max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]
  target_selected_q_values = torch.gather(input = target_q_values, dim=1, index = best_online_q_index)

  #if the episode is over (dones_t = 1) then we zero out the rest of the function only leaving the reward
  targets = rewards_t + GAMMA * (1 - dones_t) * target_selected_q_values

  #Compute Loss
  #############
  q_values = online_network(obses_t)

  #getting q_value for the action taken
  action_q_values = torch.gather(input=q_values, dim=1, index=actions_t)

  #huber loss function (smooth_l1_loss in pytorch)
  loss = nn.functional.smooth_l1_loss(action_q_values, targets)

  #Gradient Descent
  ##################

  optimiser.zero_grad()
  loss.backward()
  optimiser.step()

  #update target network parameters
  if step % TARGET_UPDATE_FREQUENCY == 0:
    target_network.load_state_dict(online_network.state_dict())

  #Logging
  if step % 1000 == 0:
    reward_mean = np.mean(reward_buffer)
    print()
    print('Step', step)
    print('Avg reward', reward_mean)

    writer.add_scalar("Avg reward", reward_mean, global_step = step)

  if step % 1000000 == 0:
    torch.save(online_network.state_dict(), "/homes/mat66/RL-team/saved_models/trained_ddqn_"+str(step))

#Limits training
  if reward_mean > 50:
    break


writer.flush()

writer.close()

#open in vs code extension or in the terminal write "tensorboard --logdir ./logs"

torch.save(online_network.state_dict(), "/homes/mat66/RL-team/saved_models/trained_ddqn")




