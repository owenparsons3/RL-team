# Atari Breakout Reinforcement Learning

## Overview

In this project, we implement variations of the DQN algorithm to train an agent to play the Atari Breakout game.

## Algorithms Implemented

1. **DQN**: Basic Deep Q-Network algorithm.
2. **DDQN**: Double Deep Q-Network, a variant of DQN with separate networks for action selection and evaluation.
3. **Dueling DQN**: Dueling Deep Q-Network, which separates the value and advantage functions for better learning.
4. **Dueling DDQN**: A combination of Dueling DQN and Double DQN for enhanced performance.

## Folders and Files

- **DQN**: Contains the implementation of the basic DQN algorithm.
  - `DQN.ipynb`: notebook running standerd DQN on cartpole
  - `DQN_Atari.ipynb`: notebook running standard DQN on breakout (no preprocessing)
  - `preprocessed_DQN_Atari.ipynb`: notebook running standerd DQN on breakout with preprocessing of environment
  - `preprocessed_DQN_Atari_hex.py`: python file running standard DQN on breakout with preprocessing of environment
  - `render.py`: python file to render trained model of DQN on breakout

- **DDQN**: Implements the Double DQN variant of the DQN algorithm.
  - `DDQN_Atari.ipynb`: notebook running DDQN on breakout with preprocessing of environment
  - `DDQN_Atari_hex.py`: python file running DDQN on breakout with preprocessing of environment

- **Dueling**: Contains the implementation of the Dueling DQN algorithm.
  - `Dueling_DQN_Atari.ipynb`: notebook running dueling DQN on breakout with preprocessing of environment
  - `Dueling_DQN_Atari_hex.py`:python file running dueling DQN on breakout with preprocessing of environment
  - `dueling_render.py`: python file to render trained model of dueling DQN on breakout

- **Dueling DDQN**: Combines Dueling DQN and Double DQN for improved performance.
  - `Dueling_DDQN_Atari.ipynb`: notebook running dueling DDQN on breakout with preprocessing of environment
  - `Dueling_DDQN_Atari_hex.py`: python file running dueling DDQN on breakout with preprocessing of environment

## Dependencies

- Python 3.x
- torch
- Gym (used in notebooks)
- Gymnasium (used in python files)

## Tutorial Used
This code is adapted from a YouTube Tutorial by brthor. (Source: https://www.youtube.com/watch?v=NP8pXZdU-5U&t=0s retrieved in April 2024.)

