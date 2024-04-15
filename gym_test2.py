
import gym
env = gym.make("ALE/Boxing-v5", render_mode="human")
env.reset()

for i in range(1000):
    env.step(env.action_space.sample())
    env.render()