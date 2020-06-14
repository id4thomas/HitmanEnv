import gym
import hitman_gym

env = gym.make('hitman-v1')  # hitman-v1: 'simple' , hitman-v2: 'blue', hitman-v3: 'yellow'
done = False
ep_reward = 0

while not done:
  obs, reward, done, info = env.step(env.action_space.sample())
  ep_reward += reward

env.close()