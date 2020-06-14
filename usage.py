import gym
import hitman_gym

from gym.envs.registration import register

map_id = 'simple'  # 'simple', 'blue', 'yellow'

register(
        id='hitman-v0',
        entry_point='hitman_gym.envs:HitmanGO',
        kwargs={'map_id': map_id}
        )

env = gym.make('hitman-v0')
done = False
ep_reward = 0

while not done:
  obs, reward, done, info = env.step(env.action_space.sample())
  ep_reward += reward

env.close()