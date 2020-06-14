# HitmanEnv
Hitman-Go environments based on OpenAI gym.

## Installation
```bash
cd HitmanEnv
pip install -e .
```

## Usage (hitman-v0)
hitman-v0 environments is collective version of Hitman-Go environments. You can select maps by parameters (map_id)

```python
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
  ep_reward += sum(reward)

env.close()
```

## Usage (hitman-v1 ~ hitman-v4)
Another versions of Hitman-Go do not need to register environments. By "gym.make" helps you use the hitman-go environments.
```python
import gym
import hitman_gym

env = gym.make('hitman-v1')  # hitman-v1: 'simple' , hitman-v2: 'blue', hitman-v3: 'yellow'
done = False
ep_reward = 0

while not done:
  obs, reward, done, info = env.step(env.action_space.sample())
  ep_reward += sum(reward)

env.close()
```


## Notice
Sungkyunkwan University Reinforcement Learning Spring 2020
Youngrok Song
Jiwung Hyun

RL OpenAI Gym Environment for Hitman Go
