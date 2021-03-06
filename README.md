# HitmanEnv
Hitman-Go environments based on OpenAI gym.

## Requirements
* gym 0.14.0
* numpy

## Installation (clone this repository and just import it)
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
  ep_reward += reward

env.close()
```

## Usage (hitman-v1 ~ hitman-v4)
Another versions of Hitman-Go do not need to register environments. By "gym.make" helps you use the hitman-go environments.
```python
import gym
import hitman_gym

env = gym.make('hitman-v1')  # hitman-v1: 'simple' , hitman-v2: 'blue', hitman-v3: 'yellow', hitman-v4: 'yellow_yr'
done = False
ep_reward = 0

while not done:
  obs, reward, done, info = env.step(env.action_space.sample())
  ep_reward += reward

env.close()
```

## Dueling DQN Train (hitman-v0)
```bash
python train.py --num_episodes 10000 --min_eps 0.01 --max_mem 1000 --map simple
```
## Dueling DQN Train (hitman-v1 ~ hitman-v4)
```bash
python train2.py --num_episodes 10000 --min_eps 0.01 --max_mem 1000 --ver 1
```

## Notice
Sungkyunkwan University Reinforcement Learning Spring 2020
Youngrok Song
Jiwung Hyun

RL OpenAI Gym Environment for Hitman Go
