# HitmanEnv
Hitman-Go environments based on OpenAI gym.

## Installation
```bash
cd HitmanEnv
pip install -e .
```

## Usage
```python
import gym
import hitman_gym

env = gym.make('hitman-v0')

while not done:
  obs, reward, done, info = env.step(env.action_space.sample())
  ep_reward += sum(reward_n)

env.close()
```


Sungkyunkwan University Reinforcement Learning Spring 2020
Youngrok Song
Jiwung Hyun

RL OpenAI Gym Environment for Hitman Go
