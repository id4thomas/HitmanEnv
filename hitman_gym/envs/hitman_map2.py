import gym
from gym import error, spaces, utils
from gym.utils import seeding

MAP={
  #hitman&enemy location
  "loc": [
    [0.,0.,0.,0.,0.,0.,0.],
    [0.,0.,0.,0.,0.,0.,0.],
    [0.,0.,0.,0.,0.,0.,0.],
    [0.,0.,0.,0.,0.,0.,0.],
    [0.,0.,0.,0.,0.,0.,0.],
    [0.,0.,0.,0.,0.,0.,0.],
    [0.,0.,0.,0.,0.,0.,0.],
  ],
  #enemies direction
  "dir": [
    [0.,0.,0.,0.,0.,0.,0.],
    [0.,0.,0.,0.,0.,0.,0.],
    [0.,0.,0.,0.,0.,0.,0.],
    [0.,0.,0.,0.,0.,0.,0.],
    [0.,0.,0.,0.,0.,0.,0.],
    [0.,0.,0.,0.,0.,0.,0.],
    [0.,0.,0.,0.,0.,0.,0.],
  ],
}

#Blue enemy
class HitmanMap1(gym.Env):
  metadata = {'render.modes': ['human']}

  #Actions
  # 0: Move Up
  # 1: Move Down
  # 2: Move Left
  # 3: Move Right

  #State
  #Hitman location & enemy location (map)

  def __init__(self):
    self.map
    self.action_space

  def step(self, action):
    pass
    #return np.array(self.state), reward, done, {}
  def reset(self):
    pass
    #return np.array(self.state)
  '''
  def render(self, mode='human'):
    ...

  '''
  def close(self):
    pass
