import gym
from gym import error, spaces, utils
from gym.utils import seeding

import numpy as np

MAP={
  #hitman&enemy location
  # >=0 : Path Exist
	#0: Hitman
	#1: Available Path
	#2: Goal
	#3: Enemy (Blue)
	#-1: Out of Bounds
  "loc": [
    [-1.,-1.,-1.,-1.,-1.,-1.,-1.],
    [-1.,-1.,-1.,-1.,-1.,-1.,-1.],
    [-1.,2.,-1.,-1.,-1.,-1.,-1.],
    [-1.,1.,-1.,-1.,-1.,-1.,-1.],
    [-1.,1.,3.,1.,1.,0.,-1.],
    [-1.,-1.,-1.,-1.,-1.,-1.,-1.],
    [-1.,-1.,-1.,-1.,-1.,-1.,-1.],
  ],
  #enemies direction
  # 0: No dir
  # 1: Up
  # 2: Down
  # 3: Left
  # 4: Right
  "dir": [
    [-1.,-1.,-1.,-1.,-1.,-1.,-1.],
    [-1.,-1.,-1.,-1.,-1.,-1.,-1.],
    [-1.,2.,-1.,-1.,-1.,-1.,-1.],
    [-1.,1.,-1.,-1.,-1.,-1.,-1.],
    [-1.,1.,3.,1.,1.,0.,-1.],
    [-1.,-1.,-1.,-1.,-1.,-1.,-1.],
    [-1.,-1.,-1.,-1.,-1.,-1.,-1.],
  ],
  # 1~16 : 4^2 
  # 0: No connection
  # 1000: Up
  # 0100: Down
  # 0010: Left
  # 0001: Right
  "conn": [
    [-1.,-1.,-1.,-1.,-1.,-1.,-1.],
    [-1.,-1.,-1.,-1.,-1.,-1.,-1.],
    [-1.,2.,-1.,-1.,-1.,-1.,-1.],
    [-1.,1.,-1.,-1.,-1.,-1.,-1.],
    [-1.,1.,3.,1.,1.,0.,-1.],
    [-1.,-1.,-1.,-1.,-1.,-1.,-1.],
    [-1.,-1.,-1.,-1.,-1.,-1.,-1.],
  ],
}

#Simple Map: Agent move to goal
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
    self.map=np.zeros((7,7))
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
    

if __name__ == "__main__":
    hm1=HitmanMap1()
    print(hm1.map)
