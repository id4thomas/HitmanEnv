import gym
from gym import error, spaces, utils
from gym.utils import seeding

MAP={
  #hitman&enemy location
  # >=0 : Path Exist
	#0: Hitman
	#1: Available Path
	#2: Goal
	#3,4,5,6: Enemy (Blue) Up,Down,Left,Right
	#-1: Out of Bounds
  "loc": [
    [-1.,-1.,-1.,-1.,-1.,-1.,-1.],
    [-1.,-1.,-1.,-1.,-1.,-1.,-1.],
    [-1.,2.,-1.,-1.,-1.,-1.,-1.],
    [-1.,1.,-1.,-1.,-1.,-1.,-1.],
    [-1.,1.,5.,1.,1.,0.,-1.],
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
    [-1.,4.,-1.,-1.,-1.,-1.,-1.],
    [-1.,12.,-1.,-1.,-1.,-1.,-1.],
    [-1.,9.,3.,3.,3.,2.,-1.],
    [-1.,-1.,-1.,-1.,-1.,-1.,-1.],
    [-1.,-1.,-1.,-1.,-1.,-1.,-1.],
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
