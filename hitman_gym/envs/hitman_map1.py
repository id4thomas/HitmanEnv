import gym
from gym import error, spaces, utils
from gym.utils import seeding

import numpy as np
from enemies import BlueEnemy
#Attempt 1
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

#Attempt2: Remove Direction Channel
MAP_a2={
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

    self.dr=[-1,1,0,0]
    self.dc=[0,0,-1,1]

    self.enemies=[]
    self.goal_loc=[2,1]
  def step(self, action):
    #check legal move
    r_legal1=(cur_loc[0]+dr[action])>=7
    r_legal2=(cur_loc[0]+dr[action])<0
    c_legal1=(cur_loc[1]+dc[action])>=7
    c_legal2=(cur_loc[1]+dc[action])<0
    illegal=r_legal1|r_legal2|c_legal1|c_legal2
    if illegal:
      done=True
      reward=-1
    else:
      #move
      prev_r=self.cur_loc[0]
      prev_c=self.cur_loc[1]

      self.cur_loc[0]+=self.dr[action]
      self.cur_loc[1]+=self.dc[action]
      #default
      reward=0
      done=False

      #1-check goal
      if self.cur_loc[0]==goal_loc[0] and self.cur_loc[1]==goal_loc[1]:
        reward=1
        done=True
      #2-check out of bounds
      elif self.cur_state[self.cur_loc[0],self.cur_loc[1]]<0:
        reward=-1
        done=True
      #3-perform
      else:
        for i in range(len(enemies)):
          e=enemies[i]
          #enemy caught
          if e.check_range(self.cur_loc[0],self.cur_loc[1]):
            done=True
            reward=-1
            break
          elif e.check_caught(self.cur_loc[0],self.cur_loc[1]):
            break

        #move hitman
        self.cur_state[prev_r,prev_c]=1
        self.cur_state[self.cur_loc[0],self.cur_loc[0]]=0

    return self.cur_state, reward, done, {}

  def reset(self):
    #Reset Map
    loc=np.array(MAP_a2['loc']) #(7,7)
    conn=np.array(MAP_a2['conn']) #(7,7)
    print('loc',loc.shape)
    print('conn',conn.shape)
    self.cur_state=np.stack([loc,conn],axis=0)

    #Reset Positions
    self.cur_loc=[4,5]

    #Reset Enemies
    self.enemies=[]
    self.enemies.append(BlueEnemy(4,2,5))
    return self.cur_state #(2,7,7)
  '''
  def render(self, mode='human'):
    ...

  '''
  def close(self):
    pass
    

if __name__ == "__main__":
    hm1=HitmanMap1()
    print(hm1.map)
    s=hm1.reset()
    print(s.shape)