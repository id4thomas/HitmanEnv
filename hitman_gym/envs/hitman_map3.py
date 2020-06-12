import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
from hitman_gym.envs.enemies import BlueEnemy
#from hitman_gym.envs.enemies import YellowEnemy
yellow_jw={
  #hitman&enemy location
  # >=0 : Path Exist
	#0: Hitman
	#1: Available Path
	#2: Goal
	#3,4,5,6: Enemy (Blue) Up,Down,Left,Right
	#-1: Out of Bounds
  "loc": [
    [-1.,-1.,-1.,-1.,-1.,-1.,-1.],
    [-1.,1.,1.,1.,1.,-1.,-1.],
    [-1.,-1.,3.,3.,1.,-1.,-1.],
    [-1.,0.,1.,1.,1.,2.,-1.],
    [-1.,-1.,1.,1.,4.,-1.,-1.],
    [-1.,-1.,1.,1.,1.,-1.,-1.],
    [-1.,-1.,-1.,-1.,-1.,-1.,-1.],
  ],
  # 1~16 : 4^2 
  # 0: No connection
  # 1000: Up 8
  # 0100: Down 4
  # 0010: Left 2
  # 0001: Right 1
  "conn": [
    [-1.,-1.,-1.,-1.,-1.,-1.,-1.],
    [-1.,1.,7.,7.,6.,-1.,-1.],
    [-1.,-1.,13.,14.,12.,-1.,-1.],
    [-1.,1.,14.,12.,13.,2.,-1.],
    [-1.,-1.,12.,8.,12.,-1.,-1.],
    [-1.,-1.,9.,3.,10.,-1.,-1.],
    [-1.,-1.,-1.,-1.,-1.,-1.,-1.],
    ],
  }
#yellow 2: 7~10
yellow_yr={
  #hitman&enemy location
  # >=0 : Path Exist
	#0: Hitman
	#1: Available Path
	#2: Goal
	#3,4,5,6: Enemy (Blue) Up,Down,Left,Right
  #7,8,9,10: Enemy (Blue) Up,Down,Left,Right
	#-1: Out of Bounds
  "loc": [
    [-1.,-1.,-1.,-1.,-1.,-1.,-1.],
    [-1.,1.,1.,1.,1.,-1.,-1.],
    [-1.,-1.,7.,7.,1.,-1.,-1.],
    [-1.,0.,1.,1.,1.,2.,-1.],
    [-1.,-1.,1.,1.,8.,-1.,-1.],
    [-1.,-1.,1.,1.,1.,-1.,-1.],
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
    [-1.,1.,7.,7.,6.,-1.,-1.],
    [-1.,-1.,13.,14.,12.,-1.,-1.],
    [-1.,1.,14.,12.,13.,2.,-1.],
    [-1.,-1.,12.,8.,12.,-1.,-1.],
    [-1.,-1.,9.,3.,10.,-1.,-1.],
    [-1.,-1.,-1.,-1.,-1.,-1.,-1.],
    ],
  }

# Blue enemy
class HitmanMap3(gym.Env):
    metadata = {'render.modes': ['human']}

    # Actions
    # 0: Move Up
    # 1: Move Down
    # 2: Move Left
    # 3: Move Right

    # State
    # Hitman location & enemy location (map)

    def __init__(self):
        self.map = np.zeros((7, 7))
        self.action_space

        self.dr = [-1, 1, 0, 0]
        self.dc = [0, 0, -1, 1]

        self.enemies = []
        self.goal_loc = [3, 5]

    def can_move(self,enemy):
        a=enemy.dir
        next_pos=enemy.get_moved()

        return True

    def step(self, action):
        # check legal move
        r_legal1 = (self.cur_loc[0] + self.dr[action]) >= 7
        r_legal2 = (self.cur_loc[0] + self.dr[action]) < 0
        c_legal1 = (self.cur_loc[1] + self.dc[action]) >= 7
        c_legal2 = (self.cur_loc[1] + self.dc[action]) < 0
        illegal = r_legal1 | r_legal2 | c_legal1 | c_legal2
        if illegal:
            done = True
            reward = -1
            #print("Illegal Move")
        else:
            # move hitman (Edit cur_loc)
            prev_r = self.cur_loc[0]
            prev_c = self.cur_loc[1]

            self.cur_loc[0] += self.dr[action]
            self.cur_loc[1] += self.dc[action]

            # default
            reward = 0
            done = False

            # 1-check goal
            if self.cur_loc[0] == self.goal_loc[0] and self.cur_loc[1] == self.goal_loc[1]:
                reward = 1
                done = True
                #print('GOAL REACHED')
            # 2-check out of bounds (-1)
            elif self.cur_state[0][self.cur_loc[0], self.cur_loc[1]] < 0:
                reward = -1
                #print("OOB!")
                done = True
            # 3-perform moves
            else:
                # Fixed Enemies: check range for moved position
                caught = []
                for i in range(len(self.enemies)):
                    e = self.enemies[i]
                    if e.check_range(self.cur_loc[0], self.cur_loc[1]):
                        done = True
                        reward = -1
                        #print('Hitman Caught!', len(self.enemies))
                        #print(e.pos)
                        break
                    # removed enemy
                    elif e.check_caught(self.cur_loc[0], self.cur_loc[1]):
                        caught.append(e)
                # remove caught enemies
                print('Caught {} enemies'.format(len(caught)))
                for c in caught:
                    self.enemies.remove(c)

                # Moving Enemies
                # Check caught
                caught = []
                
                for i in range(len(self.move_enemies)):
                    e = self.move_enemies[i]
                    #Hitman Caught
                    e_pos=e.pos
                    if e.check_range(self.cur_loc[0], self.cur_loc[1],self.cur_state[1]e.pos):
                        done = True
                        reward = -1
                        break
                    # removed enemy
                    elif e.check_caught(self.cur_loc[0], self.cur_loc[1]):
                        caught.append(e)

                for c in caught:
                    self.move_enemies.remove(c)

                #Move Enemies & Check after moving
                for m_e in self.move_enemies:
                    #Move
                    prev_pos=m_e.pos
                    moved=m_e.moved_pos(prev_pos)
                    m_e.pos=moved

                    #check next move illegal - if need to turn around
                    next_moved=m_e.moved_pos(moved)
                    r_legal1 = (next_moved[0] + self.dr[m_e.dir]) >= 7
                    r_legal2 = (next_moved[0] + self.dr[m_e.dir])  < 0
                    c_legal1 = (next_moved[1] + self.dc[m_e.dir])  >= 7
                    c_legal2 = (next_moved[1] + self.dc[m_e.dir])  < 0
                    conn="{0:4b}".format(int(self.cur_state[1]moved))[-4:]
                    oob = (self.cur_state[0]next_moved=='-1')
                    not_conn=conn[m_e.dir]!='1'
                    illegal = r_legal1 | r_legal2 | c_legal1 | c_legal2 | not_conn | oob

                    if illegal:
                        #turn around
                        new_dir={0:1,1:0,2:3,3:2}.get(m_e.dir)
                        m_e.dir=new_dir
                        #update map
                        self.cur_state[0]prev_pos='1'
                        self.cur_state[0]moved=str(3+self.dir)

                # Check Again
                for i in range(len(self.move_enemies)):
                    e = self.move_enemies[i]
                    #Hitman Caught
                    e_pos=e.pos
                    if e.check_range(self.cur_loc[0], self.cur_loc[1],self.cur_state[1]e.pos):
                        done = True
                        reward = -1
                        break
                # move hitman
                self.cur_state[0][prev_r, prev_c] = 1
                self.cur_state[0][self.cur_loc[0], self.cur_loc[1]] = 0

        return self.cur_state, reward, done, {}
        # return np.array(self.state), reward, done, {}

    def reset(self):
        # Reset Map
        MAP=yellow_jw
        # MAP=yellow_yr
        loc = np.array(MAP['loc'])  # (7,7)
        conn = np.array(MAP['conn'])  # (7,7)
        print('loc', loc.shape)
        print('conn', conn.shape)
        self.cur_state = np.stack([loc, conn], axis=0)
        print('stacked', self.cur_state.shape)

        # Reset Positions
        self.cur_loc = [3, 1]

        # Reset Enemies
        self.enemies = []
        self.move_enemies=[]
        # self.enemies.append(YellowEnemy(2, 2, 3, self.cur_state[1][2,2]))  # lane1
        # self.enemies.append(YellowEnemy(2, 3, 3, self.cur_state[1][2,3]))  # lane2
        # self.enemies.append(YellowEnemy(4, 4, 4, self.cur_state[1][4,4]))  # lane3
        self.move_enemies.append(YellowEnemy(2, 2, 3))  # lane1
        self.move_enemies.append(YellowEnemy(2, 3, 3))  # lane2
        self.move_enemies.append(YellowEnemy(4, 4, 4))  # lane3


        return self.cur_state  # (2,7,7)

    '''
  def render(self, mode='human'):
    ...

  '''

    def close(self):
        pass


