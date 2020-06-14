import gym
from gym import error, spaces, utils
from gym.utils import seeding

import numpy as np
from hitman_gym.envs.enemies import BlueEnemy
from hitman_gym.envs.enemies import YellowEnemy

#MAP FORMAT
# loc: Hitman & Enemy Location
# hitman&enemy location
# >=0 : Path Exist
# 0: Hitman
# 1: Available Path
# 2: Goal
# 3,4,5,6: Enemy Up,Down,Left,Right
# -1: Out of Bounds

# conn: How maps are connected
# 1~16 : 4^2
# 0: No connection
# 1000: Up
# 0100: Down
# 0010: Left
# 0001: Right

#init : Hitman Start location
#goal : Goal Location
 
#Moving Enemies
#sol: ans_path = [3,1,1,3,3,2,2,3,2,3,3,0,0,3]
yellow={
    "loc": [
        [-1., -1., -1., -1., -1., -1., -1.],
        [-1., 1., 1., 1., 1., -1., -1.],
        [-1., -1., 3., 3., 1., -1., -1.],
        [-1., 0., 1., 1., 1., 2., -1.],
        [-1., -1., 1., 1., 4., -1., -1.],
        [-1., -1., 1., 1., 1., -1., -1.],
        [-1., -1., -1., -1., -1., -1., -1.],
    ],
    "conn": [
        [-1., -1., -1., -1., -1., -1., -1.],
        [-1., 1., 7., 7., 6., -1., -1.],
        [-1., -1., 13., 14., 12., -1., -1.],
        [-1., 1., 14., 12., 13., 2., -1.],
        [-1., -1., 12., 8., 12., -1., -1.],
        [-1., -1., 9., 3., 10., -1., -1.],
        [-1., -1., -1., -1., -1., -1., -1.],
    ],
    "init":[3,1],
    "goal":[3,5],
    "fixed":[],
    "moving":[[2,2,0],[2,3,0],[4,4,1]]
}

# Simple Map: Agent move to goal
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
        self.action_space = spaces.Discrete(4)  
        self.dr = [-1, 1, 0, 0]
        self.dc = [0, 0, -1, 1]
        self.map=yellow

    def step(self, action):

        #Check if Move Possible (OOB, Connection)
        conn = "{0:4b}".format(int(self.cur_state[1][self.cur_loc[0],self.cur_loc[1]]))[-4:]
        not_conn = conn[action] != '1'
        #No Path - Done
        if not_conn:
            done=True
            reward=-1
            self.cur_loc[0] += self.dr[action]
            self.cur_loc[1] += self.dc[action]
            #make info
            hitman_loc = self.cur_loc.copy()
            state = self.cur_state.copy()
            return state, reward, done, [hitman_loc, self.goal_loc]

        # move hitman (Edit cur_loc)
        prev_r = self.cur_loc[0]
        prev_c = self.cur_loc[1]

        self.cur_loc[0] += self.dr[action]
        self.cur_loc[1] += self.dc[action]

        # default Reward
        reward = 0
        done = False


        # 1-check goal reached
        if self.cur_loc[0] == self.goal_loc[0] and self.cur_loc[1] == self.goal_loc[1]:
            reward = 1
            # move hitman
            self.cur_state[0][prev_r, prev_c] = 1
            self.cur_state[0][self.cur_loc[0], self.cur_loc[1]] = 0
            done = True
            print('GOAL REACHED', self.goal_loc)

        # 2-check out of bounds (-1)
        elif self.cur_state[0][self.cur_loc[0], self.cur_loc[1]] < 0:
            reward = -1
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
                    # print('Hitman Caught!', len(self.enemies))
                    # print(e.pos)
                    break
                # removed enemy
                elif e.check_caught(self.cur_loc[0], self.cur_loc[1]):
                    caught.append(e)
            # remove caught enemies
            for c in caught:
                self.enemies.remove(c)

            # Moving Enemies
            # Check if moving enemies caught
            caught = []
            for i in range(len(self.move_enemies)):
                e = self.move_enemies[i]
                # Hitman Caught
                e_pos = e.pos
                if e.check_range(self.cur_loc[0], self.cur_loc[1], self.cur_state[1][e_pos[0], e_pos[1]]):
                    done = True
                    reward = -1
                    break
                # removed enemy
                elif e.check_caught(self.cur_loc[0], self.cur_loc[1]):
                    caught.append(e)

            # Remove Caught Enemies
            for c in caught:
                self.move_enemies.remove(c)

            # Move Enemies & Check if turning around
            for m_e in self.move_enemies:
                # Move
                prev_pos = m_e.pos
                moved = m_e.moved_pos(prev_pos)
                m_e.pos = moved

                # check next move illegal - if need to turn around
                next_moved = m_e.moved_pos(moved) #Potential Next Position
                conn = "{0:4b}".format(int(self.cur_state[1][moved[0], moved[1]]))[-4:]
                oob = (self.cur_state[0][next_moved[0], next_moved[1]] == -1) #Out of Bounds
                not_conn = conn[m_e.dir] != '1' #No Path
                illegal = not_conn | oob

                # update map
                self.cur_state[0][prev_pos[0], prev_pos[1]] = 1.

                # turn around
                if illegal:
                    new_dir = {0: 1, 1: 0, 2: 3, 3: 2}.get(m_e.dir)
                    m_e.dir = new_dir
                    # Update Mape
                    self.cur_state[0][moved[0], moved[1]] = 3 + m_e.dir

                    # check if hitman caught after turning
                    if e.check_range(self.cur_loc[0], self.cur_loc[1], self.cur_state[1][moved[0], moved[1]]):
                        done = True
                        reward = -1
                        break
                else:
                    # Update Map
                    self.cur_state[0][moved[0], moved[1]] = 3 + m_e.dir

            # move hitman
            self.cur_state[0][prev_r, prev_c] = 1
            self.cur_state[0][self.cur_loc[0], self.cur_loc[1]] = 0

        #make info
        hitman_loc = self.cur_loc.copy()
        state = self.cur_state.copy()

        return state, reward, done, [hitman_loc, self.goal_loc]

    def reset(self):
        #load map
        selected_map=self.map

        init_loc = selected_map["init"].copy()
        self.goal_loc = selected_map["goal"].copy()

        # Reset Map
        loc = np.array(selected_map['loc'].copy())  # (7,7)
        conn = np.array(selected_map['conn'].copy())  # (7,7)
        self.cur_state = np.stack([loc, conn], axis=0)

        # Reset Positions
        self.cur_loc = init_loc.copy()

        # Reset Enemies
        self.enemies = []
        self.move_enemies = []
        for e in selected_map['fixed']:
            self.enemies.append(BlueEnemy(e[0],e[1],e[2],e[3]))#row,col,dir,conn
        for e in selected_map['moving']:
            self.move_enemies.append(YellowEnemy(e[0],e[1],e[2]))

        return self.cur_state.copy()  # (2,7,7)


def close(self):
    pass


if __name__ == "__main__":
    hm1 = HitmanMap1()
    print(hm1.map)
    s = hm1.reset()
    print(s.shape)
    # ans_path = [2, 2, 2, 2, 0, 0]
    ans_path = [1, 1, 3, 3, 1, 3]
    for i in range(len(ans_path)):
        # self.cur_state, reward, done, {}
        a = ans_path[i]
        print('\n\nSTEP {} {}'.format(i + 1, a))
        s, r, d, _ = hm1.step(a)
        print('Step {} Reward {} Pos{},{}'.format(a, r, hm1.cur_loc[0], hm1.cur_loc[1]))
        print(hm1.cur_state[0])
        if d:
            break
