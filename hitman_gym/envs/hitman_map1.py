import gym
from gym import error, spaces, utils
from gym.utils import seeding

import numpy as np
from hitman_gym.envs.enemies import BlueEnemy

# Attempt 1
MAP = {

    # Hitman & Enemies Location
    # >=0 : Path Exist
    # 0: Hitman (본인 위치)
    # 1: Available Path (이동 가능한 구역)
    # 2: Goal (목표 지점)
    # 3: Enemy (Blue) (적의 위치)
    # -1: Out of Bounds (장애물)
    "loc": [
        [-1., -1., -1., -1., -1., -1., -1.],
        [-1., -1., -1., -1., -1., -1., -1.],
        [-1., 2., -1., -1., -1., -1., -1.],
        [-1., 1., -1., -1., -1., -1., -1.],
        [-1., 1., 3., 1., 1., 0., -1.],
        [-1., -1., -1., -1., -1., -1., -1.],
        [-1., -1., -1., -1., -1., -1., -1.],
    ],

    # Enemies direction
    # 0: No dir
    # 1: Up
    # 2: Down
    # 3: Left
    # 4: Right
    "dir": [
        [-1., -1., -1., -1., -1., -1., -1.],
        [-1., -1., -1., -1., -1., -1., -1.],
        [-1., 2., -1., -1., -1., -1., -1.],
        [-1., 1., -1., -1., -1., -1., -1.],
        [-1., 1., 3., 1., 1., 0., -1.],
        [-1., -1., -1., -1., -1., -1., -1.],
        [-1., -1., -1., -1., -1., -1., -1.],
    ],

    # 1~16 : 4^2
    # 0: No connection
    # 1000: Up
    # 0100: Down
    # 0010: Left
    # 0001: Right
    "conn": [
        [-1., -1., -1., -1., -1., -1., -1.],
        [-1., -1., -1., -1., -1., -1., -1.],
        [-1., 2., -1., -1., -1., -1., -1.],
        [-1., 1., -1., -1., -1., -1., -1.],
        [-1., 1., 3., 1., 1., 0., -1.],
        [-1., -1., -1., -1., -1., -1., -1.],
        [-1., -1., -1., -1., -1., -1., -1.],
    ],
}
# Attempt2: Remove Direction Channel
MAP_a2 = {
    # hitman&enemy location
    # >=0 : Path Exist
    # 0: Hitman
    # 1: Available Path
    # 2: Goal
    # 3,4,5,6: Enemy (Blue) Up,Down,Left,Right
    # -1: Out of Bounds
    "loc": [
        [-1., -1., -1., -1., -1., -1., -1.],
        [-1., -1., -1., -1., -1., -1., -1.],
        [-1., 2., -1., -1., -1., -1., -1.],
        [-1., 1., -1., -1., -1., -1., -1.],
        [-1., 1., 5., 1., 1., 0., -1.],
        [-1., -1., -1., -1., -1., -1., -1.],
        [-1., -1., -1., -1., -1., -1., -1.],
    ],
    # 1~16 : 4^2
    # 0: No connection
    # 1000: Up
    # 0100: Down
    # 0010: Left
    # 0001: Right
    "conn": [
        [-1., -1., -1., -1., -1., -1., -1.],
        [-1., -1., -1., -1., -1., -1., -1.],
        [-1., 4., -1., -1., -1., -1., -1.],
        [-1., 12., -1., -1., -1., -1., -1.],
        [-1., 9., 3., 3., 3., 2., -1.],
        [-1., -1., -1., -1., -1., -1., -1.],
        [-1., -1., -1., -1., -1., -1., -1.],
    ],
}

map2 = {
    # hitman&enemy location
    # >=0 : Path Exist
    # 0: Hitman
    # 1: Available Path
    # 2: Goal
    # 3,4,5,6: Enemy (Blue) Up,Down,Left,Right
    # -1: Out of Bounds
    "loc": [
        [-1., -1., -1., -1., -1., -1., -1.],
        [-1., 0., 1., -1., -1., -1., -1.],
        [-1., 1., 1., 1., -1., -1., -1.],
        [-1., 1., 1., 1., -1., -1., -1.],
        [-1., -1., -1., 1., 2., -1., -1.],
        [-1., -1., -1., -1., -1., -1., -1.],
        [-1., -1., -1., -1., -1., -1., -1.],
    ],
    # 1~16 : 4^2
    # 0: No connection
    # 1000: Up
    # 0100: Down
    # 0010: Left
    # 0001: Right
    "conn": [
        [-1., -1., -1., -1., -1., -1., -1.],
        [-1., 5., 6., -1., -1., -1., -1.],
        [-1., 12., 13., 6., -1., -1., -1.],
        [-1., 9., 11., 14., -1., -1., -1.],
        [-1., -1., -1., 9., 2., -1., -1.],
        [-1., -1., -1., -1., -1., -1., -1.],
        [-1., -1., -1., -1., -1., -1., -1.],
    ],
}


# Simple Map: Agent move to goal
class HitmanMap1(gym.Env):
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

        self.cur_state = None
        self.cur_loc = None

        self.dr = [-1, 1, 0, 0]
        self.dc = [0, 0, -1, 1]

        self.enemies = []
        # self.goal_loc=[2,1]
        self.goal_loc = [4, 4]

    def step(self, action):
        # check legal move

        r_legal1 = (self.cur_loc[0] + self.dr[action]) >= 7
        r_legal2 = (self.cur_loc[0] + self.dr[action]) < 0
        c_legal1 = (self.cur_loc[1] + self.dc[action]) >= 7
        c_legal2 = (self.cur_loc[1] + self.dc[action]) < 0
        #check if connected move
        conn = "{0:4b}".format(int(self.cur_state[1][self.cur_loc[0],self.cur_loc[1]]))[-4:]
        not_conn = conn[action] != '1'
        illegal = r_legal1 | r_legal2 | c_legal1 | c_legal2 | not_conn
        if illegal:
            done = True
            reward = -1
        else:
            # move
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
                print("reach goal loc {} {}".format(self.cur_loc[0], self.cur_loc[1]))
                # print('GOAL REACHED')
            # 2-check out of bounds
            elif self.cur_state[0][self.cur_loc[0], self.cur_loc[1]] < 0:
                reward = -1
                done = True
            # 3-perform
            else:
                # all enemies: check range
                caught = []
                for i in range(len(self.enemies)):
                    e = self.enemies[i]
                    # enemy caught
                    if e.check_range(self.cur_loc[0], self.cur_loc[1]):
                        done = True
                        reward = -1
                        break
                    elif e.check_caught(self.cur_loc[0], self.cur_loc[1]):
                        caught.append(e)
                # remove caught enemies
                # print('Caught {} enemies'.format(len(caught)))
                for c in caught:
                    self.enemies.remove(c)
                # move hitman
                self.cur_state[0][prev_r, prev_c] = 1
                self.cur_state[0][self.cur_loc[0], self.cur_loc[1]] = 0

        hitman_loc = self.cur_loc.copy()
        state = self.cur_state.copy()

        return state, reward, done, [hitman_loc, self.goal_loc]

    def reset(self):
        # Reset Map
        selected_map = map2
        loc = np.array(selected_map['loc'])  # (7,7)
        conn = np.array(selected_map['conn'])  # (7,7)
        # print('loc', loc.shape)
        # print('conn', conn.shape)
        self.cur_state = np.stack([loc, conn], axis=0)

        # Reset Positions
        # self.cur_loc = [4, 5]
        self.cur_loc = [1, 1]  # map2

        # Reset Enemies
        self.enemies = []
        # self.enemies.append(BlueEnemy(4, 2, 5, self.cur_state[1][4,2])) #step 3
        return self.cur_state  # (2,7,7)


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
