import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
from enemies import BlueEnemy

MAP = {
    # hitman&enemy location
    # >=0 : Path Exist
    # 0: Hitman
    # 1: Available Path
    # 2: Goal
    # 3,4,5,6: Enemy (Blue) Up,Down,Left,Right
    # -1: Out of Bounds
    "loc": [
        [-1., -1., -1., -1., -1., -1., -1.],
        [-1., 0., 1., 1., 4., -1., -1.],
        [-1., 1., 3., 5., 1., 1., -1.],
        [-1., 1., 3., 5., 3., 2., -1.],
        [-1., 6., 1., 1., 1., -1., -1.],
        [-1., -1., -1., -1., -1., -1., -1.],
        [-1., -1., -1., -1., -1., -1., -1.],
    ],
    # 0~15 : 4^2
    # 0: No connection
    # 1000: Up
    # 0100: Down
    # 0010: Left
    # 0001: Right
    "conn": [
        [-1., -1., -1., -1., -1., -1., -1.],
        [-1., 5., 7., 3., 6., -1., -1.],
        [-1., 12., 15., 12., 14., 3., -1.],
        [-1., 12., 15., 14., 13., 4., -1.],
        [-1., 9., 11., 3., 10., -1., -1.],
        [-1., -1., -1., -1., -1., -1., -1.],
        [-1., -1., -1., -1., -1., -1., -1.],
    ],
}


# Blue enemy
class HitmanMap2(gym.Env):
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
            print("Illegal Move")
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
                print('GOAL REACHED')
            # 2-check out of bounds
            elif self.cur_state[0][self.cur_loc[0], self.cur_loc[1]] < 0:
                reward = -1
                print("OOB!")
                done = True
            # 3-perform
            else:
                # all enemies: check range
                caught = []
                for i in range(len(self.enemies)):
                    e = self.enemies[i]
                    if e.check_range(self.cur_loc[0], self.cur_loc[1]):
                        done = True
                        reward = -1
                        print('Hitman Caught!', len(self.enemies))
                        print(e.pos)
                        break
                    # removed enemy
                    elif e.check_caught(self.cur_loc[0], self.cur_loc[1]):
                        caught.append(e)
                # remove caught enemies
                print('Caught {} enemies'.format(len(caught)))
                for c in caught:
                    self.enemies.remove(c)
                # move hitman
                self.cur_state[0][prev_r, prev_c] = 1
                self.cur_state[0][self.cur_loc[0], self.cur_loc[1]] = 0

        return self.cur_state, reward, done, {}
        # return np.array(self.state), reward, done, {}

    def reset(self):
        # Reset Map
        loc = np.array(MAP['loc'])  # (7,7)
        conn = np.array(MAP['conn'])  # (7,7)
        print('loc', loc.shape)
        print('conn', conn.shape)
        self.cur_state = np.stack([loc, conn], axis=0)
        print('stacked', self.cur_state.shape)

        # Reset Positions
        self.cur_loc = [1, 1]

        # Reset Enemies
        self.enemies = []
        self.enemies.append(BlueEnemy(4, 1, 6, self.cur_state[1][4,1]))  # step 3
        self.enemies.append(BlueEnemy(3, 4, 5, self.cur_state[1][3,4]))  # step 7
        self.enemies.append(BlueEnemy(3, 3, 5, self.cur_state[1][3,3]))  # step 8
        self.enemies.append(BlueEnemy(3, 2, 3, self.cur_state[1][3,2]))  # step 9
        self.enemies.append(BlueEnemy(2, 3, 5, self.cur_state[1][2,3]))  # step 11
        self.enemies.append(BlueEnemy(2, 2, 3, self.cur_state[1][2,2]))  # step 13
        self.enemies.append(BlueEnemy(1, 4, 4, self.cur_state[1][1,4]))  # step 15

        return self.cur_state  # (2,7,7)

    '''
  def render(self, mode='human'):
    ...

  '''

    def close(self):
        pass


if __name__ == "__main__":
    hm1 = HitmanMap2()
    print(hm1.map)
    s = hm1.reset()
    print(s.shape)
    # 0 up 1 down 2 left 3 right
    ans_path = [1, 1, 1, 3, 3, 3, 0, 2, 2, 3, 0, 2, 0, 3, 3, 1, 3, 1]
    for i in range(len(ans_path)):
        a = ans_path[i]
        print('\n\nSTEP {} {}'.format(i + 1, a))
        # self.cur_state, reward, done, {}
        s, r, d, _ = hm1.step(a)
        print('Step {} Reward {} Pos{},{}'.format(a, r, hm1.cur_loc[0], hm1.cur_loc[1]))
        print(hm1.cur_state[0])
        if d:
            break
