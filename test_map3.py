import gym
import hitman_gym

import os

hm = gym.make('hitman-v2')

s = hm.reset()
print(s.shape)
# ans_path = [2, 2, 2, 2, 0, 0]
#0 up 1 down 2 left 3 right
ans_path = [3,1,1,3,3,2,2,3,2,3,3,0,0,3]
for i in range(len(ans_path)):
    # self.cur_state, reward, done, {}
    a = ans_path[i]
    print('\n\nSTEP {} {}'.format(i + 1, a))
    s, r, d, _ = hm.step(a)
    print('Step {} Reward {} Pos{},{}'.format(a, r, hm.cur_loc[0], hm.cur_loc[1]))
    print(hm.cur_state[0])
    if d:
        break