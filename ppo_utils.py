import numpy as np
import tensorflow as tf
from tensorflow import keras

import gym

import tensorflow.keras.backend as K
import os
import random

class Utils:
    def __init__(self):
        self.gamma=0.99
        self.gae=0.95
    def calc_traj_vals(self,T):
        vals=[]
        for records in T:
            #(s,action,reward,next_s,done,value)
            records.reverse() 
            
            states=np.array([r[0] for r in records])
            actions=np.array([r[1] for r in records])
            values=np.array([r[5] for r in records])

            rewards=[]
            advs=[]

            s_v_next=0
            prevgae = 0
            reward=0
            for r_idx in range(len(records)):
                r=records[r_idx][2]
                done=records[r_idx][4]
                s_v=records[r_idx][5]
                if done:
                    mask=0
                else:
                    mask=1
                #reward=r + self.gamma*reward#*mask
                #reward=r + self.gamma*s_v_next*mask
                delta = r + self.gamma * s_v_next*mask - s_v
                adv = prevgae = delta + self.gae * self.gamma * mask * prevgae
                adv=reward-s_v
                rewards.append(adv+s_v)
                #rewards.append(reward)
                advs.append(adv)
                s_v_next=s_v

            rewards=np.array(rewards)
            advs=np.array(advs).flatten()
            #print(advs.shape,values.shape)
            returns=advs+values.flatten()
            #normalize advantage vals
            advs = (advs - advs.mean()) / (advs.std() + 1e-6)
            advs=np.expand_dims(advs,axis=1)
            vals+=[(states[i],actions[i],rewards[i],advs[i],returns[i]) for i in range(len(records))]
        #print(len(vals))
        return vals
        