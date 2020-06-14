import numpy as np
import tensorflow as tf
from tensorflow import keras

import gym
import hitman_gym

import tensorflow.keras.backend as K
import os
import random
import argparse
from tensorflow.keras import layers
#my utilss
from ppo_utils import Utils

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class ACModel():
    def __init__(self,env):
        self.action_size=4
        #self.s_size=28


        self.actor_lr=1e-4
        self.critic_lr=1e-3
        self.lr=1e-4
        #self.actor,self.critic=self.make_model()
        self.actor=self.make_actor()
        self.critic=self.make_critic()
        self.critic,self.actor=self.make_model()
        self.gamma=0.99

        #PPO Parameters
        self.ppo_epochs=10
        self.batch_size=64
        self.cliprange=0.2

        self.op=tf.keras.optimizers.Adam(learning_rate=self.lr)
        self.e=1

    def make_model(self):
        in1=tf.keras.layers.Input(shape=(7, 7, 2,))
        d1=tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(in1)
        d1=tf.keras.layers.Dropout(0.1)(d1)
        d2=tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(d1)
        d2=tf.keras.layers.Dropout(0.1)(d2)
        d2=tf.keras.layers.Flatten()(d2)
        d3=tf.keras.layers.Dense(32, activation='tanh')(d2)
        d3 = tf.keras.layers.Dropout(0.1)(d3)
        s_v = tf.keras.layers.Dense(1, activation='linear')(d3)
        pi = tf.keras.layers.Dense(self.action_size, activation='softmax')(d3)
        critic=keras.models.Model(inputs=in1,outputs=s_v)
        actor=keras.models.Model(inputs=in1,outputs=pi)
        return critic,actor

    def make_critic(self):
        in1=tf.keras.layers.Input(shape=(7, 7, 2,))
        d1=tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(in1)
        d1=tf.keras.layers.Dropout(0.1)(d1)
        d2=tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(d1)
        d2=tf.keras.layers.Dropout(0.1)(d2)
        d2=tf.keras.layers.Flatten()(d2)
        d3=tf.keras.layers.Dense(32, activation='relu')(d2)
        d3 = tf.keras.layers.Dropout(0.1)(d3)
        s_v = tf.keras.layers.Dense(1, activation='linear')(d3)
        critic=keras.models.Model(inputs=in1,outputs=s_v)
        #critic.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.critic_lr),loss='mean_squared_error')
        return critic

    def make_actor(self):
        in1=tf.keras.layers.Input(shape=(7, 7, 2,))
        d1=tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(in1)
        d1=tf.keras.layers.Dropout(0.1)(d1)
        d2=tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(d1)
        d2=tf.keras.layers.Dropout(0.1)(d2)
        d2=tf.keras.layers.Flatten()(d2)
        d3=tf.keras.layers.Dense(32, activation='relu')(d2)
        d3 = tf.keras.layers.Dropout(0.1)(d3)
        pi = tf.keras.layers.Dense(self.action_size, activation='softmax')(d3)
        actor=keras.models.Model(inputs=in1,outputs=pi)
        return actor
    
    def get_action(self,state):
        if self.e > np.random.rand():
            a = random.choice([0, 1, 2, 3])
        else:
            state = np.transpose(state, (1, 2, 0))
            state = np.reshape(state, (1, 7, 7, 2))
            a=self.actor(state,training=False)
            val=self.critic(state,training=False)
            a=np.argmax(a)
        #print(a,np.argmax(a))
        return a,val

    def update_epsilon(self):
        self.e = self.e * 0.95
        if self.e < 0.05:
            self.e = 0.05

    def apply_grads(self,a_grads,c_grads):
    #def apply_grads(self,grads,trainables):
        grads=a_grads+c_grads
        trainables=self.actor.trainable_weights+self.critic.trainable_weights
        #print(trainables)
        self.op.apply_gradients(zip(grads,trainables))



    

class PPO():
    def __init__(self,env,num_iters,map_id):
        self.env=env
        self.map_id=map_id
        #models
        self.net=ACModel(env)
        self.old_net=ACModel(env)

        #PPO parameters
        self.num_iters=num_iters
        self.num_epochs=10
        self.batch_size=64
        self.clip_range=0.2

        #For entropy for exploration
        self.ent_coef=0.001

        self.utils=Utils()
        #self.trainables=self.net.actor.trainable_weights+[self.net.sigma]+self.net.critic.trainable_weights

    def save_models(self,iter):
        #save model
        self.net.critic.save('./weight_ppo_'+self.map_id+'/critic'+str(iter)+'.h5')
        self.net.actor.save('./weight_ppo_'+self.map_id+'/actor'+str(iter)+'.h5')

    def get_samples(self):
        #Run episode
        s = self.env.reset(self.map_id)
        reward_sum = 0
        records=[]
        steps=0
        path=[self.env.cur_loc.copy()]
        while True:
            #env.render()
            action,value = self.net.get_action(s)

            next_s, reward, done, info = self.env.step(action)
            if reward==0 and steps>20:
                reward=-1

            if reward ==-1:
                reward=-10
            reward_sum += reward

            path.append(info[0])
            record=(s,action,reward,next_s,done,value)
            records.append(record)
            steps+=1
            if done:
                break
            s = next_s
        return records,reward_sum,steps,path

    def calc_grads(self,batch):
        s=np.array([mem[0] for mem in batch])#batch state
        a=np.array([mem[1] for mem in batch])#batch action
        td=np.array([mem[2] for mem in batch])#batch td
        adv=np.array([mem[3] for mem in batch])

        with tf.GradientTape(persistent=True) as t:
            t.watch(self.net.actor.trainable_weights)
            t.watch(self.net.critic.trainable_weights)
            #t.watch(self.trainables)
            #print(self.net.get_logp(s,a))

            s = np.transpose(s, (0, 2, 3, 1))
            net_pi=self.net.actor(s)
            old_net_pi=self.old_net.actor(s)

            row_indices = tf.range(len(net_pi))
            # zip row indices with column indices
            a_indices = tf.stack([row_indices, a], axis=1)

            # retrieve values by indices
            pi = tf.gather_nd(net_pi, a_indices)
            old_pi = tf.gather_nd(old_net_pi, a_indices)
            '''print(pi[:3],a[:3])
            print(S[:3])
            exit()'''

            

            ratio=tf.exp(tf.math.log(pi)-tf.stop_gradient(tf.math.log(old_pi)))
            #print('ratio',ratio)
            print(pi[:3],old_pi[:3])
            #ratio=tf.exp(self.net.actor(s)[:,a] - tf.stop_gradient(self.old_net.actor(s)[:,a]))

            pg_loss1=adv*ratio
            #print(pg_loss1[0])
            pg_loss2=adv*tf.clip_by_value(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range)
            #batch grad
            a_loss = -tf.reduce_mean(tf.minimum(pg_loss1, pg_loss2))
            v_pred=self.net.critic(s,training=True)
            c_loss = tf.reduce_mean(tf.square(v_pred - td))

            loss=a_loss+c_loss
        a_grads=t.gradient(loss,self.net.actor.trainable_weights)
            
        c_grads=t.gradient(loss,self.net.critic.trainable_weights)
        #grads=t.gradient(loss,self.trainables)
        return a_grads,c_grads,a_loss,c_loss
        #return grads,a_loss,c_loss

    def run(self):
        avg = 0
        avg_iter=100

        #PPO Iteration
        
        cur_iter=0

        #Trajectory collection
        num_steps=10
        
        while cur_iter<self.num_iters:
            T=[]
            score_sum=0
            print("\n\nIteration",cur_iter)
            #update old net
            self.old_net.actor.set_weights(self.net.actor.get_weights())
            self.old_net.critic.set_weights(self.net.critic.get_weights())

            #get trajectories
            cur_steps=0
            cur_episode=0 #collected episodes
            while cur_steps<num_steps:
                trajectory,score,steps,path=self.get_samples()
                score_sum+=score
                cur_steps+=steps
                T.append(trajectory)
                cur_episode+=1
                print('Reward {} Steps {} Path {}'.format(score,steps,path))

            #Train
            self.train(T)
            self.net.update_epsilon()
            cur_iter+=1
            avg += score_sum/cur_episode

            print(f"run {cur_iter} total reward: {score_sum/cur_episode}")
            #save every avg_iter iterations
            if (cur_iter)%avg_iter==0:
                print(f"average {cur_iter} total reward: {avg/avg_iter}")
                avg=0
                self.save_models(cur_iter)


    def train(self,T):
        #Get values
        vals=self.utils.calc_traj_vals(T)

        for ep in range(self.num_epochs):
            #epochs sgd
            #make batch
            batch_iters=int(len(vals)/self.batch_size)
            #print('batch',batch_iters,len(vals))
            v_loss=0
            p_loss=0
            if batch_iters==0:
                a_grads,c_grads,p_loss,v_loss=self.calc_grads(vals)

                self.net.apply_grads(a_grads,c_grads)
                #self.net.apply_grads(grads,self.trainables)
            else:
                for i in range(batch_iters):
                    #run batch
                    cur_idx=i*self.batch_size
                    batch=vals[cur_idx:cur_idx+self.batch_size]

                    a_grads,c_grads,a_loss,c_loss=self.calc_grads(batch)
                    #grads,a_loss,c_loss=self.calc_grads(vals)
                    self.net.apply_grads(a_grads,c_grads)
                    #self.net.apply_grads(grads,self.trainables)
                    v_loss+=c_loss
                    p_loss+=a_loss
                v_loss/=batch_iters
                p_loss/=batch_iters

            print("vf_loss: {:.5f}, pol_loss: {:.5f}".format(v_loss, p_loss))
            



def main():
    parser = argparse.ArgumentParser(description='DQN Training for Hitman GO')
    parser.add_argument('--num_iters', type=int, default=1000000,
                    help='Number of Training Episodes')
    parser.add_argument('--min_eps', type=float, default=0.01,
                    help='Minimum Epsilon')
    parser.add_argument('--map', default='simple',
                    help='Map ID')
    args = parser.parse_args()

    env = gym.make('hitman-v4')#blue enemy
    num_iters=args.num_iters
    ppo=PPO(env,num_iters,args.map)
    if not os.path.exists('weight_ppo_'+args.map):
        os.makedirs('weight_ppo_'+args.map)
    ppo.run()

if __name__ == "__main__":
    main()
