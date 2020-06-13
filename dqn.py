import gym
import hitman_gym

import tensorflow as tf
import numpy as np
import random
from collections import defaultdict

import argparse

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


class DuelingDQN:
    def __init__(self, env):
        self.env = env
        # self.state_size = self.env.observation_space[0].shape[0]  # 1 (mode) + 10
        self.state_size = (2, 7, 7) # (2, 7, 7)  # tf.transpose(state_tensor, perm=[1, 2, 0])
        self.action_size = 4

        self.hideen_size = 32
        self.learning_rate = 5e-4

        self.e = 1

        self.inputs = tf.keras.layers.Input(shape=(7, 7, 2,))

        # v (1), advantage (action space) -> aggregation -> Q

        x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(self.inputs)  # 7,7,2 -> 5,5,16
        x = tf.keras.layers.Dropout(0.1)(x)
        x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(x)  # 5,5,16 -> 3,3,16
        x = tf.keras.layers.Dropout(0.1)(x)
        x = tf.keras.layers.Flatten()(x)  # 3,3,16 -> 144
        x = tf.keras.layers.Dense(32, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.1)(x)
        x = tf.keras.layers.Dense(1 + self.action_size, activation='linear')(x)

        self.q = tf.keras.layers.Lambda(lambda x: tf.keras.backend.expand_dims(x[:, 0], -1)
                                                  + x[:, 1:] - tf.keras.backend.mean(x[:, 1:], axis=1, keepdims=True),
                                        output_shape=(self.action_size,))(x)

        self.model = tf.keras.models.Model(inputs=self.inputs, outputs=self.q)
        adam = tf.keras.optimizers.Adam(lr=self.learning_rate, epsilon=1e-08)
        self.model.compile(optimizer=adam, loss='mean_squared_error')


    def predict(self, state):
        # state를 넣어 policy에 따라 action을 반환
        if self.e > np.random.rand():
            a = random.choice([0, 1, 2, 3])
        else:
            Q = self.model.predict(state)
            a = np.argmax(Q)

        return a

    def update_epsilon(self,min_eps):
        self.e = self.e * 0.95
        if self.e < min_eps:
            self.e = min_eps

    def get_epsilon(self):
        return self.e

    def save_model(self, path):
        self.model.save(path)
        print("Model Saved")


def train_minibatch(main_network, target_network):
    if len(replay_memory) < batch_size:
        return False
    # mini batch를 받아 policy를 update
    minibatch_loss = list()
    for sample in random.sample(replay_memory, batch_size):
        state, action, reward, next_state, done = sample
        state = np.transpose(state, (1, 2, 0))
        state = np.reshape(state, (1, 7, 7, 2))
        next_state = np.transpose(next_state, (1, 2, 0))
        next_state = np.reshape(next_state, (1, 7, 7, 2))
        Q = main_network.model.predict(state)

        if done:
            Q[0, action] = reward
        else:
            Q1 = target_network.model.predict(next_state)
            Q[0, action] = reward + gamma * np.max(Q1)

        hist = main_network.model.fit(state, Q, verbose=0)
        loss = hist.history['loss'][0]
        minibatch_loss.append(loss)

    loss = sum(minibatch_loss) / len(minibatch_loss)

    return loss


def copy_network(main_network, target_network):
    target_network.model.set_weights(main_network.model.get_weights())


def replay_memory_append(replay_memory, memory, max_mem):
    replay_memory.append(memory)
    #original 10000
    if len(replay_memory) > max_mem:
        del replay_memory[0]

if __name__ == '__main__':

    # main
    parser = argparse.ArgumentParser(description='DQN Training for Hitman GO')
    parser.add_argument('--num_episodes', type=int, default=100000,
                    help='Number of Training Episodes')
    parser.add_argument('--min_eps', type=float, default=0.01,
                    help='Minimum Epsilon')
    parser.add_argument('--max_mem', type=int, default=5000,
                    help='Max Replay memory size')

    parser.add_argument('--map', default='simple',
                    help='Map ID')

    args = parser.parse_args()

    env = gym.make('hitman-v4')#blue enemy

    replay_memory = list()
    num_episodes=args.num_episodes
    batch_size = 32
    gamma = 0.95

    action_size = 4

    main_network = DuelingDQN(env)
    target_network = DuelingDQN(env)

    map_id=args.map

    if not os.path.exists('weight_'+map_id):
        os.makedirs('weight_'+map_id)

    for ep_i in range(num_episodes):
        done = False
        ep_reward = 0
        env.seed(ep_i)
        obs = env.reset(map_id)

        cnt = 0

        step_count = 0
        previous_memory = None
        round_loss = list()
        path=[env.cur_loc.copy()]
        while not done:
            obs = np.transpose(obs, (1, 2, 0))
            obs = np.reshape(obs, (1, 7, 7, 2))

            action = main_network.predict(obs)  # my

            obs, reward, done, info = env.step(action)
            if step_count>100:
                done=True
                reward=-1
            path.append(info[0])
            # 추가 리워드
            # reward = 00
            if reward == 0:
                reward = -0.05

            if reward == 1:
                reward = 10
                #print("Cong")

            if previous_memory is not None and not previous_memory[3]:
                replay_memory_append(replay_memory, [previous_memory[0], previous_memory[1], previous_memory[2], obs, previous_memory[3]],args.max_mem)

            previous_memory = [obs, action, reward, done]

            cnt += reward

            loss = train_minibatch(main_network, target_network)
            round_loss.append(loss)
            step_count += 1

        print('Episode #{} total reward: {} step: {} epsilon {} path{}: '.format(ep_i, cnt, step_count, main_network.get_epsilon(),path))
        copy_network(main_network, target_network)

        main_network.update_epsilon(args.min_eps)

        # save model
        if ep_i % 50 == 0 and ep_i != 0:
            main_network.save_model('./weight_'+map_id+'/model_ep{}.h5'.format(ep_i))
