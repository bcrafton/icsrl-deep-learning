
import argparse
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=2.5e-4)
parser.add_argument('--eps', type=float, default=1e-5)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--mini_batch_size', type=int, default=512)
parser.add_argument('--name', type=str, default="flappy")
parser.add_argument('--train', type=int, default=1)
parser.add_argument('--render', type=int, default=0)
args = parser.parse_args()

if args.gpu >= 0:
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)

import numpy as np
import tensorflow as tf
import cv2
import gym
import gym_ple
import random
import time
from collections import deque

from PPOModel import PPOModel

action_set = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
]
action_space = len(action_set)
total_episodes = int(1e4)

####################################

def returns_advantages (replay_buffer, next_value, gamma=0.99, lam=0.95):
    rewards = [rb['r'] for rb in replay_buffer]
    values = [rb['v'] for rb in replay_buffer] + [next_value]
    dones = [rb['d'] for rb in replay_buffer]

    gae = 0
    returns = np.zeros_like(rewards)
    advantages = np.zeros_like(rewards)
    for t in reversed(range(len(replay_buffer))):
        delta = rewards[t] + gamma * values[t+1] * (1-dones[t]) - values[t]
        gae = delta + gamma * lam * (1-dones[t]) * gae
        advantages[t] = gae
        returns[t] = advantages[t] + values[t]

    return returns, advantages

####################################

class FlappyBirdEnv:
    def __init__(self):
        self.env = gym.make('FlappyBird-v0')
        self.env.seed(np.random.randint(0, 100000))
        self.total_reward = 0.0
        self.total_step = 0
        self.state = None

    def reset(self):
        self.total_reward = 0.0
        self.total_step = 0
        
        frame = self.env.reset()
        frame = self._process(frame)
        self.state = deque([frame] * 4, maxlen=4)
        
        return np.stack(self.state, axis=2)
    
    def step(self, action):
        cumulated_reward = 0.0
        for a in action_set[action]:
            next_frame, reward, done, _ = self.env.step(a)
            reward = self._reward_shaping(reward)
            cumulated_reward += reward
            self.total_step += 1
            self.total_reward += reward
            if done:
                break

            if args.render:
                self.env.render(mode='human')
        
        next_frame = self._process(next_frame)
        self.state.append(next_frame)
        return np.stack(self.state, axis=2), cumulated_reward, done

    def _reward_shaping(self, reward):
        if  reward > 0.0:
            return 1.0
        elif reward < 0.0:
            return -1.0
        else:
            return 0.01

    def _process(self, state):
        output = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
        output = output[:410, :]
        output = cv2.resize(output, (84, 84))
        output = output / 255.0
        return output

####################################

sess = tf.InteractiveSession()

####################################

weights_filename = 'weights/flappy_bird/weights.npy'
weights = np.load(weights_filename, allow_pickle=True).item()

if args.train:
    model = PPOModel(sess=sess, nbatch=64, nclass=4, epsilon=0.1, decay_max=8000, lr=args.lr, eps=args.eps, weights=None, train=args.train)
else:
    model = PPOModel(sess=sess, nbatch=64, nclass=4, epsilon=0.1, decay_max=8000, lr=args.lr, eps=args.eps, weights=weights, train=args.train)

replay_buffer = []
env = FlappyBirdEnv()
state = env.reset()

####################################

sess.run(tf.initialize_all_variables())

####################################

reward_list = []
for e in range(total_episodes):

    print ("%d/%d" % (e, total_episodes), reward_list)
    reward_list = []
            
    #####################################

    replay_buffer = []
    for _ in range(args.mini_batch_size):

        action, value, nlps = model.predict(state)
        
        ################################
        
        next_state, reward, done = env.step(action)

        if done and env.total_step >= 10000:
            # wait not sure what other dude used here ...
            _, next_value, _ = model.predict(next_state)
            reward += 0.99 * next_value
        
        replay_buffer.append({'s':state, 'v': value, 'a':action, 'r':reward, 'd':done, 'n':nlps})
        state = next_state
        
        if done:
            reward_list.append(round(env.total_reward, 2))
            state = env.reset()

    # other dude just used next_value=0
    _, next_value, _ = model.predict(next_state)
    rets, advs = returns_advantages(replay_buffer, next_value)

    #####################################

    if args.train:

        states = [d['s'] for d in replay_buffer]
        rewards = rets
        advantages = advs
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
        actions = [d['a'] for d in replay_buffer]
        values = [d['v'] for d in replay_buffer]
        nlps = [d['n'] for d in replay_buffer]
        
        for _ in range(args.epochs):
            for batch in range(0, args.mini_batch_size, args.batch_size):
                a = batch
                b = batch + args.batch_size
                model.train(states[a:b], rewards[a:b], advantages[a:b], actions[a:b], values[a:b], nlps[a:b])

        model.set_weights()

        if ((e + 1) % 1000 == 0):
            print ('saving weights')
            model.save_weights(weights_filename)










