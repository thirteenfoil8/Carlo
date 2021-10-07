import numpy as np
import os
import gym
import torch
import torch.nn as nn
from PPO.network import Net
from PPO.agent import Agent, Env
from gym.wrappers.monitoring.video_recorder import VideoRecorder

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
torch.manual_seed(0)
if use_cuda:
    torch.cuda.manual_seed(0)


render=True
if __name__ == "__main__":
    agent = Agent()
    env = Env()
    env.render=True
    agent.load_param('PPO/param/ppo_net_params.pkl')
    training_records = []
    running_score = 0
    state = env.reset()
    for i_ep in range(10): # change the values if you want to test more than 1 time
        score = 0
        state = env.reset()
        env.die = False

        for t in range(1000):
            action,_ = agent.select_action(state)
            state_, reward = env.step(action* np.array([0.2, 0.1]),t)
            score += reward
            state = state_
            if env.die:
                break

        print('Ep {}\tScore: {:.2f}\t'.format(i_ep, score))


