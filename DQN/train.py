import argparse

import numpy as np
import gym
import torch
from DQN.agent import Agent,Env
from DQN.network import DQN
vis = False
ACTIONS = [[0,0],[0.5,0.0],[-0.5,0.],[0.,1.5],[0.,-1.5]]

if __name__ == "__main__":
    agent = Agent()
    env = Env()

    training_records = []
    running_score = 0
    best_score = 0
    state = env.reset()
    for i_ep in range(100000):
        env.die = False
        score = 0
        state = env.reset()

        for t in range(1000):
            action = agent.select_action(state)
            state_, reward = env.step(action,t)
            if env.die :
                score += reward
                break
            if agent.store((state, ACTIONS.index(action), reward, state_)):
                agent.update()
                print('update')
                print(agent.eps)
            score += reward
            state = state_
        running_score = running_score * 0.99 + score * 0.01
        #print('Score: {:.2f}, Action taken: {}'.format(score, t+1))
        if i_ep % 10 == 0:

            print('Ep {}\tLast score: {:.2f}\tMoving average score: {:.2f}'.format(i_ep, score, running_score))
            agent.save_param()

            agent.save_param()

