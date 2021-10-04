import argparse

import numpy as np
import gym
import torch
from agent import Agent,Env
from network import DQN
vis = False

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
            if agent.store((state, action, reward, state_)):
                agent.update()
                print('update')
            score += reward
            state = state_
        running_score += score
        #print('Score: {:.2f}, Action taken: {}'.format(score, t+1))
        if i_ep % 10 == 0:
            if running_score/10 > best_score:
                best_score = running_score/10
                print('NEW BEST SCORE : {:.2f}'.format(best_score))
            print('Ep {}\tLast score: {:.2f}\tMoving average score: {:.2f}'.format(i_ep, score, running_score/10))
            running_score = 0

            agent.save_param()

