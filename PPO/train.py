import argparse

import numpy as np
import gym
import torch
from agent import Agent,Env
from network import Net

if __name__ == "__main__":
    agent = Agent()
    env = Env()

    training_records = []
    running_score = 0
    best_score = 0
    state = env.reset()
    for i_ep in range(100000):
        env.die = False
        env.render =False
        score = 0
        state = env.reset()
        if i_ep % 100 == 0:
                env.render= False

        for t in range(10000):
            action, a_logp = agent.select_action(state)
            state_ , reward = env.step(action* np.array([0.2, 0.1]) + np.array([-0.15,0.]),t)
            if env.die :
                score += reward
                break
            if agent.store((state, action, a_logp, reward, state_)):
                agent.update()
                print('update')
            
            score += reward
            state = state_
        running_score = running_score * 0.99 + score * 0.01
        #print('Score: {:.2f}, Action taken: {}'.format(score, t+1))
        if i_ep % 10 == 0:

            print('Ep {}\tLast score: {:.2f}\tMoving average score: {:.2f}'.format(i_ep, score, running_score))
            agent.save_param()

            agent.save_param()

