import numpy as np
from world import World
from agents import Car, RingBuilding, CircleBuilding, Painting, Pedestrian
from geometry import Point
from network import DQN
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import *

human_controller = False
PID = True
dqn_param = True
render = False
dt = 0.3 # time steps in terms of seconds. In other words, 1/dt is the FPS.
world_width = 120 # in meters
world_height = 120
inner_building_radius = 30
num_lanes = 2
lane_marker_width = 0.5
num_of_lane_markers = 50
lane_width = 3.5
GAMMA = 0.99
TARGET_UPDATE = 10
BATCH_SIZE = 1000

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
transition = np.dtype([('s', np.float64, (6)), ('a', np.float64, (2)),
                       ('r', np.float64), ('s_', np.float64, (6))])


ACTIONS = [[0.5,0.0],[-0.5,0.],[0.,1.5],[0.,-1.5]]
nb_actions= len(ACTIONS)
def create_world():
    w = World(dt, width = world_width, height = world_height, ppm = 8) # The world is 120 meters by 120 meters. ppm is the pixels per meter.



    # Let's add some sidewalks and RectangleBuildings.
    # A Painting object is a rectangle that the vehicles cannot collide with. So we use them for the sidewalks / zebra crossings / or creating lanes.
    # A CircleBuilding or RingBuilding object is also static -- they do not move. But as opposed to Painting, they can be collided with.

    # To create a circular road, we will add a CircleBuilding and then a RingBuilding around it
    cb = CircleBuilding(Point(world_width/2, world_height/2), inner_building_radius, 'gray80')
    w.add(cb)
    rb = RingBuilding(Point(world_width/2, world_height/2), inner_building_radius + num_lanes * lane_width + (num_lanes - 1) * lane_marker_width, 1+np.sqrt((world_width/2)**2 + (world_height/2)**2), 'gray80')
    w.add(rb)

    # Let's also add some lane markers on the ground. This is just decorative. Because, why not.
    for lane_no in range(num_lanes - 1):
        lane_markers_radius = inner_building_radius + (lane_no + 1) * lane_width + (lane_no + 0.5) * lane_marker_width
        lane_marker_height = np.sqrt(2*(lane_markers_radius**2)*(1-np.cos((2*np.pi)/(2*num_of_lane_markers)))) # approximate the circle with a polygon and then use cosine theorem
        for theta in np.arange(0, 2*np.pi, 2*np.pi / num_of_lane_markers):
            dx = lane_markers_radius * np.cos(theta)
            dy = lane_markers_radius * np.sin(theta)
            w.add(Painting(Point(world_width/2 + dx, world_height/2 + dy), Point(lane_marker_width, lane_marker_height), 'white', heading = theta))
    

    # A Car object is a dynamic object -- it can move. We construct it using its center location and heading angle.
    c1 = Car(Point(91.75,60), np.pi/2)
    c1.max_speed = 30.0 # let's say the maximum is 30 m/s (108 km/h)
    c1.velocity = Point(0, 3.0)
    w.add(c1)
    if render:
        w.render() # This visualizes the world we just constructed.
    return  w,c1, cb, rb



class Env():
    def __init__(self):
        self.w,self.car,self.cb,self.rb = create_world()
        self.v = 0
        self.die = False

   

    def reset(self):
        self.av_r = self.reward_memory()
        self.w.close()
        self.w,self.car,self.cb,self.rb  = create_world()
        state = np.array([self.car.center.x,self.car.center.y,self.v,self.car.angular_velocity,self.car.distanceTo(self.cb),self.car.distanceTo(self.rb)])
        return state

    def step(self, action,t):
        total_reward = 0
        self.car.set_control(action[0], action[1])
        self.w.tick() # This ticks the world for one time step (dt second)
        if render:
            self.w.render()
        v = np.sqrt(np.square(self.car.velocity.x)+ np.square(self.car.velocity.y))
        self.v = v
        total_reward += t + v*0.1
        #total_reward += 1/(abs((self.car.distanceTo(self.cb)-self.car.distanceTo(self.rb))))
        if self.v < 0.05:
            self.die = True
            #total_reward -= 100
        if self.w.collision_exists():
            self.die = True
            #total_reward -= 100
        state = np.array([self.car.center.x,self.car.center.y,self.v,self.car.angular_velocity,self.car.distanceTo(self.cb),self.car.distanceTo(self.rb)])
        return state, total_reward

    def reward_memory(self):
        # record reward for last 100 steps
        count = 0
        length = 100
        history = np.zeros(length)

        def memory(reward):
            nonlocal count
            history[count] = reward
            count = (count + 1) % length
            return np.mean(history)

        return memory

class Agent():
    """
    Agent for training
    """
    max_grad_norm = 0.5
    clip_param = 0.1  # epsilon in clipped loss
    buffer_capacity = 1000

    def __init__(self):
        self.training_step = 0
        self.net = DQN().double().to(device)
        self.buffer = np.empty(self.buffer_capacity, dtype=transition)
        self.counter = 0
        self.eps=1
        self.ppo_epoch = 10
        self.ACTIONS = [[0.5,0.0],[-0.5,0.],[0.,1.5],[0.,-1.5]]
        self.nb_actions= len(ACTIONS)

        self.optimizer = optim.Adam(self.net.parameters(), lr=1e-3)

    def select_action(self,state):
        if random.random() < self.eps:
            action_index = np.random.choice(self.nb_actions)
            return self.ACTIONS[action_index]

        else:
            with torch.no_grad():
                state = torch.from_numpy(state).double().to(device).unsqueeze(0)
                action_index = torch.argmax(self.net(state))
                action_index = action_index.squeeze().cpu().numpy()
                return self.ACTIONS[action_index]

    def save_param(self):
        torch.save(self.net.state_dict(), 'DQN/param/ppo_net_params.pkl')

    def store(self, transition):
        self.buffer[self.counter] = transition
        self.counter += 1
        if self.counter == self.buffer_capacity:
            self.counter = 0
            return True
        else:
            return False

    def update(self):

        s = torch.tensor(self.buffer['s'], dtype=torch.double).to(device)
        a = torch.tensor(self.buffer['a'], dtype=torch.int64).to(device)
        r = torch.tensor(self.buffer['r'], dtype=torch.double).to(device).view(-1, 1)
        s_ = torch.tensor(self.buffer['s_'], dtype=torch.double).to(device)
        criterion = nn.MSELoss()
        with torch.no_grad():
            target_v = r + GAMMA *torch.max(self.net(s_))
        for _ in range(TARGET_UPDATE):
            for index in BatchSampler(SubsetRandomSampler(range(self.buffer_capacity)), BATCH_SIZE, True):
                loss = criterion(torch.sum(self.net(s[index])*a[index],dim=1), target_v[index].squeeze())
                self.optimizer.zero_grad()
                loss.backward()
                #for param in self.net.parameters():
                #    param.grad.data.clamp_(-1, 1)
                self.optimizer.step()
        self.eps = max(0.999 * self.eps, 0.05)
        del s, a, r, s_


def pid(error,previous_error):
    Kp = 0.02
    Ki = 0.07
    Kd = 0.8

    steering = Kp * error + Ki * (error + previous_error) + Kd * (error - previous_error)

    return steering



if not dqn_param:
    if not human_controller and PID:
        # Let's implement some simple policy for the car c1
        env=Env()
        desired_lane = 1
        past_lp = 0.
        for k in range(600):
            if env.die:
                break
            lp = 0.
            if env.car.distanceTo(env.cb) < desired_lane*(lane_width + lane_marker_width) + 0.2:
                lp += 0.
            elif env.car.distanceTo(env.rb) < (num_lanes - desired_lane - 1)*(lane_width + lane_marker_width) + 0.3:
                lp += 1.
            v = env.car.center - env.cb.center
            v = np.mod(np.arctan2(v.y, v.x) + np.pi/2, 2*np.pi)
            if env.car.heading < v:
                lp += 0.7
            else:
                lp += 0.
            steering= pid(lp,past_lp)
            reward= env.step([steering, 0.1])
            print(reward)
            #if np.random.rand() < lp: c1.set_control(0.2, 0.1)
            #else: c1.set_control(-0.1, 0.1)
            past_lp = lp

    else: # Let's use the keyboard input for human control
        from interactive_controllers import KeyboardController
        env= Env()
        env.car.set_control(0., 0.) # Initially, the car will have 0 steering and 0 throttle.
        controller = KeyboardController(env.w)
        for k in range(600):
            reward = env.step([controller.steering, controller.throttle])
            print(reward)
            env.car.set_control(controller.steering, controller.throttle)
            time.sleep(dt/4) # Let's watch it 4x
            if env.w.collision_exists(): # We can check if there is any collision at all.
                env.reset()
                controller = KeyboardController(env.w)
                k = 0
                print('Collision exists somewhere...')