import numpy as np
import argparse
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
import random
import os

from Agent import DQN_Agent
from Atari_Wrapper import Atari_Wrapper
from Env_Runner import Env_Runner
from Experience_Replay import Experience_Replay

device = torch.device("cuda:0")
dtype = torch.float

def make_transitions(obs, actions, rewards, dones):
    # observations are in uint8 format
    
    tuples = []

    steps = len(obs) - 1
    for t in range(steps):
        tuples.append((obs[t],
                       actions[t],
                       rewards[t],
                       obs[t+1],
                       int(not dones[t])))
        
    return tuples
    
def train(args):  

    # create folder to save networks, csv, hyperparameter
    folder_name = time.asctime(time.gmtime()).replace(" ","_").replace(":","_")
    os.mkdir(folder_name)
    
    # save the hyperparameters in a file
    f = open(f'{folder_name}/args.txt','w')
    for i in args.__dict__:
        f.write(f'{i},{args.__dict__[i]}\n')
    f.close()
    
    # arguments
    env_name = args.env
    num_stacked_frames = args.stacked_frames
    replay_memory_size = args.replay_memory_size
    min_replay_size_to_update = args.replay_size_to_update
    lr = args.lr 
    gamma = args.gamma
    minibatch_size = args.minibatch_size
    steps_rollout = args.steps_rollout
    start_eps = args.start_eps
    final_eps = args.final_eps
    final_eps_frame = args.final_eps_frame
    total_steps = args.total_steps
    target_net_update = args.target_net_update
    save_model_steps = args.save_model_steps

    # init
    raw_env = gym.make(env_name)
    env = Atari_Wrapper(raw_env, env_name, num_stacked_frames, use_add_done=args.lives)

    in_channels = num_stacked_frames
    num_actions = env.action_space.n

    eps_interval = start_eps-final_eps

    agent = DQN_Agent(in_channels, num_actions, start_eps).to(device)
    target_agent = DQN_Agent(in_channels, num_actions, start_eps).to(device)
    target_agent.load_state_dict(agent.state_dict())

    replay = Experience_Replay(replay_memory_size)
    runner = Env_Runner(env, agent, folder_name)
    optimizer = optim.Adam(agent.parameters(), lr=lr)
    huber_loss = torch.nn.SmoothL1Loss()

    num_steps = 0
    num_model_updates = 0

    start_time = time.time()
    while num_steps < total_steps:
        
        # set agent exploration | cap exploration after x timesteps to final epsilon
        new_epsilon = np.maximum(final_eps, start_eps - ( eps_interval * num_steps/final_eps_frame))
        agent.set_epsilon(new_epsilon)
        
        # get data
        obs, actions, rewards, dones = runner.run(steps_rollout)
        transitions = make_transitions(obs, actions, rewards, dones)
        replay.insert(transitions)
        
        # add
        num_steps += steps_rollout
        
        # check if update
        if num_steps < min_replay_size_to_update:
            continue
        
        # update
        for update in range(4):
            optimizer.zero_grad()
            
            minibatch = replay.get(minibatch_size)
            
            # uint8 to float32 and normalize to 0-1
            obs = (torch.stack([i[0] for i in minibatch]).to(device).to(dtype)) / 255 
            
            actions = np.stack([i[1] for i in minibatch])
            rewards = torch.tensor([i[2] for i in minibatch]).to(device)
            
            # uint8 to float32 and normalize to 0-1
            next_obs = (torch.stack([i[3] for i in minibatch]).to(device).to(dtype)) / 255
            
            dones = torch.tensor([i[4] for i in minibatch]).to(device)
            
            #  *** double dqn ***
            # prediction
            
            Qs = agent(torch.cat([obs, next_obs]))
            obs_Q, next_obs_Q = torch.split(Qs, minibatch_size ,dim=0)
            
            obs_Q = obs_Q[range(minibatch_size), actions]
            
            # target
            
            next_obs_Q_max = torch.max(next_obs_Q,1)[1].detach()
            target_Q = target_agent(next_obs)[range(minibatch_size), next_obs_Q_max].detach()
            
            target = rewards + gamma * target_Q * dones
            
            # loss
            loss = huber_loss(obs_Q, target)
            loss.backward()
            optimizer.step()
            
        num_model_updates += 1
         
        # update target network
        if num_model_updates%target_net_update == 0:
            target_agent.load_state_dict(agent.state_dict())
        
        # print time
        if num_steps%50000 < steps_rollout:
            end_time = time.time()
            print(f'*** total steps: {num_steps} | time(50K): {end_time - start_time} ***')
            start_time = time.time()
        
        # save the dqn after some time
        if num_steps%save_model_steps < steps_rollout:
            torch.save(agent,f'{folder_name}/{env_name}-{num_steps}.pt')

    env.close()
    
if __name__ == "__main__":
    
    args = argparse.ArgumentParser()
    
    # set hyperparameter
    
    args.add_argument('-lr', type=float, default=8e-5)
    args.add_argument('-env', default='PongNoFrameskip-v4')
    args.add_argument('-lives', type=bool, default=False)
    args.add_argument('-stacked_frames', type=int, default=4)
    args.add_argument('-replay_memory_size', type=int, default=250000)
    args.add_argument('-replay_size_to_update', type=int, default=25000)
    args.add_argument('-gamma', type=float, default=0.99)
    args.add_argument('-minibatch_size', type=int, default=32)
    args.add_argument('-steps_rollout', type=int, default=16)
    args.add_argument('-start_eps', type=float, default=1)
    args.add_argument('-final_eps', type=float, default=0.05)
    args.add_argument('-final_eps_frame', type=int, default=500000)
    args.add_argument('-total_steps', type=int, default=3000000)
    args.add_argument('-target_net_update', type=int, default=625)
    args.add_argument('-save_model_steps', type=int, default=1000000)
    args.add_argument('-report', type=int, default=50000)
    
    train(args.parse_args())
