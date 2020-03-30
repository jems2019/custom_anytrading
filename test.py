import json
import datetime as dt
import random 

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from pandas_datareader.data import DataReader


import gym
import custom_anytrading

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import PPO2

from sklearn.preprocessing import MinMaxScaler

print('loading data')
data = DataReader('AAPL', 'yahoo', start='2000-01-01', end='2019-01-01')

print(data.head)

test_data = data.tail(200)
train_data = data.head(-500)

env = gym.make(
    'custom_stocks-v0',
    stock_df = train_data,
    pred_df = train_data,
    window_size = 14,
    initial_balance = 5000,
    min_percent_loss = .25,
    with_pred=False
    )

test_env = gym.make(
    'custom_stocks-v0',
    stock_df = test_data,
    pred_df = test_data,
    window_size = 14,
    initial_balance = 5000,
    min_percent_loss = .25,
    with_pred=False,
    test_env=True,
    train_df=train_data
    )

print("env information:")
print("> shape:", env.shape)
print("> df.shape:", env.df.shape)
env.render()
env.reset()

temp_action = [1,0,0,.3]

obs, rewards, done, info = env.step(temp_action)
print(obs)


env.reset()
model = PPO2(MlpPolicy, env, verbose=0)
model.learn(total_timesteps=1000)

print('done')

profits = []
sims = 10

for i in range(sims):
    obs = test_env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, done, info = test_env.step(action)
        if done:
            if(i%10 == 0):
                print('finished sim %d/%d'%(i,sims))
            profits.append(info['profit'])
            break

print(profits)

pos_count = len(list(filter(lambda x: (x >= 0), profits))) 
print('made profit - ' + str(pos_count/len(profits)))

test_env.render()



