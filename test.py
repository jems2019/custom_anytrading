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
data = DataReader('F', 'yahoo', '2000-01-01')


def scale_stock_data(df):
    price_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close']
    high = df['High'].max()
    low = df['Adj Close'].min()
    diff = high - low

    df[price_columns] = df[price_columns].applymap(lambda x: ((x-low)/diff))

    scaler = MinMaxScaler()
    df['Volume'] = scaler.fit_transform(df['Volume'].to_numpy().reshape(-1, 1))

    return df

print(data.head)

#need to scale high low open close adj close with same values

env = gym.make(
    'custom_stocks-v0',
    stock_df = data,
    pred_df = data,
    window_size = 14,
    initial_balance = 5000,
    min_percent_loss = 1,
    )


print("env information:")
print("> shape:", env.shape)
print("> df.shape:", env.df.shape)
env.render()
env.reset()

temp_action = [1,0,0,.3]

obs, rewards, done, info = env.step(temp_action)
print(obs)



# vec_env = make_vec_env(
#     custom_anytrading.envs.StocksEnv, 
#     env_kwargs = {'df':price_data, 'frame_bound':(1000,len(price_data)), 'window_size':14},
#     n_envs=4)


model = PPO2(MlpPolicy, env, verbose=0)
model.learn(total_timesteps=100000)

print('done')

# Enjoy trained agent
obs = env.reset()
c = 0
while c<260:
    # print('my obs')
    # print(obs)
    action, _states = model.predict(obs)
    # print('resulting action')
    # print(action)
    obs, rewards, done, info = env.step(action)
    # print('stepping')
    # print(rewards)
    c += 1
    if done:
        print("info:", info)
        break

print(c)

env.render()


