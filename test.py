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

data = DataReader('F', 'yahoo', '2000-01-01')

price_data = data.loc[:,data.columns != 'Volume']
print(price_data.head())

high = price_data['High'].max()
low = price_data['Low'].min()

print(high)
print(low)

#min max scale the prices
price_data = (price_data-low)/(high-low)

print(price_data.head())


scaler = MinMaxScaler()
scaler.fit(data)
scaled_data = scaler.transform(data)
scaled_data = pd.DataFrame(scaled_data, columns=data.columns)

#need to scale high low open close adj close with same values

env = gym.make(
    'stocks-v0',
    df = price_data,
    frame_bound = (1000,len(price_data)),
    window_size = 14,
    )


print("env information:")
print("> shape:", env.shape)
print("> df.shape:", env.df.shape)
print("> prices.shape:", env.prices.shape)
print("> signal_features.shape:", env.signal_features.shape)
print("> max_possible_profit:", env.max_possible_profit())
env.reset()

obs, rewards, done, info = env.step(1)
print(obs)



vec_env = make_vec_env(
    custom_anytrading.envs.StocksEnv, 
    env_kwargs = {'df':price_data, 'frame_bound':(1000,len(price_data)), 'window_size':14},
    n_envs=4)


model = PPO2(MlpPolicy, vec_env, verbose=0)
model.learn(total_timesteps=10000)


# Enjoy trained agent
obs = env.reset()
c = 0
while True:
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

plt.cla()
env.render_all()
plt.show()


