import gym
import random
from gym import spaces
from gym.utils import seeding
import numpy as np
from enum import Enum
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler


# based off of https://towardsdatascience.com/creating-a-custom-openai-gym-environment-for-stock-trading-be532be3910e
class CustomStockEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self, stock_df, pred_df, window_size, initial_balance, min_percent_loss):
        # dataframe of stock timeline, the data needed to run the sim
        self.df = stock_df
        self.pred_df = pred_df

        self.scale_df = stock_df.copy()
        self.scale_pred_df = pred_df.copy()

        self._scale_df()

        self.initial_balance = initial_balance
        self.min_percent_loss = min_percent_loss

        self.min_balance = initial_balance*(1-min_percent_loss)
        print('min balance')
        print(self.min_balance)

        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance

        #start with no shares 
        self.shares_held = 0
        #cost basis - original value of asset, purchase price adjusted by stock splits, dividends, and return of capital distributions
        self.cost_basis = 0
        self.total_shares_sold = 0
        self.total_sales_value = 0

        self.window_size = window_size
        self.shape = (window_size, self.df.shape[1])

        self.current_step = self.window_size
    
        # action space low value and high value
        self.action_space = spaces.Box(low=np.array([0, 0, 0, 0]), high=np.array([1, 1, 1, 1]), dtype=np.float16) 
        # observation space is 2 arrays size 6
        # first array holds stock timeline info, last 5 days of opening, high, low, closing, and volume of stock
        # second array is info of the agent: balance, stocks held, net worth etc.
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.shape, dtype=np.float32)

        
        #self.reset()

    def _scale_df(self):
            price_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close']
            high = self.scale_df['High'].max()
            low = self.scale_df['Adj Close'].min()
            diff = high - low

            self.scale_df[price_columns] = self.scale_df[price_columns].applymap(lambda x: ((x-low)/diff))
            self.scale_pred_df[price_columns] = self.scale_pred_df[price_columns].applymap(lambda x: ((x-low)/diff))

            scaler = MinMaxScaler()
            self.scale_df['Volume'] = scaler.fit_transform(self.scale_df['Volume'].to_numpy().reshape(-1, 1))
            self.scale_pred_df['Volume'] = scaler.transform(self.scale_pred_df['Volume'].to_numpy().reshape(-1, 1))
        
    def reset(self):
        # Reset the state of the environment to an initial state
        # set these as the same to start
        self.balance = self.initial_balance # available money to buy stock
        self.net_worth = self.initial_balance # balance + stock equity value
        self.max_net_worth = self.initial_balance # greatest net worth

        #start with no shares 
        self.shares_held = 0
        #cost basis - original value of asset, purchase price adjusted by stock splits, dividends, and return of capital distributions
        self.cost_basis = 0
        self.total_shares_sold = 0
        self.total_sales_value = 0

        # Set the current step to a random point within the data frame to get different 'experiences' per reset
        self.current_step = random.randint(self.window_size, int(self.df.shape[0]/3)) 
        #print('reset on ' + self.df.loc[self.current_step, 'Date'])
        return self.next_observation()

    
    def next_observation(self):

        window_offset = self.window_size//2

        stock_info = self.df.iloc[(self.current_step-window_offset):self.current_step, :]
        pred_info = self.pred_df.iloc[self.current_step:(self.current_step+window_offset), :]

        scale_stock_info = self.scale_df.iloc[(self.current_step-window_offset):self.current_step, :]
        scale_pred_info = self.scale_pred_df.iloc[self.current_step:(self.current_step+window_offset), :]

        obs = stock_info.append(pred_info, ignore_index=True)
        scale_obs = scale_stock_info.append(scale_pred_info, ignore_index=True)

        # Append additional data and scale each value to between 0-1
        # obs = np.append(frame, [[
        #     self.balance / MAX_ACCOUNT_BALANCE,
        #     self.max_net_worth / MAX_ACCOUNT_BALANCE,
        #     self.shares_held / MAX_NUM_SHARES,
        #     self.cost_basis / MAX_SHARE_PRICE,
        #     self.total_shares_sold / MAX_NUM_SHARES,
        #     self.total_sales_value / (MAX_NUM_SHARES * MAX_SHARE_PRICE),
        # ]], axis=0)

        return scale_obs
    

    def take_action(self, action):
        # Set the current price to a random price within the time step
        # this mimics day by day variance
        # current price of the stock 

        current_date = self.df.index[self.current_step]
        current_price = random.uniform(
            self.df.loc[current_date, "Open"],
            self.df.loc[current_date, "Close"]) 

        # first three parts of action is type of action
        # taking the max between action[0:2] 
        action_type = np.argmax(action[0:2])
        # second part is the amount of stocks to sell/buy in percent (0,1) of possible total stocks you can buy at the time
        # ex. if action[3] = .5 and buying, buy 50% of the possible total stocks you can buy with current balance
        # ex. if action[3] = .5 and selling, sell 50% of your held stocks
        # clip the value to be between [0,1]
        amount = np.clip(action[3], 0, 1)  

        if action_type == 0:
            # Buy amount % of balance in shares
            #total possible amount of shares you can buy
            total_possible = int(self.balance / current_price)
            # number of shares bought in step
            shares_bought = int(total_possible * amount)

            # calculate cost to buy the shares, and buy them by subtracting the cost from your balance
            additional_cost = shares_bought * current_price
            self.balance -= additional_cost
            
            #calculate cost basis  
            prev_cost = self.cost_basis * self.shares_held
            self.cost_basis = (prev_cost + additional_cost) / (self.shares_held + shares_bought)

            # update the number of shares hold
            self.shares_held += shares_bought  

        elif action_type == 1:
            # Sell amount % of shares held
            shares_sold = int(self.shares_held * amount)
            
            # sell shares and add to balance
            self.balance += shares_sold * current_price
            self.shares_held -= shares_sold

            # update some logging variables
            self.total_shares_sold += shares_sold
            self.total_sales_value += shares_sold * current_price  

        # calcuate net worth, which is balance(cash) + shares
        self.net_worth = self.balance + self.shares_held * current_price 

        # update max_net_worth and cost basis
        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth  
        if self.shares_held == 0:
            self.cost_basis = 0


    def step(self, action):
        # Execute one time step within the environment
        self.take_action(action)  
        self.current_step += 1  

        obs = self.next_observation()  

        # delay the reward based on the number of steps taken
        # so early on in timeline, rewards are small  
        # this can probably be changed to something more interesting
        delay_modifier = (self.current_step / self.df.shape[0])

        # check if you have lost more than half your starting money
        done = self.net_worth < self.min_balance

        if (done):
            print('im done')
            reward = -100
            return obs, reward, done, {}
        else:
            if(self.current_step >= self.df.shape[0]-self.window_size):
                done = True
                self.current_step = self.window_size
        
        reward = (self.net_worth/self.initial_balance) + delay_modifier
        return obs, reward, done, {}

        # # check if there are still possible days in the stock timeline
        # # if theere are no more days, reset to the begining of the timeline
        # if(self.current_step >= self.df.shape[0]-self.window_size):
        #     done = 
        #     self.current_step = self.window_size

        # # reward for current action is the net worth with the delay
        # # reward = self.net_worth * delay_modifier
        # if not (done): # positive reward if didn't lose money
        #   reward = (self.net_worth/self.initial_balance) + delay_modifier
        # else:
        #   reward = -10



        # obs = self.next_observation()  
        # return obs, reward, done, {}

            
    def render(self, mode='human', close=False):
        # Render the environment to the screen
        # profit is the total increase of net worth from the start
        profit = self.net_worth - self.initial_balance

        print(f'Step: {self.current_step}')
        print(f'initial_balance: {self.initial_balance}')
        print(f'min_balance: {self.min_balance}')
        print(f'Balance: {self.balance}')
        print(
            f'Shares held: {self.shares_held} (Total sold: {self.total_shares_sold})')
        print(
            f'Avg cost for held shares: {self.cost_basis} (Total sales value: {self.total_sales_value})')
        print(
            f'Net worth: {self.net_worth} (Max net worth: {self.max_net_worth})')
        print(f'Profit: {profit}')

    def get_profit(self):
        return self.net_worth - self.initial_balance

