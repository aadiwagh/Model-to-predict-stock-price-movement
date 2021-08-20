#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
from pandas_datareader import data as da
import matplotlib as mtplt
import matplotlib.pyplot as plt
from scipy.stats import norm


# In[7]:


def MC_Simulation(ticker, stock):
    data = pd.DataFrame()
    data[ticker] = da.DataReader(ticker, data_source = 'yahoo' , start='2020-1-1')['Adj Close']
    
    log_returns = np.log(1 + data.pct_change())
    Avg = log_returns.mean()
    Var = log_returns.var()
    drift = Avg - 0.5*Var
    SD = log_returns.std()
    
    time_intervals = 250
    iterations = 100
    
    daily_returns = np.exp( drift.values + SD.values * norm.ppf(np.random.rand(time_intervals , iterations)))
    S0 = data.iloc[-1]
    
    price_list = np.zeros_like(daily_returns)
    price_list[0] = S0
    for n in range(1, time_intervals):
        price_list[n] = price_list[n - 1] * daily_returns[n]
        
    plt.figure(figsize=(10,6))
    plt.title("Monte Carlo Simulation for " + stock)
    plt.ylabel("price")
    plt.xlabel("time(days)")
    plt.plot(price_list)
    plt.show()


# In[8]:


MC_Simulation('INFY', "INFOSYS LIMITED")


# In[ ]:




