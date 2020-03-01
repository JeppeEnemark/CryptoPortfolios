import numpy as np
import pandas as pd
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import random

#############################################################################
#                               \Data manipulation/                         #
#       \ If you have already done this the dont run this next line of code/#
#############################################################################


# This is the first data frame created that only has the close and date as index.
# Here I also twisted the table, so now all the crypto currencies are the column
# headers. 
'''
df2 = df.loc['2017-01-01':].dropna()
df2 = df2[['name', 'close']]
df3 = df2.pivot_table(values = 'close', index = df2.index, columns = 'name')
df3.to_csv("data.csv")
df = pd.read_csv("data.csv", parse_dates=['date'], index_col=0)
'''
# This here below represents the workable dataset that we should work with.
# It is the cumulative sum of the first data frame, log transformed and shifted 
# by a whole year. What this in effect means is that the first non-NAN in a 
# give column represents that bitcoins cumulative return summed up over year. 
# Next I drop all columns that dont have ANY values in them, so the dataset is
# very small now, 333x880 if I remember correctly. 

'''
df = pd.read_csv("data.csv", parse_dates = ["date"], index_col = 0)
df4 = np.cumsum(df.pct_change().shift(364)).dropna(axis = 1, how = 'all')
df5 = df4.dropna(axis = 0, how = 'all')
df5.to_csv("cumsum.csv")
print(df5.head())
'''
#################################################################################
#################################################################################
#                           \ Code that seems to work/                          #
#              \Please run it and see if it makes any sense to you/             #
#################################################################################
#################################################################################

df = pd.read_csv("main.csv", parse_dates = ["date"], index_col = 0)

y = 1
aveReturn = []
aveVolatility = pd.DataFrame(columns = ['Size', 'Volatility'])

while y < 1000: 
    randomcryptos = random.randint(1,100)
    cryptos = random.sample(range(1,285), randomcryptos)
    randomday = random.sample(range(1, 699), 1)
    portfolioSize = len(cryptos)
    #weights = random.random(10)
    data = df
    return_data = data.iloc[:,cryptos].dropna(axis = 1, how = 'all')
    returns = return_data.iloc[randomday].sum()
    std_data = returns.std()
    volatility = std_data.sum()

    aveReturn.append(returns)
    aveVolatility = aveVolatility.append({'Size': portfolioSize, 'Volatility': volatility}, ignore_index = True)
    y +=1

print(aveVolatility.head())

#################################################
#################################################
#       \THIS CODE SHOULD NOT BE CONSIDERED/    #
#################################################
#################################################

'''
stocks = random.sample(range(0, 2071), 5)

# Randomly generated dataset
data = df.iloc[:,stocks]

#log returns
returns = data.pct_change()

#mean daily return and covariance
mean_daily_returns = returns.mean()
cov_matrix = returns.cov()

#number of runs of random portfolio weights
num_portfolios = 1000

#set up array to hold results
results = np.zeros((3,num_portfolios))

TODO: This works but I need

1. Print the returns and volatility for each iteration
2. Put this into a while loop so that it runs 1000x for 20 down to 1 asset portfolio



for i in range(num_portfolios):
    #select random weights for portfolio holdings
    weights = np.random.random(5)
    #rebalance weights to sum to 1
    weights /= np.sum(weights)
    
    #calculate portfolio return and volatility
    portfolio_return = np.sum(mean_daily_returns * weights) * 252
    portfolio_std_dev = np.sqrt(np.dot(weights.T,np.dot(cov_matrix, weights))) * np.sqrt(252)
    
    #store results in results array
    results[0,i] = portfolio_return
    results[1,i] = portfolio_std_dev
    #store Sharpe Ratio (return / volatility) - risk free rate element excluded for simplicity
    results[2,i] = results[0,i] / results[1,i]

#convert results array to Pandas DataFrame
results_frame = pd.DataFrame(results.T,columns=['Return','Volatility','SharpeRatio'])
results_frame = results_frame.dropna(axis = 1, how = 'all')
print(results_frame.head())
print(data.columns)
print(results_frame.info())
print(results_frame.shape)
#create scatter plot coloured by Sharpe Ratio


plt.scatter(results_frame.Volatility,results_frame.Return,c=results_frame.SharpeRatio,cmap='RdYlBu')
plt.colorbar()
plt.show()
'''