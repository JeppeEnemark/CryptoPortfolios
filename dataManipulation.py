import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("crypto-markets.csv", parse_dates = ["date"], index_col = 3)
df = df[['name', 'close']]
df = df['2017-01-01':]

df = df.pivot_table(index = df.index, columns = 'name', values = 'close')
log_ret = np.log(df/df.shift(1))
log_ret = log_ret.dropna(axis = 1, how = 'all').dropna(axis = 0, how = 'all')

log_ret.to_csv("main.csv")