import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

random.seed(123)
log_ret = pd.read_csv("main.csv", parse_dates = ["date"], index_col = 0)
log_ret = log_ret.dropna(axis = 1, how = "all").dropna(axis = 0, how = "all")

vol = log_ret.std()
f = 1
results = pd.DataFrame(columns = ["Size", "Return", "Volatility"])
for i in range(1000):
    randomcryptos = random.randint(1, 30)
    cryptos = random.sample(range(1,2071), randomcryptos)
    randomday = random.sample(range(1,699), 1)
    size = len(cryptos)
    mean = log_ret.iloc[randomday, cryptos].mean(axis = 1)
    std = np.sqrt(vol.iloc[cryptos].sum())/len(cryptos)
    results = results.append({"Size": size, "Return": mean[0], "Volatility": std}, ignore_index = True)

df = results.sort_values(by = "Size", ascending = False)
df2 = df.groupby('Size').mean()

plt.plot(df2["Volatility"])
plt.show()