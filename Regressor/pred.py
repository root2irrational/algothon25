import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from findPairs import log_returns
from findPairs import data, nInst, nt

stock = 2
best_stock = -1

log_r = np.copy(log_returns[stock])   # <-- Replace with your 1D log return array
# log_r_sd = np.std(log_r)
predicted = np.zeros((nt,1))
prices = np.copy(data[stock])
season_trend = 150
start, end = season_trend + 1, nt
for i in range(0, season_trend + 1):
  predicted[i] = -1


# Parameters
def predError(stock):
  log_r = np.copy(log_returns[stock])
  prices = np.copy(data[stock])
  for i in range(season_trend + 1, nt):
    predicted[i] = prices[i - 1] * np.mean(np.power(np.exp(log_r[i - season_trend:i]), 1/2))
  percentage_error = np.abs((predicted[start:end] - prices[start:end]) / prices[start:end]) * 100
  return np.mean(percentage_error)

stocksRankByMAPE = []

def run():
  min = 100
  best_stock = -1
  min = 100
  for i in range(0, nInst):
    err = predError(i)
    stocksRankByMAPE.append((i, err))
    if (err < min):
      min = err
      best_stock = i
  print("Best stock is " + str(best_stock))
  return
run()
stocksRankByMAPE = sorted(stocksRankByMAPE, key=lambda x: x[1])
print(stocksRankByMAPE)

best_stock = 2
prices = np.copy(data[best_stock])

# start, end = season_trend + 1, nt
print("\nFor stock "  + str(best_stock) + ":\n")
print("MAPE:", predError(best_stock), "%") #Mean Absolute Percentage Error
mse = np.mean((predicted[start:end] - prices[start:end])**2)
print("MSE:", mse)


end = start+100
plt.plot(predicted[start:end], label=' Predicted')
plt.plot(prices[start:end], label='Actual')
plt.legend()
plt.title('Price vs day')
plt.xlabel('day')
plt.ylabel('Price')
plt.show()
