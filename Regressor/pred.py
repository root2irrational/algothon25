import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from findPairs import log_returns
from findPairs import data, nInst, nt

stock = 2
best_stock = -1

log_r = np.copy(log_returns[stock])
predicted = np.zeros((nt,1))
prices = np.copy(data[stock])
season_trend = 5
start, end = season_trend + 1, nt
for i in range(0, season_trend + 2):
  predicted[i] = -1


# Parameters
def predError(stock):
  log_r = np.copy(log_returns[stock])
  prices = np.copy(data[stock])
  for i in range(start, nt):
    window = log_r[i - season_trend:i - 1]
    weights = np.linspace(1, 2, season_trend - 1)
    weighted_avg = np.average(window, weights=weights)
    predicted[i] = prices[i - 1] * np.exp(weighted_avg / 10) ###
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

best_stock = 6
prices = np.copy(data[best_stock])

print("\nFor stock "  + str(best_stock) + ":\n")
print("MAPE:", predError(best_stock), "%") #Mean Absolute Percentage Error
mse = np.mean((predicted[start:end] - prices[start:end])**2)
print("MSE:", mse)

start += 100
end = start+250
for i in range(290, 301):
  print("Pred: " + str(predicted[i + 1]) + " Actual: " + str(prices[i]))

x = np.arange(start, end)
plt.plot(x, predicted[start + 1:end + 1], label=' Predicted') 
## predicted price is lagging 1 day ahead of actual so to check visually i did + 1
plt.plot(x, prices[start:end], label='Actual')
plt.legend()
plt.title('Price vs day')
plt.xlabel('day')
plt.ylabel('Price')
plt.show()
