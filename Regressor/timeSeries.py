import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.ar_model import AutoReg
import pandas as pd
from scipy.stats import norm

season_trend = 5
stock = 2
data = np.loadtxt('prices.txt')
prc = data.T 
(nInst, nt) = prc.shape
rt = np.ones_like(prc)
rt[:, 1:] = (prc[:, 1:] / prc[:, :-1]) # returns
# prc = prc[stock]
# rt = rt[stock]
idx = int(np.ceil(0.8 * len(prc)))
# print(idx)
train = prc[0:idx]  # remember boxcox -> differencing
test = prc[idx:len(prc)]

def corr(prcSoFar):
  nInst, nt = prcSoFar.shape
  corr = np.zeros((nInst, nInst))
  for i in range(0, nInst):
    for j in range(0, nInst):
      if j != i:
        r = np.corrcoef(prcSoFar[i], prcSoFar[j])
        corr[i][j] = r[0, 1]
  return corr

def filterCorr(prcSoFar, strength):
  corrData = corr(prcSoFar)
  if (strength < 0):
    return np.where(corrData < strength, corrData, 0)
  elif (strength > 0):
    return np.where(corrData > strength, corrData, 0)
  return corrData
    
def writeCorrData(matrix):
    with open("allCorr.txt", "w") as f:
        n = len(matrix)
        # Write column headers
        f.write("       " + "  ".join(f"{j:>2}" for j in range(n)) + "\n")
        
        # Write each row with its label
        for i, row in enumerate(matrix):
            row_str = "  ".join(f"{val:>6.3f}" for val in row)  # format values nicely
            f.write(f"{i:>2}  {row_str}\n")
    return
  
def isStationary(series):
  result = adfuller(series)
  print("ADF Statistic:", result[0])
  print("p-value:", result[1])
  print("Critical Values:")
  for key, value in result[4].items():
    print(f"  {key}: {value}")
  if (result[1] < 0.01):
    return True
  return False

def pcf(data):
  plt.rc("figure", figsize=(11,5))
  plot_pacf(data, method='ywm')
  plt.xlabel('Lags', fontsize=18)
  plt.ylabel('Correlation', fontsize=18)
  plt.xticks(fontsize=18)
  plt.yticks(fontsize=18)
  plt.title('Partial Autocorrelation Plot', fontsize=20)
  plt.tight_layout()
  plt.show()
  return

def plotTimeSeries(series1, series2):
  x = np.arange(0, len(series1))
  if len(series2) == 0:
    series2 = np.zeros(len(series1))

  plt.plot(x, series1, label='series1')
  plt.plot(x, series2, label='series2')
  plt.legend()
  plt.title('pred vs test')
  plt.xlabel('day')
  plt.ylabel('prc')
  plt.show()
  return

def calcSpread(stockA, stockB, beta):
  spread = stockA - beta * stockB
  return spread

def sortPairsHalfLife(cointegratedPairs):
  best_pairs = {}
  for i, j, p, half_life in cointegratedPairs:
    key = tuple(sorted((i, j)))  # Treat (i, j) and (j, i) as same pair
    if key not in best_pairs or half_life < best_pairs[key][3]:
      best_pairs[key] = [i, j, p, half_life]
  # cointegratedPairs = cointegratedPairs.sort(key=lambda x: x[3])
  return list(best_pairs.values())

def sortPairsPvalue(cointegratedPairs):
  # Sort by p-value (3rd element in each [i, j, p] entry)
  best_pairs = {}
  for i, j, p, half_life in cointegratedPairs:
    key = tuple(sorted((i, j)))
    if key not in best_pairs or p < best_pairs[key][2]:
      best_pairs[key] = [i, j, p, half_life]
    
  # cointegratedPairs = sorted(cointegratedPairs, key=lambda x: x[2])
  return list(best_pairs.values())

def cointegratedPairs(prcSoFar, corr, pVal):
  cointegratedPairs = []
  seen_pairs = set()
  r,c = corr.shape
  for i in range(0, r):
    for j in range (0, c):
      if (i == j or corr[i][j] == 0):
        continue
      i, j = int(i), int(j)
      series1 = prcSoFar[i]
      series2 = prcSoFar[j]
      
      coint_t, p, _ = coint(series1, series2)
      if p < pVal:
        beta, intercept = np.polyfit(series1, series2, 1)
        if (beta < 0):
          beta = -beta
        spread = calcSpread(series1, series2, beta)
        halfLife = spreadHalfLife(spread)
        pair = tuple(sorted((i, j, p, halfLife)))
        
        if np.isfinite(halfLife):
          if pair in seen_pairs:
            continue
          seen_pairs.add(pair)
          cointegratedPairs.append([i, j, p, halfLife])
        
  return cointegratedPairs

def spreadHalfLife(spread):
  lagged = spread[:-1]
  delta = spread[1:] - spread[:-1]
  beta = sm.OLS(delta, sm.add_constant(lagged)).fit().params[1]
  halflife = -np.log(2) / beta if beta < 0 else np.inf
  return halflife

def arModel(arr, lag):
  model = AutoReg(arr, lags=lag, old_names=False)
  model_fit = model.fit()
  return model_fit

def group_correlated_stocks(corr_matrix, threshold):
    n = corr_matrix.shape[0]
    # Build adjacency list for graph: edge exists if corr > threshold
    adjacency = {i: set() for i in range(n)}
    for i in range(n):
        for j in range(i+1, n):
            if corr_matrix[i][j] > threshold:
                adjacency[i].add(j)
                adjacency[j].add(i)

    visited = set()
    groups = []

    def dfs(node, group):
        visited.add(node)
        group.append(node)
        for neighbor in adjacency[node]:
            if neighbor not in visited:
                dfs(neighbor, group)

    for stock in range(n):
        if stock not in visited:
            group = []
            dfs(stock, group)
            groups.append(sorted(group))

    return groups

# arr = 2D, each row is its own series
def plotMultipleSeries(arr):
  x = np.arange(0, len(arr[0]))
  for i in range(0, len(arr)):
    plt.plot(x, arr[i], label=f'series{i}')

  plt.legend()
  plt.title('pred vs test')
  plt.xlabel('day')
  plt.ylabel('prc')
  plt.show()
  return

def syntheticIndexGivenStocks(arr, prcSoFar):
  arr = list(arr)
  selected_prices = prcSoFar[arr, :]
  synthetic_index = np.mean(selected_prices, axis=0)

  return synthetic_index

def ema(arr, days):
  arr = pd.Series(arr)
  ema = arr.ewm(span=days, adjust=False).mean()
  return ema

def findLowCorrPairs(corrData, threshold_low, threshold_high):
    n = len(corrData)
    pairs = []

    for i in range(n):
        for j in range(i+1, n):  # ensures i < j → no duplicates or self-pairs
            corr = corrData[i][j]
            if threshold_low < corr < threshold_high:
                pairs.append((i, j))

    return pairs

def plotHistogram(prices):
  # Assuming prices is a 1D array or list
  plt.hist(prices, bins=30, density=True, edgecolor='black')
  xmin, xmax = plt.xlim()
  mu, std = norm.fit(prices)
  x = np.linspace(xmin, xmax, 100)
  p = norm.pdf(x, mu, std)
  plt.plot(x, p, 'r', linewidth=2)
  plt.title(f"Normal Fit: μ = {mu:.2f}, σ = {std:.2f}")
  plt.xlabel('Price')
  plt.ylabel('Density')
  plt.show()
  return