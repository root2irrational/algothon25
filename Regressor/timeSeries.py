import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.stattools import coint

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
  with open("output.txt", "w") as f:
    for row in matrix:
      f.write(" ".join(map(str, row)) + "\n")
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


