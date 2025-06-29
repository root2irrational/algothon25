import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from timeSeries import filterCorr, plotTimeSeries, isStationary, pcf
window1 = 2
window2 = 5
window3 = 10
window4 = 20
window5 = 50
windowSysytem = 100
MAX = 10000
# stock = 2
data = np.loadtxt('prices.txt')
prc = data.T 
(nInst, nt) = prc.shape
# rt = np.ones_like(prc)
# rt[:, 1:] = (prc[:, 1:] / prc[:, :-1]) # returns

# corr = 0.8, coint < 0.1
pairs = np.array(
  [[2, 6], [2, 16], [2, 20], [2, 35], [6, 20], [7, 11], [7, 13], [9, 31], [11, 13], 
  [11, 23], [11, 39], [14, 31], [22, 29], [25, 33], [29, 42], [31, 21], [31, 27], 
  [33, 1], [33, 6], [35, 20], [43, 2], [48, 49]])

# corr = 0.8, coint < 0.1
negPairs = np.array(
  [[1, 49], [2, 22], [2, 29], [6, 22], [6, 29], [16, 22], [16, 29], [20, 22], [22, 2], 
   [22, 4], [22, 6], [22, 16], [22, 20], [22, 35], [25, 49], [29, 2], [29, 6], [29, 16], 
   [33, 49], [35, 22], [49, 1], [49, 25], [49, 33]]
)
(r,c) = pairs.shape
print(f"Number of ++pairs r {r}")
(m,n) = negPairs.shape
print(f"Number of --pairs m {m}")
# idx = int(0.8 * nt)
# train = prc[:, 0:idx]
# test = prc[:, idx:nt]

def reverseTrades(position):
  for i in range(0, 50):
    position[i] = -position[i]
  return position

def IndexMomentum(prcSoFar):
  arr = syntheticIndex(prcSoFar)
  rt = arr[1:] / arr[:-1]
  w = 20
  sum = np.sum(rt[-w:]) - w
  L = 0.25
  S = 0.25
  if (sum > L):
    return 1
  elif (-S < sum < S):
    return -1
  return 0

def longShortCointegrated(prcSoFar, pos):
  for i in range(0, r):
    a = pairs[i][0]
    b = pairs[i][1]
    prcA = prcSoFar[a][-windowSysytem:]
    prcB = prcSoFar[b][-windowSysytem:]
    beta, intercept = np.polyfit(prcA, prcB, 1)
    spread = calcSpread(prcA, prcB, beta)
    if (isStationary(spread) == True or spread[-1] <= 0):
      continue
    spreadRt = spread[1:]/spread[:-1] 
    spreadRt += -np.ones_like(spreadRt)
    systemMomtumen = np.sum(spreadRt[-windowSysytem:])
    s = 10
    shortTermSpreadRt = np.sum(spreadRt[-s:])
    sharesA = np.floor(MAX / prcSoFar[a][-1])
    sharesB = np.floor(MAX / (prcSoFar[b][-1] * beta))
    LF = 0.6
    SF = 0.4
    if (systemMomtumen < SF):
      # spread is decreasing in long run (at least down 40%)
      # pos[a] += -sharesA
      pos[b] += sharesB
    elif (systemMomtumen > LF):
      # spread is increasing in long run (at least up 60%)
      pos[a] += sharesA
      # pos[b] += -sharesB
      
  return pos

def longShortRandom(prcSoFar, pos):
  nInst, nt = prcSoFar.shape
  amt = 10000

  corr = filterCorr(prcSoFar[:,-windowSysytem:], -0.9)
  # print(corr)
  for i in range(0, nInst):
    for j in range(0, nInst):
      r = corr[i][j]
      if (pos[i] != 0 and pos[j] != 0):
        continue
      if (i != j and r != 0):
        rt = prcSoFar[i][1:] / prcSoFar[i][:-1]
        w = 50
        sum = np.sum(rt[-w:])
        # print(f"Sum is {sum}")
        s = 5
        shortTerm = np.sum(rt[-s:])
        sharesA = np.floor(amt / prcSoFar[i][-1])
        sharesB = np.floor(amt / prcSoFar[j][-1])
        cond1 = sum < w and shortTerm < s
        cond2 = sum > w and shortTerm > s
        f = 0.3
        if (cond1):
          pos[i] += -f*sharesA
          pos[j] += sharesB
        elif (cond2):
          pos[i] += sharesA
          pos[j] += -f*sharesB
          
  
  return pos

def calcSpread(stockA, stockB, beta):
  spread = stockA - beta * stockB
  return spread

def syntheticIndex(prcSoFar):
  nInst, nt = prcSoFar.shape
  arr = np.zeros(nt)
  for i in range(0, nt):
    arr[i] = np.mean(prcSoFar[:, i])
  return arr

def maCrossoverSignal(prcSoFar):

  prices = pd.Series(prcSoFar)  # your price data here
  w1 = 2
  w2 = 10
  w3 = 20
  ema1 = prices.ewm(span=w1, adjust=False).mean().iloc[-1]
  ema2 = prices.ewm(span=w2, adjust=False).mean().iloc[-1]
  ema3 = prices.ewm(span=w3, adjust=False).mean().iloc[-1]
  if (ema1 > ema2 > ema3):
    return 1
  elif(ema1 < ema2 < ema3):
    return -1
  return 0

def maCrossover(prcSoFar, pos):
  nInst, nt = prcSoFar.shape
  amt = 10
  for i in range(0, nInst):
    if (pos[i] != 0):
      continue
    prices = pd.Series(prcSoFar[i])  # your price data here
    S = maCrossoverSignal(prcSoFar[i])
    if (S == 1):
      # buy
      pos[i] += amt
    elif(S == -1):
      pos[i] += -amt
      
  return pos

def momentumSignal(prcSoFar):
  prcSoFar = pd.Series(prcSoFar)
  log_returns = np.log(prcSoFar / prcSoFar.shift(1))
  # vol_2d = log_returns.rolling(window=2).std()
  # vol_20d = log_returns.rolling(window=20).std()
  w = 10
  momentum = log_returns.rolling(window=w).sum()
  rolling_mean = momentum.rolling(window=w).mean()
  rolling_std = momentum.rolling(window=w).std()
  Z = ((momentum.iloc[-1] - rolling_mean.iloc[-1]) / rolling_std.iloc[-1])
  z = 1.66
  if (Z > z):
    return -1
  elif(Z < -z):
    return +1

  return 0

def momentumTrade(prcSoFar, pos):
  nInst, nt = prcSoFar.shape
  for i in range(0, nInst):
    if (pos[i] != 0):
      continue
    amt = 10
    S = momentumSignal(prcSoFar[i])
    if (S == 1):
      pos[i] += amt
    elif(S == -1):
      pos[i] += -amt
  # print("MOMENTUMMMM")
  return pos

def spreadSignalDivergence(spread):
  # assumes spread return today == spread return yesterday
  rt = spread[1:] / spread[:-1] # > 1 means spread increased
  w = 1
  sum = np.sum(rt[-w:])
  if (sum > w):
    return 1
  elif (sum < w):
    return -1
  return 0

def spreadTradeSignal(spread):
  recent = len(spread)
  mean = np.mean(spread[recent - window4: recent])
  sd = np.std(spread[recent - window4: recent])
  
  recentSpread = np.mean(spread[recent - window1: recent])
  Z = (recentSpread - mean) / sd
  z = 0
  if (Z > z):
    return 1
  elif(Z < -z):
    return -1
  else:
    return 0

# pairs = [[2, 20], [2, 6], [11, 23]] # <- spreads trading
def negPairsTradeSpread(prcSoFar, position):
  for i in range (0, m):
    ###############################
    c = negPairs[i][0]
    d = negPairs[i][1]
    # if (position[c] and position[d] != 0):
    #   continue
    spreadLen = windowSysytem
    prcC = prcSoFar[c][-windowSysytem:]
    prcD = prcSoFar[d][-windowSysytem:]
    beta, intercept = np.polyfit(prcC, prcD, 1)
    spread = calcSpread(prcC, prcD, -beta)
    # if (isStationary(spread) == False):
    #   continue
    S = spreadTradeSignal(spread)
    # S = -S
    amtC = 10000
    amtD = np.floor(amtC / beta)
    sharesC = np.floor(amtC / prcSoFar[c][-1])
    sharesD = np.floor(amtD / prcSoFar[d][-1])
    
    if (S == 1): # same sign
      position[c] +=  -sharesC
      position[d] += sharesD
    elif (S == -1):
      position[c] += sharesC
      position[d] += -sharesD
  # print("PAIRSSSS")
  return position


def pairsTradeSpread(prcSoFar, position):
  for i in range(0, r):
    a = pairs[i][0]
    b = pairs[i][1]
    # if (position[a] and position[b] != 0):
    #   continue
    spreadLen = windowSysytem
    prcA = prcSoFar[a][-windowSysytem:]
    prcB = prcSoFar[b][-windowSysytem:]
    beta, intercept = np.polyfit(prcA, prcB, 1)
    spread = calcSpread(prcA, prcB, beta)
    if (isStationary(spread) == False):
      continue
    S = spreadTradeSignal(spread)
    amtA = 10000
    amtB = np.floor(amtA / beta)
    sharesA = np.floor(amtA / prcSoFar[a][-1])
    sharesB = np.floor(amtB / prcSoFar[b][-1])
    
    if (S == 1):
      position[a] += -sharesA
      position[b] += sharesB
    elif (S == -1):
      position[a] += sharesA
      position[b] += -sharesB
      
  position = negPairsTradeSpread(prcSoFar, position)
  return position

def checkSpreadSeries(arr):
  for i in range (0, r):
    a = arr[i][0]
    b = arr[i][1]
    prcA = prc[a]
    prcB = prc[b]
    beta, intercept = np.polyfit(prcA, prcB, 1)
    spread = calcSpread(prcA, prcB, beta)
    print(f"Pair [{a}, {b}]")
    isStationary(spread)
    plotTimeSeries(spread, spread)
    # pcf(spread)
  return

# checkSpreadSeries(np.array([[2,6], [11,23]]))
# non stationary = [2,6], [2, 20], [11, 23], [22, 29], [29, 42 iffy], [31, 9], [33,1]

arr = syntheticIndex(prc)
plotTimeSeries(arr, arr)