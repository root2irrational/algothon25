import numpy as np
from scipy.stats import boxcox
from timeSeries import filterCorr, plotTimeSeries, isStationary, spreadHalfLife, pcf, calcSpread, cointegratedPairs, sortPairsHalfLife
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

pairs = np.array([[6,2],[20,2],[ 7,13],[11,13],[11,23],[29,22],[42,29],[31,9],[31,14],[33,1],[33,6],[49,48]])

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
  elif (sum < -S):
    return -1
  return 0


def syntheticIndex(prcSoFar):
  nInst, nt = prcSoFar.shape
  arr = np.zeros(nt)
  for i in range(0, nt):
    arr[i] = np.mean(prcSoFar[:, i])
  return arr

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
  z = 1.66
  if (Z > z):
    return 1
  elif(Z < -z):
    return -1
  else:
    return 0


def pairsTradeSpread(prcSoFar, position):
  # pairs = np.array(getPairs(prcSoFar[:,-windowSysytem:], 0.8, 0.05))
  r,c = pairs.shape
  # pairs = getPairsTuples(prcSoFar[:,-windowSysytem:], 0.8, 0.05)
  # (r,c, _, _) = pairs.shape
  top = 3
  # r = top
  for i in range(0, r):
    a = pairs[i][0]
    b = pairs[i][1]
    if (position[a] != 0 or position[b] != 0):
      continue
    spreadLen = windowSysytem
    prcA = prcSoFar[a][-windowSysytem:]
    prcB = prcSoFar[b][-windowSysytem:]
    beta, intercept = np.polyfit(prcA, prcB, 1)
    spread = calcSpread(prcA, prcB, beta)
    halfLife = spreadHalfLife(spread)
    # if (isStationary(spread) == False):
    #   continue
    spreadRt = spread[1:] / spread[:-1]
    spreadRt -= np.ones_like(spreadRt)
    w = 20
    mom = np.sum(spreadRt[-w:])
    S = 0
    F = 0.8
    # halfLife > 5
    if (abs(mom) > F * w):
      if (mom < -F * w):
        S = -1
      elif (mom > F * w):
        S = 1
      # S = -S
    else:
      S = spreadTradeSignal(spread)
      S = -S
    amtA = 10000
    amtB = np.floor(amtA / beta)
    sharesA = np.floor(amtA / prcSoFar[a][-1])
    sharesB = np.floor(amtB / prcSoFar[b][-1])
    # S = -S
    if (S == 1):
      position[a] += -sharesA
      position[b] += sharesB
    elif (S == -1):
      position[a] += sharesA
      position[b] += -sharesB
      
  # position = negPairsTradeSpread(prcSoFar, position)
  return position

def checkSpreadSeries(arr):
  r,c = pairs.shape
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


arr = syntheticIndex(prc)
arr = arr[2:] - arr[:-2] # lags 1,2,3

isStationary(arr)
x = spreadHalfLife(arr)
print(x)
plotTimeSeries(arr, arr)
pcf(arr)
