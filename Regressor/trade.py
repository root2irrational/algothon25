import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg, ar_select_order
from Regressor.model import plotTimeSeries, isStationary, pcf

season_trend = 5
stock = 2
data = np.loadtxt('prices.txt')
prc = data.T 
(nInst, nt) = prc.shape
rt = np.ones_like(prc)
rt[:, 1:] = (prc[:, 1:] / prc[:, :-1]) # returns

A = 2
B = 20
prcA = prc[A]
prcB = prc[B]
beta, intercept = np.polyfit(prcB, prcA, 1)
print(f"Beta is {beta}")
spread = prcA - beta * prcB # pcf highest for 1
idx = int(np.ceil(0.8 * len(spread)))
train = spread[0:idx]
test = spread[idx:len(spread)]



def model(series, lags):
  selector = ar_select_order(series, lags)
  model = AutoReg(series, lags=selector.ar_lags).fit()
  against = test
  pred = list(model.forecast(steps=len(against)))
  return pred

isStationary(train)
plotTimeSeries(train, train)
# pcf(train)
# model = model(train, 10) # <-- AR model is super ass for spread as well, only gives general trend
# plotTimeSeries(model, test)