import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from scipy.stats import boxcox
from scipy.special import inv_boxcox
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.ar_model import AutoReg, ar_select_order

season_trend = 5
stock = 2
data = np.loadtxt('prices.txt')
prc = data.T 
(nInst, nt) = prc.shape
rt = np.ones_like(prc)
rt[:, 1:] = (prc[:, 1:] / prc[:, :-1]) # returns
prc = prc[stock]
rt = rt[stock]
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
    

def isStationary(series):
  result = adfuller(series)
  # print("ADF Statistic:", result[0])
  # print("p-value:", result[1])
  # print("Critical Values:")
  # for key, value in result[4].items():
  #   print(f"  {key}: {value}")
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

def convertForecastFromDiff(transformed_forecasts):
  boxcox_forecasts = []
  for idx in range(len(test)):
    if idx == 0:
      boxcox_forecast = transformed_forecasts[idx] + trainBoxcox[-1]
    else:
      boxcox_forecast = transformed_forecasts[idx] + boxcox_forecasts[idx-1]

    boxcox_forecasts.append(boxcox_forecast)

  forecasts = inv_boxcox(boxcox_forecasts, lambda_mle)
  return forecasts

def arModel(series, lags, test):
  selector = ar_select_order(series, lags)
  model = AutoReg(series, lags=selector.ar_lags).fit()
  against = test
  transformed_forecasts = list(model.forecast(steps=len(against)))
  pred = transformed_forecasts
  last_price = train[-1]
  pred = last_price * np.exp(np.cumsum(transformed_forecasts))
  return pred

