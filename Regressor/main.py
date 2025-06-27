
import numpy as np
import matplotlib.pyplot as plt
#from trade import predictReturnNextDay, predictPriceNextDay
# predictAll, daysTraded,
##### TODO #########################################
### IMPLEMENT 'getMyPosition' FUNCTION #############
### TO RUN, RUN 'eval.py' ##########################

nInst = 50
currentPos = np.zeros(nInst)
season_trend = 5
day_avg1 = 2
day_avg2 = 5
day_avg3 = 10
day_avg4 = 20
stock = 2
commRate = 0.0005

def predictReturnNextDay(prcSofar):
  data = prcSofar
  mostRecent = len(prcSofar[stock]) - 1
  returns = np.ones_like(data)  # initialize with 1s
  returns[:, 1:] = (data[:, 1:] / data[:, :-1])
  global predictedReturn
  predictedReturn = np.ones(nInst)
  for i in range(0, nInst):
    predictedReturn[i] = np.average(returns[i][mostRecent - season_trend:mostRecent + 1])
    #predictedReturn[i] = np.average(returns[i][mostRecent - season_trend:mostRecent], predictedReturn[i])
  return predictedReturn

def predictPriceNextDay(prcSofar):
  data = prcSofar
  mostRecent = len(prcSofar[stock]) - 1
  #global predicted
  predicted = np.zeros(nInst)
  predictedReturn = np.zeros(nInst)
  predictedReturn = predictReturnNextDay(prcSofar)
  for i in range(0, nInst):
    predicted[i] = data[i][mostRecent] * predictedReturn[i]
  return predicted

# Co-integrated pairs {(2, 20), (11, 23), (2, 6)}
def getMyPosition(prcSoFar):
    global currentPos
    (nins, nt) = prcSoFar.shape
    currentPos = np.zeros(nins)
    if (nt <= season_trend):
        return currentPos
    
    mostRecent = len(prcSoFar[stock]) - 1
    returns = np.ones_like(prcSoFar)  # initialize with 1s
    returns[:, 1:] = (prcSoFar[:, 1:] / prcSoFar[:, :-1])
    predictedReturn = predictReturnNextDay(prcSoFar)
    pricePredicted = predictPriceNextDay(prcSoFar)
    # print("Predicted return: " + str(predictedReturn[2]) + " Actual return: " + str(returns[2][mostRecent]))
    # print("Predicted Price next: " + str(pricePredicted[stock]) + " Actual Price: " + str(prcSoFar[stock][mostRecent]))
    p = '+++'
    a = '+++'
    if (predictedReturn[stock] < 1):
        p = '---'
    if (returns[stock][mostRecent] < 1):
        a = '---'
    # print("Pred RETURN:" + str(predictedReturn[stock]) + p + "Act RETURN: " + str(returns[stock][mostRecent]) + a)
    
    currentPos = decision(prcSoFar, currentPos, returns, predictedReturn)
    return currentPos

def zscore(arr):
    return (arr - np.mean(arr)) / np.std(arr)

tradeStart = 0 / 552
def decision(prcSoFar, currentPos, returns, predictedReturn):
    nInst, nt = prcSoFar.shape
    stock = range(0, nInst)
    # stock = [2, 22, 6, 11, 23]
    mostRecent = nt
    days = mostRecent - tradeStart - season_trend
    mostRecent -= 1
    if (days <= 20):
        return currentPos
    r = predictedReturn
    for i in stock:
        series = prcSoFar[:, 1:] - prcSoFar[:, :-1]
        # zeros = np.zeros((prcSoFar.shape[0], 1))
        # series = np.hstack([zeros, series])
        window = 200
        mean = np.mean(series[i][-window:])
        f = 2.576
        dev = np.std(series[i])
        # if (mean - f*dev < series[i][-1] < mean):
        #     currentPos[i] =  +10
        # elif(mean + f*dev > series[i][-1] > mean):
        #     currentPos[i] =  -10
        # f = 3
        MAX = 10000
        NORM = 100
        if (mean >= 0):
            
            # if (0 < series[i][-1] < mean and mean < series[i][-2] < mean + 1.66*dev):
            #     currentPos[i] =  np.ceil(MAX / prcSoFar[i][-1])
            # elif(0 < series[i][-1] < mean and series[i][-2] > mean + 1.66*dev):
            #     currentPos[i] =  np.ceil(NORM / prcSoFar[i][-1])
            
            # if (0 < series[i][-2] < mean and mean < series[i][-3] < mean + 1.66*dev):
            #     currentPos[i] = - np.ceil(MAX / prcSoFar[i][-2]) * returns[i][-1]
            # elif(0 < series[i][-2] < mean and series[i][-3] > mean + 1.66*dev):
            #     currentPos[i] =  -np.ceil(NORM / prcSoFar[i][-1]) * returns[i][-1]
            z = (series[i][-1] - mean) / dev
            # if (z > f):
            #     currentPos[i] =  -np.ceil(MAX / prcSoFar[i][-1])
            if(days % 2 == 0 and z < -f):
                currentPos[i] =  +np.ceil(MAX / prcSoFar[i][-1])
            elif (days % 2 == 1 and z > f):
                z = (series[i][-2] - mean) / dev
                if (z < -f):
                    currentPos[i] = -np.ceil(MAX / prcSoFar[i][-1])
    return currentPos
