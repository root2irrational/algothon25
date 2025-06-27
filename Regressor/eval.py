#!/usr/bin/env python

import numpy as np
import pandas as pd
# from Regressor.main import getMyPosition as getPosition
from main import getMyPosition as getPosition
from main import predictPriceNextDay, predictReturnNextDay
nInst = 50
nt = 0
commRate = 0.0005
dlrPosLimit = 10000

def loadPrices(fn):
    global nt, nInst
    df=pd.read_csv(fn, sep='\s+', header=None, index_col=None)
    (nt,nInst) = df.shape
    return (df.values).T

pricesFile="./priceSlice_test.txt"
prcAll = loadPrices(pricesFile)
print ("Loaded %d instruments for %d days" % (nInst, nt))

def calcPL(prcHist, numTestDays):
    cash = 0
    curPos = np.zeros(nInst)
    totDVolume = 0
    totDVolumeSignal = 0
    totDVolumeRandom = 0
    value = 0
    todayPLL = []
    (_,nt) = prcHist.shape
    startDay = nt + 1 - numTestDays
    predictAllDays = [] ###
    predictReturn = []
    actualReturns = []
    actualPrices = [] ###
    for t in range(startDay, nt+1):
        prcHistSoFar = prcHist[:,:t]
        curPrices = prcHistSoFar[:,-1]
        if (t < nt):
            # Trading, do not do it on the very last day of the test
            newPosOrig = getPosition(prcHistSoFar)
            posLimits = np.array([int(x) for x in dlrPosLimit / curPrices])
            newPos = np.clip(newPosOrig, -posLimits, posLimits)
            deltaPos = newPos - curPos
            dvolumes = curPrices * np.abs(deltaPos)
            dvolume = np.sum(dvolumes)
            totDVolume += dvolume
            comm = dvolume * commRate
            cash -= curPrices.dot(deltaPos) + comm
            #######
            predicted = predictPriceNextDay(prcHistSoFar)
            predictAllDays.append(predicted)
            actualNextDayPrice = prcHist[:, t]
            actualPrices.append(actualNextDayPrice)
            predictReturn.append(predictReturnNextDay(prcHistSoFar))
            next_day_return = prcHist[:, t] / prcHist[:, t-1]
            actualReturns.append(next_day_return)
            
            returns = np.ones_like(prcHist)  # initialize with 1s
            returns[:, 1:] = (prcHist[:, 1:] / prcHist[:, :-1])
        else:
            newPos = np.array(curPos)
        curPos = np.array(newPos)
        posValue = curPos.dot(curPrices)
        todayPL = cash + posValue - value
        value = cash + posValue
        ret = 0.0
        if (totDVolume > 0):
            ret = value / totDVolume
        if (t > startDay):
            print ("Day %d value: %.2lf todayPL: $%.2lf $-traded: %.0lf return: %.5lf" % (t,value, todayPL, totDVolume, ret))
            todayPLL.append(todayPL)
    pll = np.array(todayPLL)
    (plmu,plstd) = (np.mean(pll), np.std(pll))
    annSharpe = 0.0
    if (plstd > 0):
        annSharpe = np.sqrt(249) * plmu / plstd
    predictAllDays = np.array(predictAllDays)
    actualPrices = np.array(actualPrices)
    actualReturns = np.array(actualReturns)
    actualReturns = actualReturns.T
    predictReturn = np.array(predictReturn)
    predictReturn = predictReturn.T
    mse = np.mean((predictAllDays - actualPrices) ** 2)
    perStockMSE = np.mean((predictAllDays - actualPrices) ** 2, axis=0)
    print(predictReturn.shape)
    print(actualReturns.shape)
    max = 0
    for i in range(0, 50):
        #print(f"Stock {i}: MSE Price = {stock_mse:.4f}")
        threshold = 0
        a = actualReturns[i]
        p = predictReturn[i]
        mse = np.mean((a - p) ** 2)
        mad = np.mean(np.abs(a - p))
        if (mad > max):
            max = mad
        # print(f'Returns {i} MAD: {mad} MSE: {mse}')
    
        
    print("Max Return MAD:", max)

    return (plmu, ret, plstd, annSharpe, totDVolume)



(meanpl, ret, plstd, sharpe, dvol) = calcPL(prcAll,200)
score = meanpl - 0.1*plstd
print ("=====")
print ("mean(PL): %.1lf" % meanpl)
print ("return: %.5lf" % ret)
print ("StdDev(PL): %.2lf" % plstd)
print ("annSharpe(PL): %.2lf " % sharpe)
print ("totDvolume: %.0lf " % dvol)
print ("Score: %.2lf" % score)

# numTestDays = 200
# todayPLL = []
# (_,nt) = prcAll.shape
# startDay = nt + 1 - numTestDays
# print("Prediction MSE's")
# nInst = 50
# for i in range(startDay, nt + 1):
#     allPred = np.zeros((nInst, numTestDays))
#     dailyPred = np.zeros(nInst)
#     dailyPred = predictPriceNextDay(prcAll[ :, 0:i])
#     break