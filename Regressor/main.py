
import numpy as np
from trade import pairsTradeSpread, windowSysytem, IndexMomentum, reverseTrades, multiPairIndex, ema3Trade, priceRatioEma, multiplePairsEma

#from trade import predictReturnNextDay, predictPriceNextDay
# predictAll, daysTraded,
##### TODO #########################################
### IMPLEMENT 'getMyPosition' FUNCTION #############
### TO RUN, RUN 'eval.py' ##########################

nInst = 50
currentPos = np.zeros(nInst)
day_avg4 = 20
stock = 2
commRate = 0.0005

# Co-integrated pairs {(2, 20), (11, 23), (2, 6)}
def getMyPosition(prcSoFar):
    global currentPos
    (nins, nt) = prcSoFar.shape
    currentPos = np.zeros(nins)
    if (nt <= windowSysytem):
        return currentPos

    # arr = [1, 4, 6]
    #    [2, 16], [5, 9, 14, 27], [29, 42]
    # [2, 16], [1, 4, 6, 19, 20, 47], [1, 4, 6], [29,42]
    # 0.75 [1, 2, 4, 5, 6, 7, 9, 13, 14, 15, 16, 18, 19, 20, 25, 27, 30, 33, 34, 35, 38, 41, 43, 44, 47]
    groups = [[1, 2, 4, 6, 16, 18, 19, 20, 25, 33, 35, 38, 47]]
    groups = [[1,4,6]]
    # groups = [
    #     [1, 2, 4, 5, 6, 7, 9, 13, 14, 15, 16, 18, 19, 20, 25, 27, 30, 33, 34, 35, 38, 41, 43, 44, 47]
    #     ]
    # currentPos = multiPairIndex(groups, prcSoFar, currentPos)
    currentPos = ema3Trade(8,13,21,prcSoFar, currentPos)
    pairs = [[1, 2], [4, 5], [6, 7], [9, 13], [14, 15], [16, 18], [19, 20], [25, 27], [30, 33], [34, 35], [38, 41], [43, 44]]
    pairs = [[0,2]]
    currentPos = multiplePairsEma(np.array(pairs),prcSoFar, currentPos)
    # currentPos = momentumTrade(prcSoFar, currentPos)
    # currentPos = maCrossover(prcSoFar, currentPos)
    # if (nt <= 500):

    currentPos = reverseTrades(currentPos)
    return currentPos
