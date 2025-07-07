
import numpy as np
from trade import pairsTradeSpread, windowSysytem, IndexMomentum, reverseTrades, multiPairIndex, ema3Trade, priceRatioEma, multiplePairsEma
from trade import ema3Spread, spreadWeakStrongCorr, experiment2
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

    pairs = [[8, 30], [9, 11], [10, 24], [19, 31]]
    pairs = [[1, 5], [1, 14], [2, 36], [2, 41], [3, 22], [3, 48], [4, 37], [5, 24], [7, 9], [7, 24], 
             [7, 45], [8, 18], [8, 30], [9, 11], [9, 12], [9, 23], [9, 48], [10, 24], [10, 28], [10, 41], 
             [11, 14], [11, 41], [13, 45], [14, 23], [14, 46], [15, 24], [16, 37], [17, 27], [19, 31], [19, 44], 
             [22, 36], [22, 37], [23, 28], [23, 38], [23, 41], [25, 40], [28, 39], [31, 46], [31, 49], [33, 40], 
             [35, 37], [36, 41], [36, 47], [39, 45], [40, 43], [41, 42], [41, 43]]
    pairs = [
        [8,30], [2,6], 
        [36,47], [11,23], 
        [6,3], [40, 43],
        [7,13], [14, 46]
        ]
    # pairs = [[2,6]]
    # pairs = [[8,30]]
    # currentPos = ema3Spread(8,13,21,pairs,prcSoFar,currentPos)
    # currentPos = spreadWeakStrongCorr(8,13,21,pairs,prcSoFar,currentPos)
    currentPos = experiment2(prcSoFar, currentPos)
    currentPos = reverseTrades(currentPos)
    return currentPos
