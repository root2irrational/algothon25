
import numpy as np
from trade import pairsTradeSpread, windowSysytem, IndexMomentum, reverseTrades

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
    # currentPos = longShortCointegrated(prcSoFar, currentPos)
    # currentPos = longShortRandom(prcSoFar, currentPos)
    # if (IndexMomentum(prcSoFar) != 0):
    #     return currentPos
    currentPos = pairsTradeSpread(prcSoFar, currentPos)
    # currentPos = momentumTrade(prcSoFar, currentPos)
    # currentPos = maCrossover(prcSoFar, currentPos)
    # if (nt <= 500):
    # currentPos = reverseTrades(currentPos)
    return currentPos
