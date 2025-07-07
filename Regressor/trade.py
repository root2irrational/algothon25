from scipy.stats import shapiro
import numpy as np
from scipy.stats import boxcox
from timeSeries import *
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

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
# prc = prc[:, 0:750]
(nInst, nt) = prc.shape
rt = np.ones_like(prc)
rt[:, 1:] = (prc[:, 1:] / prc[:, :-1]) # returns

pairs = np.array([[6, 2], [20, 2], [7, 13], [11, 13], [11, 23], [29, 22], [
                 42, 29], [31, 9], [31, 14], [33, 1], [33, 6], [49, 48]])


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
    rt = spread[1:] / spread[:-1]  # > 1 means spread increased
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
    elif (Z < -z):
        return -1
    else:
        return 0


def pairsTradeSpread(prcSoFar, position):
    # pairs = np.array(getPairs(prcSoFar[:,-windowSysytem:], 0.8, 0.05))
    r, c = pairs.shape
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
 
        S = spreadTradeSignal(spread)
        # S = -S
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
    r, c = pairs.shape
    for i in range(0, r):
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


"""_summary_
  corr = 0.8
  # [
  # [1, 2, 4, 6, 16, 18, 19, 20, 25, 33, 35, 38, 47], 
  # [5, 9, 14, 15, 27],
  # [7, 13, 34, 41],
  # [11, 23], 
  # [12], [17],
  # [29, 42],
  # ]
"""

"""
corr = 0.85
[1, 4, 6, 19, 20, 47],
[2, 16],
[5, 9, 14, 27]
[29, 42]
"""

"""_corr 0.9_
  [1,4,6]
"""


def multiplePairsEma(arr, prcSoFar, pos):
    for row in arr:
        res = priceRatioEma(row[0], row[1], prcSoFar, pos)
        pos[row[0]] = res[row[0]]
        pos[row[1]] = res[row[1]]
    return pos

def priceRatioEma(s1, s2, prcSoFar, pos):
    # stock 2, stock 0
    arr = prcSoFar[s2]/prcSoFar[s1]
    w1,w2,w3 = 8,13,21
    e1 = ema(arr, w1).iloc[-1]
    e2 = ema(arr, w2).iloc[-1]
    e3 = ema(arr, w3).iloc[-1]
    X = ema3MomentumSignal(8,13,21, prcSoFar[s1])
    Y = ema3MomentumSignal(8,13,21, prcSoFar[s2])
    shares1 = int(MAX / prcSoFar[s1][-1])
    shares2 = int(MAX / prcSoFar[s2][-1])

    if ((X == 1 and Y == -1)):
        pos[s1] += -shares1
        pos[s2] += shares2
    elif((X == -1 and Y == 1)):
        pos[s1] += shares1
        pos[s2] += -shares2

    return pos

def multiPairIndex(groups, prcSoFar, pos):
    for arr in groups:
        index = syntheticIndexGivenStocks(arr, prc)
        for i in arr:
            stock = prc[i]
            beta, intercept = np.polyfit(stock, index, 1)
            if (beta < 0):
                beta = -beta
            spread = calcSpread(stock, index, beta)
            Z = pd.Series(spread)
            z = ema(spread, 20)
            S = Z - z
            X = (spread[-1] - z.iloc[-1])/np.std(spread)
            shares = MAX / prcSoFar[i][-1]
            M = spread[-1] - z.iloc[-1]
            f = np.std(spread)
            if (M > f):
                pos[i] -= shares
            elif (M < f):
                pos[i] += shares
    return pos

# mean reverting. cut window before passing
def zScoreSignal(prices):
    mean = np.mean(prices)
    sd = np.std(prices)
    Z = (prices[-1] - mean)/sd
    z = 0
    if (Z > z):
        return -1
    elif(Z < -z):
        return 1
    return 0

def spreadWeakStrongCorr(w1, w2, w3, pairs, prices, pos):
    for row in pairs:
        a = row[0]
        b = row[1]
        arrA = prices[a]
        arrB = prices[b]
        if (pos[a] != 0 or pos[b] != 0): continue
            
        beta, intercept = np.polyfit(arrA, arrB, 1)
        if (beta < 0):
            beta = -beta
        spread = calcSpread(arrA, arrB, beta)
        spread = np.array(spread)
        h = spreadHalfLife(spread[-windowSysytem:])
        sharesA = int(MAX / (prices[a][-1]))
        sharesB = int(MAX / (prices[b][-1] * beta))
        r = np.corrcoef(arrA, arrB)
        r = r[0][1]
        if (h > w2 and 0 < r < 0.25):
            # momentumEma
            signal = ema3MomentumSignal(w1,w2,w3,spread)

            s1 = ema3MomentumSignal(w1,w2,w3,prices[a])
            s2 = ema3MomentumSignal(w1,w2,w3,prices[b])
            if (signal == 1): # a up b down
                if (s1 == 1 and s2 == -1):
                    pos[a] += sharesA
                    pos[b] += -sharesB
            if(signal == -1): # a down b up
                if (s1 == -1 and s2 == 1):
                    pos[a] += -sharesA
                    pos[b] += sharesB
        elif (h < w2 and r > 0.75):
            signal = zScoreSignal(spread)

            if (signal == 1):
                pos[a] += sharesA
                pos[b] += -sharesB
            if(signal == -1):
                pos[a] += -sharesA
                pos[b] += sharesB
        
    return pos

def ema3MomentumSignal(w1,w2,w3, prices):
    # 1 == buy, -1 == sell else 0
    e1 = ema(prices, w1).iloc[-1]
    e2 = ema(prices, w2).iloc[-1]
    e3 = ema(prices, w3).iloc[-1]
    if ((e1 > e2 > e3)): # assume both increase?
        return 1
    elif((e1 < e2 < e3)): #assume both decrease?
        return -1
    return 0

def ema3Trade(w1, w2, w3, prcSoFar, pos):
    nInst,nt = prcSoFar.shape
    i = 2
    e1 = ema(prcSoFar[i], w1).iloc[-1]
    e2 = ema(prcSoFar[i], w2).iloc[-1]
    e3 = ema(prcSoFar[i], w3).iloc[-1]
    shares = int(MAX / prcSoFar[i][-1])
    if (e1 > e2 and e1 > e3):
        pos[i] += shares
    elif(e1 < e2 and e1 < e3):
        pos[i] += -shares
    return pos

def ema3Spread(w1, w2, w3, pairs, prcSoFar, pos):
    for row in pairs:
        prices = prcSoFar
        a = row[0]
        b = row[1]
        arrA = prices[a]
        arrB = prices[b]
        if (pos[a] != 0 or pos[b] != 0): continue
            
        beta, intercept = np.polyfit(arrA, arrB, 1)
        if (beta < 0):
            beta = -beta
        spread = calcSpread(arrA, arrB, beta)
        spread = np.array(spread)
        signal = ema3MomentumSignal(w1,w2,w3, spread)
        sharesA = int(MAX / prcSoFar[a][-1])
        sharesB = int(MAX / prcSoFar[b][-1])
        # signal = -signal
        s1 = ema3MomentumSignal(w1,w2,w3,prices[a])
        s2 = ema3MomentumSignal(w1,w2,w3,prices[b])
        if (signal == 1): # a up b down
            if (s1 == 1 and s2 == -1):
                pos[a] += sharesA
                pos[b] += -sharesB
        # if(s1 == 1 and s2 == 1):
        #     pos[a] += sharesA
        #     pos[b] += sharesB
        elif(signal == -1): # a down b up
            if (s1 == -1 and s2 == 1):
                pos[a] += -sharesA
                pos[b] += sharesB
        # if(s1 == -1 and s2 == -1):
        #     pos[a] += -sharesA
        #     pos[b] += -sharesB
        
    return pos

def experiment(prices):
    a, b = 0,5
    # arrA = np.array(rt[a][200:300])
    # arrB = np.array(rt[b][200:300])

    # arrA = prices[a]
    # arrB = prices[b]
    # if (prices[a][0] < prices[b][0]):
    #     temp = arrB
    #     arrB = arrA
    #     arrA = temp
        
    # beta, intercept = np.polyfit(arrA, arrB, 1)
    # if (beta < 0):
    #     beta = -beta
    # spread = calcSpread(arrA, arrB, beta)
    # arr = np.array([spread])
    # plotMultipleSeries(arr)
    
    # a, b = 0,2
    # arrA = prices[a]
    # arrB = prices[b]
    # if (prices[a][0] < prices[b][0]):
    #     temp = arrB
    #     arrB = arrA
    #     arrA = temp
        
    # beta, intercept = np.polyfit(arrA, arrB, 1)
    # if (beta < 0):
    #     beta = -beta
    # spread = calcSpread(arrA, arrB, beta)
    # arr = np.array([spread])
    # plotMultipleSeries(arr)
    # pairs = findLowCorrPairs(corr(prices), 0, 0.25)
    # print(pairs)
    s = 2
    # arr = np.log(prices[s])
    # plotMultipleSeries([prc[s]])
    stockPrice = prices[s]
    stockPrice = pd.Series(stockPrice)
    # mean = stockPrice.mean()
    # sd = stockPrice.std()
    # rolling_std_21 = stockPrice.rolling(window=21).std()
    # arr = (stockPrice - mean)/sd
    # arr = stockPrice - ema(stockPrice, 13)
    # # arr = stockPrice.diff()

    # arr = arr.dropna()
    # plotHistogram(arr)
    
    # stats.probplot(arr.dropna(), dist="norm", plot=plt)
    # plt.title("QQ Plot: Residual = Price - EMA(21)")
    # plt.show()
    prices = stockPrice
    # Calculate EMAs
    # Calculate EMAs
    ema_8 = prices.ewm(span=8, adjust=False).mean()
    ema_13 = prices.ewm(span=13, adjust=False).mean()
    ema_21 = prices.ewm(span=21, adjust=False).mean()
    df = pd.DataFrame({'price': prices})
    for lag in range(1, 6):  # lag 1 to lag 5
        df[f'price_lag_{lag}'] = prices.shift(lag)
    df[f'ema_8'] = ema_8
    df[f'ema_13'] = ema_13
    df[f'ema_21'] = ema_21
    # Create DataFrame with predictors
    # df = pd.DataFrame({
        # 'price': prices,
        # 'ema_8': ema_8,
        # 'ema_13': ema_13,
        # 'ema_21': ema_21,
        # 'price_lag_1': prices.shift(1)   # lag 1 of price (previous day's price)
    # })

    # Target: next day price
    df['target'] = df['price'].shift(-1)

    # Drop rows with NaNs from EMA and shifting
    df = df.dropna()

    # Define predictors and target
    X = df[['price_lag_1', 'ema_8']]
    y = df['target']

    # Add constant for intercept
    X = sm.add_constant(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    # Fit GLM (multiple linear regression)
    model = sm.GLM(y_train, X_train, family=sm.families.Gaussian())
    results = model.fit()
    y_pred = results.predict(X_test)
    print(results.summary())
    # Plot actual vs predicted
    plt.figure(figsize=(12,6))
    plt.plot(y_test.index, y_test, label='Actual Price', color='blue')
    plt.plot(y_test.index, y_pred, label='Predicted Price', color='red', linestyle='--')
    plt.xlabel('Time Index')
    plt.ylabel('Price')
    plt.title('Actual vs Predicted Stock Price')
    plt.legend()
    plt.show()
    
    return

def experiment2(prices, pos):
    s = 8
    arr = list(range(0, 50))
    # arr.remove(23)
    # arr.remove(28)
    # arr.remove(21)
    # arr.remove(22)
    # arr.remove(48)
    # arr.remove(31)
    # arr.remove(26)
    # arr.remove(29)
    # arr.remove(37)
    # arr.remove(39)
    # arr.remove(11)
    # arr.remove(3)
    # arr.remove(7)
    # arr.remove(16)
    # arr.remove(49)
    # arr.remove(17)
    # arr.remove(42)
    # arr.remove(45)
    # arr.remove(46)
    # arr = [1,6,20,30,47]
    # arr.remove(30)
    for s in arr:
        e = np.array(ema(prices[s], 21))
        windowSysytem = 251
        nig = prices[s][-windowSysytem:] -  e[-windowSysytem:]
        stat, p = shapiro(nig)
        if (p < 0.01):
            # print(f'reject here {p}')
            continue
        S = nig[-1]
        shares = int(MAX / prices[s][-1])
        z = np.std(prices[s])
        if (S > z):
            pos[s] += -shares
        elif (S < z):
            pos[s] += shares
    return pos
experiment2(np.array(prc[0:750]), np.zeros(50))

