import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import coint

nInst = 50
nt = 750
data = np.loadtxt('prices.txt')
data = data.T # row = inst [0: 49], col = day [0: 750]
arr = data
i = 0
x = np.arange(arr.shape[1])
STRONG = 0.9

#after we pick pairs we can make prediction/trade model

# just a general visualiser, plots prices for all instruments in array (on same axis)
def plotInst(arr):
    for i in range(0, len(arr)):
        plt.plot(x, data[i], label='Instrument ' + str(arr[i]))
        plt.xlabel("Day")
        plt.ylabel("Price")
        plt.grid(True)
        plt.legend()

    plt.title("Instruments:")
    plt.show()
    return

# gets correlation array of instruments, arr[i][j] is correlation co-ef of instrument i with instrument j
def getCorrelation(corrData):
    for i in range(0, nInst):
        for j in range(0, nInst):
            if j != i:
                r = np.corrcoef(data[i], data[j])
                corrData[i][j] = r[0, 1]

    return corrData

"""
filters correlation array, leaving only strong ones
    filter (int): -1 means only take strong negative correlation
                    + 1 means only take strong positive
                    0 means take both strong pos and strong neg
"""
def filterCorrelation(corrData, filterNum):
    if filterNum == 0:
        return np.where(abs(corrdata) < STRONG, 0, corrdata)
    elif filterNum == 1:
        return np.where(corrData < STRONG, 0, corrData)
    elif filterNum == -1:
        return np.where(corrData > -STRONG, 0, corrData)
    return corrData

# returns 2D array. arr[0][0] = 21 means stock i = 0 can be paired with stock 21
def getPairs(pairs, corr):
    for i in range(0, nInst):
        k = 0
        for j in range(0, nInst):
            if (abs(corr[i][j]) != 0):
                pairs[i][k] = j
                k += 1
    return pairs

# writes corelations and pairs in new files, also assigns to global variables plotPosPairs and plotNegPairs
def writePairs():
    # allCorr has info on each stocks correlation with each other
    corrData = np.zeros((nInst, nInst))
    corrData = getCorrelation(corrData)
    df = pd.DataFrame(corrData)
    with open("allCorr.txt", "w") as f:
        f.write(df.to_string(index=False))
    
    # get strongly positively and negatively correlated pairs separately
    posStrongCorr = np.copy(corrData)
    negStrongCorr = np.copy(corrData)
    posStrongCorr = filterCorrelation(posStrongCorr, 1)
    negStrongCorr = filterCorrelation(negStrongCorr, -1)
    posPairs = np.full((nInst, nInst), -1) # not setting to 0 since stock i can be paired with stock 0
    negPairs = np.full((nInst, nInst), -1)
    posPairs = getPairs(posPairs, posStrongCorr)
    negPairs = getPairs(negPairs, negStrongCorr)
    posPairsFilt = [[val for val in row if val != -1] for row in posPairs] # Remove all -1s from each row
    negPairsFilt = [[val for val in row if val != -1] for row in negPairs]
    posPairsDf = pd.DataFrame(posPairsFilt)
    negPairsDf = pd.DataFrame(negPairsFilt)
    global corrPosPairs 
    corrPosPairs = [
        (i, val)
        for i, row in posPairsDf.iterrows()
        for val in row
        if pd.notna(val)
    ]
    global corrNegPairs
    corrNegPairs = [
        (i, val)
        for i, row in negPairsDf.iterrows()
        for val in row
        if pd.notna(val)
    ]
    
    with open("posPairs.txt", "w") as f:
        for idx, row in posPairsDf.iterrows():
            row_str = ' '.join([str(val) if pd.notna(val) else '' for val in row])
            f.write(f"row {idx}: {row_str}\n")
    
    with open("negPairs.txt", "w") as f:
        for idx, row in negPairsDf.iterrows():
            row_str = ' '.join([str(val) if pd.notna(val) else '' for val in row])
            f.write(f"row {idx}: {row_str}\n")
            
    return

writePairs()

cointegrated_pairs = []
# filters the highly correlated pairs (of 0.9 co-ef) even more
def coIntegratedTest(cointegrated_pairs):
    for i, j in corrPosPairs:
        i, j = int(i), int(j)  # Convert float to int
        series1 = data[i]
        series2 = data[j]

        coint_t, p_value, _ = coint(series1, series2)

        if p_value < 0.05:
            cointegrated_pairs.append((i, j))

    print("Cointegrated +ve pairs:", cointegrated_pairs)
    arr = list(set([x for tup in cointegrated_pairs for x in tup]))
    #plotInst(arr)
    negPairs = []
    for i, j in corrNegPairs:
        i, j = int(i), int(j)  # Convert float to int
        series1 = data[i]
        series2 = data[j]

        coint_t, p_value, _ = coint(series1, series2)

        if p_value < 0.05:
            negPairs.append((i, j))
            cointegrated_pairs.append((i, j))

    print("Cointegrated -ve pairs:", negPairs)
    arr = list(set([x for tup in negPairs for x in tup]))
    #plotInst(arr)
    
    return

coIntegratedTest(cointegrated_pairs)
cointegrated_pairs = set(tuple(sorted(cointegrated_pairs)) for cointegrated_pairs in cointegrated_pairs)
print(cointegrated_pairs)
