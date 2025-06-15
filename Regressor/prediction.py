import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

nInst = 50
nt = 750
data = np.loadtxt('prices.txt')
data = data.T # row = inst [0: 49], col = day [0: 750]
arr = data
i = 0
x = np.arange(arr.shape[1])

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
def filterCorrelation(corrData, filter):
    if filter == 0:
        for i in range(0, nInst):
            for j in range(0, nInst):
                if abs(corrData[i][j]) < 0.75:
                    corrData[i][j] = 0
    elif filter == 1:
        for i in range(0, nInst):
            for j in range(0, nInst):
                if abs(corrData[i][j]) < 0.75 or corrData[i][j] < 0:
                    corrData[i][j] = 0
    elif filter == -1:
        for i in range(0, nInst):
            for j in range(0, nInst):
                if abs(corrData[i][j]) < 0.75 or corrData[i][j] > 0:
                    corrData[i][j] = 0
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
    global plotPosPairs 
    plotPosPairs = np.copy(posPairsDf)
    global plotNegPairs 
    plotNegPairs = np.copy(negPairsDf)
    
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

arr = plotPosPairs[1]
arr = np.append(arr, 1)
arr = arr[~np.isnan(arr)]
print(arr)
plotInst(arr)
arr = plotNegPairs[0]
arr = arr[~np.isnan(arr)]
arr = np.append(arr, 0)
print(arr)
plotInst(arr)