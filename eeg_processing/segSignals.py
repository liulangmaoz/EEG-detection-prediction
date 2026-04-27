import numpy as np

def segSignals(signalsIn, labelsIn, targetLength, channels):
    signalsOut = []
    labelsOut = []

    for idx_ch in channels:
        for idx in range(len(signalsIn)):
            x = signalsIn[idx]
            y = labelsIn[idx]

            # 取出当前通道
            x = x[idx_ch, :]

            numSigs = len(x) // targetLength
            if numSigs == 0:
                continue

            x = x[:numSigs * targetLength]
            M = x.reshape(numSigs, targetLength)

            for row in M:
                signalsOut.append(row.copy())
                labelsOut.append(y)

    return np.array(signalsOut), np.array(labelsOut)