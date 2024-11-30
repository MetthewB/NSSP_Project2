import numpy as np
import pywt
from scipy.signal import lfilter

def getmavfeat(x, winsize, wininc, datawin=None, dispstatus=False):
    if datawin is None:
        datawin = np.ones(winsize)
    datasize = x.shape[0]
    Nsignals = x.shape[1]
    numwin = (datasize - winsize) // wininc + 1
    feat = np.zeros((numwin, Nsignals))
    st = 0
    en = winsize
    for i in range(numwin):
        curwin = x[st:en, :] * datawin[:, np.newaxis]
        feat[i, :] = np.mean(np.abs(curwin), axis=0)
        st += wininc
        en += wininc
    return np.mean(feat, axis=0)

def getrmsfeat(x, winsize, wininc, datawin=None, dispstatus=False):
    if datawin is None:
        datawin = np.ones(winsize)
    datasize = x.shape[0]
    Nsignals = x.shape[1]
    numwin = (datasize - winsize) // wininc + 1
    feat = np.zeros((numwin, Nsignals))
    st = 0
    en = winsize
    for i in range(numwin):
        curwin = x[st:en, :] * datawin[:, np.newaxis]
        feat[i, :] = np.sqrt(np.mean(curwin**2, axis=0))
        st += wininc
        en += wininc
    return np.mean(feat, axis=0)

def getzcfeat(x, deadzone, winsize, wininc, datawin=None, dispstatus=False):
    if datawin is None:
        datawin = np.ones(winsize)
    datasize = x.shape[0]
    Nsignals = x.shape[1]
    numwin = (datasize - winsize) // wininc + 1
    feat = np.zeros((numwin, Nsignals))
    st = 0
    en = winsize
    for i in range(numwin):
        y = x[st:en, :] * datawin[:, np.newaxis]
        y = (y > deadzone) ^ (y < -deadzone)
        a = 1
        b = np.exp(-(np.arange(1, winsize // 2 + 1)))
        z = lfilter(b, a, y, axis=0)
        z = (z > 0) ^ (z < 0)
        dz = np.diff(z, axis=0)
        feat[i, :] = np.sum(np.abs(dz) == 2, axis=0)
        st += wininc
        en += wininc
    return np.mean(feat, axis=0)

def getsscfeat(x, deadzone, winsize, wininc, datawin=None, dispstatus=False):
    if datawin is None:
        datawin = np.ones(winsize)
    x = np.vstack([np.zeros((1, x.shape[1])), np.diff(x, axis=0)])
    datasize = x.shape[0]
    Nsignals = x.shape[1]
    numwin = (datasize - winsize) // wininc + 1
    feat = np.zeros((numwin, Nsignals))
    st = 0
    en = winsize
    for i in range(numwin):
        y = x[st:en, :] * datawin[:, np.newaxis]
        y = (y > deadzone) ^ (y < -deadzone)
        a = 1
        b = np.exp(-(np.arange(1, winsize // 2 + 1)))
        z = lfilter(b, a, y, axis=0)
        z = (z > 0) ^ (z < 0)
        dz = np.diff(z, axis=0)
        feat[i, :] = np.sum(np.abs(dz) == 2, axis=0)
        st += wininc
        en += wininc
    return np.mean(feat, axis=0)

def getwlfeat(x, winsize, wininc, datawin=None, dispstatus=False):
    if datawin is None:
        datawin = np.ones(winsize)
    datasize = x.shape[0]
    Nsignals = x.shape[1]
    numwin = (datasize - winsize) // wininc + 1
    feat = np.zeros((numwin, Nsignals))
    st = 0
    en = winsize
    for i in range(numwin):
        curwin = x[st:en, :] * datawin[:, np.newaxis]
        feat[i, :] = np.sum(np.abs(np.diff(curwin, axis=0)), axis=0)
        st += wininc
        en += wininc
    return np.mean(feat, axis=0)

def getmavsfeat(x, winsize, wininc, datawin=None, dispstatus=False):
    if datawin is None:
        datawin = np.ones(winsize)
    datasize = x.shape[0]
    Nsignals = x.shape[1]
    numwin = (datasize - winsize) // wininc + 1
    feat = np.zeros((numwin, Nsignals))
    st = 0
    en = winsize
    for i in range(numwin - 1):
        curwin = x[st:en, :] * datawin[:, np.newaxis]
        curwinSucc = x[st + wininc:en + wininc, :] * datawin[:, np.newaxis]
        feat[i, :] = np.mean(np.abs(curwinSucc), axis=0) - np.mean(np.abs(curwin), axis=0)
        st += wininc
        en += wininc
    return np.mean(feat, axis=0)


def getmDWTfeat(x, winsize, wininc, dispstatus=False):
    datasize = x.shape[0]
    Nsignals = x.shape[1]
    numwin = (datasize - winsize) // wininc + 1
    feat = np.zeros((numwin, Nsignals * 4))
    st = 0
    en = winsize
    for i in range(numwin):
        curwin = x[st:en, :]
        m_xk = np.zeros((4, Nsignals))
        for colInd in range(curwin.shape[1]):
            C, L = pywt.wavedec(curwin[:, colInd], 'db7', level=3)
            L = np.cumsum(L)
            L = np.insert(L, 0, 0)
            sReal = [0, 3, 2, 1]
            for s in range(4):
                d_xk = C[L[s]:L[s+1]]
                MaxSum = min(int(np.ceil(winsize / (2**sReal[s] - 1))), len(d_xk))
                m_xk[s, colInd] = np.sum(np.abs(d_xk[:MaxSum]))
        feat[i, :] = m_xk.flatten()
        st += wininc
        en += wininc
    return feat


def getiavfeat(x, winsize, wininc, datawin=None, dispstatus=False):
    if datawin is None:
        datawin = np.ones(winsize)
    datasize = x.shape[0]
    Nsignals = x.shape[1]
    numwin = (datasize - winsize) // wininc + 1
    feat = np.zeros((numwin, Nsignals))
    st = 0
    en = winsize
    for i in range(numwin):
        curwin = x[st:en, :] * datawin[:, np.newaxis]
        feat[i, :] = np.sum(np.abs(curwin), axis=0)
        st += wininc
        en += wininc
    return feat


def getTDfeat(x, deadzone=0, winsize=None, wininc=None, datawin=None, dispstatus=0):
    if winsize is None:
        winsize = x.shape[0]
    if wininc is None:
        wininc = winsize
    if datawin is None:
        datawin = np.ones(winsize)
    feat1 = getmavfeat(x, winsize, wininc)
    feat2 = getmavsfeat(x, winsize, wininc, datawin, dispstatus)
    feat3 = getzcfeat(x, deadzone, winsize, wininc)
    feat4 = getsscfeat(x, deadzone, winsize, wininc)
    feat5 = getwlfeat(x, winsize, wininc)
    feat = np.hstack((feat1, feat2, feat3, feat4, feat5))
    return feat


def getHISTfeat(x, winsize, wininc, edges):
    datasize = x.shape[0]
    Nsignals = x.shape[1]
    numwin = (datasize - winsize) // wininc + 1
    feat = np.zeros((numwin, Nsignals * (len(edges) - 1)))
    st = 0
    en = winsize
    for i in range(numwin):
        curwin = x[st:en, :] * np.ones((winsize, 1))
        F0 = np.zeros((len(edges) - 1, Nsignals))
        for j in range(Nsignals):
            F0[:, j], _ = np.histogram(curwin[:, j], bins=edges)
        feat[i, :] = F0.flatten()  
        st += wininc
        en += wininc
    return feat


def ParFeatureExtractor(emg, stimulus, repetition, deadzone, winsize, wininc, featFunc):
    datawin = np.ones(winsize)
    numwin = emg.shape[0]
    nSignals = emg.shape[1]
    edges = np.arange(-3, 3.3, 0.3)
    if featFunc == 'getHISTfeat':
        emg = (emg - np.mean(emg, axis=0)) / np.std(emg, axis=0)
        feat = np.zeros((numwin, nSignals * len(edges)), dtype=np.float32)
    elif featFunc == 'getTDfeat':
        feat = np.zeros((numwin, nSignals * 5), dtype=np.float32)
    elif featFunc == 'getmDWTfeat':
        feat = np.zeros((numwin, nSignals * 4), dtype=np.float32)
    else:
        feat = np.zeros((numwin, nSignals), dtype=np.float32)
    featStim = np.zeros(numwin, dtype=np.float32)
    featRep = np.zeros(numwin, dtype=np.float32)
    checkStimRep = np.zeros(numwin, dtype=np.float32)
    for winInd in range(numwin - winsize):
        if (winInd - 1) % wininc == 0:
            print(f'Feature Extraction Progress: {round(winInd * 10000 / numwin) / 100}%')
            curStimWin = stimulus[winInd:winInd + winsize]
            curRepWin = repetition[winInd:winInd + winsize]
            if len(np.unique(curStimWin)) == 1 and len(np.unique(curRepWin)) == 1:
                checkStimRep[winInd] = 1
                featStim[winInd] = curStimWin[0]
                featRep[winInd] = curRepWin[0]
                curwin = emg[winInd:winInd + winsize]
                if featFunc == 'getrmsfeat':
                    feat[winInd, :] = getrmsfeat(curwin, winsize, wininc)
                elif featFunc == 'getTDfeat':
                    feat[winInd, :] = getTDfeat(curwin, deadzone, winsize, wininc)
                elif featFunc == 'getmavfeat':
                    feat[winInd, :] = getmavfeat(curwin, winsize, wininc, datawin)
                elif featFunc == 'getzcfeat':
                    feat[winInd, :] = getzcfeat(curwin, deadzone, winsize, wininc, datawin)
                elif featFunc == 'getsscfeat':
                    feat[winInd, :] = getsscfeat(curwin, deadzone, winsize, wininc, datawin)
                elif featFunc == 'getwlfeat':
                    feat[winInd, :] = getwlfeat(curwin, winsize, wininc, datawin)
                elif featFunc == 'getiavfeat':
                    feat[winInd, :] = getiavfeat(curwin, winsize, wininc, datawin)
                elif featFunc == 'getHISTfeat':
                    feat[winInd, :] = getHISTfeat(curwin, winsize, wininc, edges)
                elif featFunc == 'getmDWTfeat':
                    feat[winInd, :] = getmDWTfeat(curwin, winsize, wininc)
                else:
                    raise ValueError('Feature function not yet implemented in FeatureExtractor')
    valid_indices = checkStimRep == 1
    feat = feat[valid_indices, :]
    featStim = featStim[valid_indices]
    featRep = featRep[valid_indices]
    return feat, featStim, featRep