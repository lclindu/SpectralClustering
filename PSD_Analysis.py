import numpy as np
from sklearn import cluster, covariance
import matplotlib

import matplotlib.pyplot as plt

import os
import scipy
from scipy import interpolate


import matplotlib.cm as cm
import scipy.stats as stats

import pandas as pd

import scipy.io as sio

import seaborn as sns 

import copy
class ComputePSD(object):
     
    
    def __init__(self, dataCondu, SmpFreq, lowlogG, highlogG):
        self.Data = dataCondu  ######
        
        #####need modify !!
        
        self.SmpFreq = SmpFreq
        self.lowlogG = lowlogG
        self.highlogG = highlogG
        
        self.dataSmps, = dataCondu.shape  ###dataCut[0].shape;
        self.dataLength = SmpFreq
        
        self.FreqRange = SmpFreq / self.dataLength * np.arange(int(self.dataLength/2)) ### 
        
        self.DatafftAmp = np.zeros((self.dataSmps, self.dataLength))
        self.DataPSD = np.zeros((self.dataSmps, self.dataLength))
        self.DataPSD_Sum = np.zeros((self.dataSmps))
        self.meanG = np.zeros((self.dataSmps))
        self.meanlogG = np.zeros((self.dataSmps))
        self.computed = False;
        self.usefulornot = np.zeros((self.dataSmps)).astype(np.int)
        self.dataSelection = []
        self.dataSelectionDist = []
        
    def computePSD1(self):
#         self.lowFreq = lowFreq
#         self.highFreq = highFreq
        self.dataSelection = []
        self.dataSelectionDist = []
        for i in range(self.dataSmps):
        
            DataConduI = self.Data[i][:, -1]
            
            index = np.where(DataConduI[:] <= self.highlogG)[0]  ##index[0] -0.3
            indexn = np.where(DataConduI[:] >= self.lowlogG)[0]  ##index[-1] -5?
            
         #   index_start = np.where(DataConduI[:] > self.highlogG)[0] #### index[-1]
          #  index_end = np.where(DataConduI[:] < self.loglogG)[0] #### index[0]
            
            if(index.shape[0] < 1):
                self.usefulornot[i] = 0
                continue

            if indexn.shape[0] < 1:
                self.usefulornot[i] = 0
                continue
            if indexn[-1] - index[0] < 10:  ####at least points number 500? \
                #too many??change to 10,,,tunneling will be less almost none???
                self.usefulornot[i] = 0
                continue
              
            
            
            X = 10**DataConduI[index[0]:indexn[-1]]
            
            self.dataSelection.append(X)
            self.dataSelectionDist.append(self.Data[i][:, 0][index[0]:indexn[-1]])

            fftData = scipy.fft(X, self.dataLength)  ##### need more carefulness....
            
            self.DatafftAmp[i] = abs(fftData) / len(X) * 2 ###### need more carefulness....
            self.DatafftAmp[i][0] /= 2.0
            self.DataPSD[i] = self.DatafftAmp[i] ** 2
            
            self.usefulornot[i] = 1
            
            self.meanG[i] = np.mean(X)
            self.meanlogG[i] = np.mean(DataConduI[index[0]:indexn[-1]])
        self.computed = True;
            
    def IntergratePSD(self, lowFreq, highFreq):
        if(self.computed == False):
                self.computePSD1()
                
        FreqIndex, = np.where((self.FreqRange >= lowFreq) &(self.FreqRange <= highFreq))
        for i in range(self.dataSmps):    
            if self.usefulornot[i] == 1:
                self.DataPSD_Sum[i] = scipy.integrate.trapz(self.DataPSD[i, FreqIndex], 
                                                self.FreqRange[FreqIndex]) 
        self.FreqIndex = FreqIndex
            
    def PlotPSD(self, bins=250):
        cmap = plt.cm.get_cmap("rainbow")
        cmap.set_under("white")
        cmap.set_over("red")

        
        
        fig = plt.figure(figsize=(14, 8))
        fig.tight_layout()
        ax = fig.add_subplot(111)
        ax.axis('equal')
        ax.axis('tight')
        nonZeros = np.where(self.usefulornot)[0]
#         nonZeros, = np.where((np.log10(self.meanG) >= self.lowlogG-2) & (np.log10(self.meanG) <= self.highlogG + 2))
#         self.meanG = self.meanG[nonZeros]
#         self.Data
        counts, xbins, ybins, cs = ax.hist2d(np.log10(self.meanG[nonZeros]),
                                             np.log10(self.DataPSD_Sum[nonZeros] / self.meanG[nonZeros]),
                                             cmin=0.5, # cmax=120,
                                             alpha=0.9,  range=((-9.5, 1), (-7.9, 1.)),
                                             bins=bins, cmap=cmap)
        plt.title('PSD analysis', fontsize=12)
        plt.xlabel('Conductance logG', fontsize=12)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.ylabel('Noise PSD/G log(PSD/G)', fontsize=12)
        plt.colorbar(cs,ax=ax)
        self.binCount = counts
        self.xbin = xbins
        self.ybin = ybins
        
    def ChangeConduArea(self, lowlogG, highlogG):
        if((lowlogG == self.lowlogG) & (highlogG == self.highlogG)):
            pass
        else:
            self.lowlogG = lowlogG
            self.highlogG = highlogG
            self.computed = False
            
    def ComputeCorrelation(self):
        n = np.arange(0.5, 2.6, 0.1)
        corr = []
        nonZeros = np.where(self.usefulornot)[0]
        for n1 in n:
            a = self.DataPSD_Sum[nonZeros] / self.meanG[nonZeros]**n1
            b = self.meanG[nonZeros]
            corr.append([n1, np.corrcoef(a,b)[0,1]])
        corr = np.array(corr)
        minIndex = np.argmin(np.abs(corr[:, 1]))
        return corr[minIndex]
    
    def ComputeCorrelationLog(self):
        n = np.arange(0.5, 2.6, 0.1)
        corr = []
        nonZeros = np.where(self.usefulornot)[0]
        for n1 in n:
            a = np.log10(self.DataPSD_Sum[nonZeros] / self.meanG[nonZeros]**n1)
            b = np.log10(self.meanG[nonZeros])
            corr.append([n1, np.corrcoef(a,b)[0,1]])
        corr = np.array(corr)
        minIndex = np.argmin(np.abs(corr[:, 1]))
        return corr[minIndex]
    
    def plotGaussPSD(self, n, xrange, yrange):
        nonZeros = np.where(self.usefulornot)[0]
        gaussaian_plot2(np.log10(self.meanG[nonZeros]), np.log10(self.DataPSD_Sum[nonZeros] / self.meanG[nonZeros].meanG**n), xrange, yrange)

from scipy import stats
def gaussaian_plot(m1, m2, xrange = None, yrange = None):
    
    m1copy = copy.deepcopy(m1)
    m2copy = copy.deepcopy(m2)
    
    if(xrange):
        xmin = xrange[0]
        xmax = xrange[1]
        ymin = yrange[0]
        ymax = yrange[1]
         ####left out the nan value????
        m1copy_NotNanIndex = np.where((m1copy >= xmin) & (m1copy <= xmax))
        m1copy = m1copy[m1copy_NotNanIndex]
        m2copy = m2copy[m1copy_NotNanIndex]
        m2copy_notNanIndex = np.where((m2copy >= ymin) & (m2copy <= ymax))
        m1copy = m1copy[m2copy_notNanIndex]
        m2copy = m2copy[m2copy_notNanIndex]
    else:
        xmin = m1.min()
        xmax = m1.max()
        ymin = m2.min()
        ymax = m2.max()
        ####left out the nan value????
        m1copy_NotNanIndex = np.where((m1copy >= xmin) & (m1copy <= xmax))
        m1copy = m1copy[m1copy_NotNanIndex]
        m2copy = m2copy[m1copy_NotNanIndex]
        m2copy_notNanIndex = np.where((m2copy >= ymin) & (m2copy <= ymax))
        m1copy = m1copy[m2copy_notNanIndex]
        m2copy = m2copy[m2copy_notNanIndex]

    X, Y = np.mgrid[xmin:xmax:50j, ymin:ymax:50j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([m1copy, m2copy])
    
    kernel = stats.gaussian_kde(values)
    Z = np.reshape(kernel(positions).T, X.shape)

    # Zgaussian = kernel()

#     import matplotlib.pyplot as plt
    cmap = plt.cm.get_cmap("rainbow")
    cmap.set_under("white")
    cmap.set_over("red")
#     cmap.min(0.5)
    fig = plt.figure(figsize=(14, 8))
    fig.tight_layout()
    ax = fig.add_subplot(111)
    ax.axis('equal')
    ax.axis('tight')
    image = ax.imshow(np.rot90(Z), cmap=cmap, #plt.cm.rainbow, #coolwarm,  ##gist_earth_r,
               extent=[xmin, xmax, ymin, ymax], interpolation='gaussian', vmin=0.005)


    ax.plot(m1, m2, 'k.', markersize=2, alpha = 0.1)


#     C = plt.contour(X, Y, Z, 10, colors = 'red', linewidth = 0.5)
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    plt.colorbar(mappable=image, ax=ax)
    plt.title('PSD analysis', fontsize=12)
    plt.xlabel('Conductance logG', fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylabel('Noise PSD/G log(PSD/G)', fontsize=12)
    plt.show()

from scipy import stats
def gaussaian_plot2(m1, m2, dx, dy, xrange = None, yrange = None):
    
    m1copy = copy.deepcopy(m1)
    m2copy = copy.deepcopy(m2)
    
    if(xrange):
        xmin = xrange[0]
        xmax = xrange[1]
        ymin = yrange[0]
        ymax = yrange[1]
         ####left out the nan value????
        m1copy_NotNanIndex = np.where((m1copy >= xmin) & (m1copy <= xmax))
        m1copy = m1copy[m1copy_NotNanIndex]
        m2copy = m2copy[m1copy_NotNanIndex]
        m2copy_notNanIndex = np.where((m2copy >= ymin) & (m2copy <= ymax))
        m1copy = m1copy[m2copy_notNanIndex]
        m2copy = m2copy[m2copy_notNanIndex]
    else:
        xmin = m1.min()
        xmax = m1.max()
        ymin = m2.min()
        ymax = m2.max()
        ####left out the nan value????
        m1copy_NotNanIndex = np.where((m1copy >= xmin) & (m1copy <= xmax))
        m1copy = m1copy[m1copy_NotNanIndex]
        m2copy = m2copy[m1copy_NotNanIndex]
        m2copy_notNanIndex = np.where((m2copy >= ymin) & (m2copy <= ymax))
        m1copy = m1copy[m2copy_notNanIndex]
        m2copy = m2copy[m2copy_notNanIndex]

    x = np.arange(xmin, xmax, dx)
    y = np.arange(ymin, ymax, dy)
    
    X, Y = np.meshgrid(x, y)
    positions = np.vstack([X.T.ravel(), Y.T.ravel()])
    values = np.vstack([m1copy, m2copy])
    
    kernel = stats.gaussian_kde(values)
    Z = np.reshape(kernel(positions).T, X.T.shape)

    # Zgaussian = kernel()

#     import matplotlib.pyplot as plt
    cmap = plt.cm.get_cmap("rainbow")
    cmap.set_under("white")
    cmap.set_over("red")
#     cmap.min(0.5)
    fig = plt.figure(figsize=(14, 8))
    fig.tight_layout()
    ax = fig.add_subplot(111)
    ax.axis('equal')
    ax.axis('tight')
    image = ax.imshow(np.rot90(Z), cmap=cmap, #plt.cm.rainbow, #coolwarm,  ##gist_earth_r,
               extent=[xmin, xmax, ymin, ymax], interpolation='gaussian', vmin=0.005)


    ax.plot(m1, m2, 'k.', markersize=2, alpha = 0.1)


#     C = plt.contour(X, Y, Z, 10, colors = 'red', linewidth = 0.5)
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    plt.colorbar(mappable=image, ax=ax)
    plt.title('PSD analysis', fontsize=12)
    plt.xlabel('Conductance logG', fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylabel('Noise PSD/G log(PSD/G)', fontsize=12)
    plt.show()

def ClusterPSD(datacut, lowlogG = -6.9, highlogG = -0.3, SmpFreq = 20000, lowFreq = 0,
              highFreq = 10000):
    
#     lowlogG = -6.9
#     highlogG = -0.3
    PSD_dataCondu = ComputePSD(datacut, SmpFreq, lowlogG, highlogG)  ##1
    PSD_dataCondu.IntergratePSD(lowFreq, highFreq)     ####2
    CountsAll  = PSD_dataCondu.PlotPSD()
    return PSD_dataCondu

def PSD_Interploate(PSD_dataCondu_6, xlim, ylim):
    nzeros = PSD_dataCondu_6.usefulornot == 1
    gaussaian_plot(np.log10(PSD_dataCondu_6.meanG[nzeros]), 
                   np.log10(PSD_dataCondu_6.DataPSD_Sum[nzeros] / PSD_dataCondu_6.meanG[nzeros]), 
                   xlim, ylim)

def PSD_Interploate2(PSD_dataCondu_6, xlim, ylim):
    nzeros = PSD_dataCondu_6.usefulornot == 1
    gaussaian_plot2(np.log10(PSD_dataCondu_6.meanG[nzeros]), 
                np.log10(PSD_dataCondu_6.DataPSD_Sum[nzeros] / PSD_dataCondu_6.meanG[nzeros]), 
                PSD_dataCondu_6.xbin[1]-PSD_dataCondu_6.xbin[0], 
                PSD_dataCondu_6.ybin[1]-PSD_dataCondu_6.ybin[0],
                xlim, ylim)

def PSD_Interploate3(PSD_dataCondu_6, xlim, ylim):
    nzeros = PSD_dataCondu_6.usefulornot == 1
    gaussaian_plot3(np.log10(PSD_dataCondu_6.meanG[nzeros]), 
                np.log10(PSD_dataCondu_6.DataPSD_Sum[nzeros] / PSD_dataCondu_6.meanG[nzeros]), 
                PSD_dataCondu_6.xbin[1]-PSD_dataCondu_6.xbin[0], 
                PSD_dataCondu_6.ybin[1]-PSD_dataCondu_6.ybin[0],
                xlim, ylim)

def gaussaian_plot3(m1, m2, dx, dy, xrange = None, yrange = None):  ####interpolation change!!!
    
    m1copy = copy.deepcopy(m1)
    m2copy = copy.deepcopy(m2)
    
    if(xrange):
        xmin = xrange[0]
        xmax = xrange[1]
        ymin = yrange[0]
        ymax = yrange[1]
         ####left out the nan value????
        m1copy_NotNanIndex = np.where((m1copy >= xmin) & (m1copy <= xmax))
        m1copy = m1copy[m1copy_NotNanIndex]
        m2copy = m2copy[m1copy_NotNanIndex]
        m2copy_notNanIndex = np.where((m2copy >= ymin) & (m2copy <= ymax))
        m1copy = m1copy[m2copy_notNanIndex]
        m2copy = m2copy[m2copy_notNanIndex]
    else:
        xmin = m1.min()
        xmax = m1.max()
        ymin = m2.min()
        ymax = m2.max()
        ####left out the nan value????
        m1copy_NotNanIndex = np.where((m1copy >= xmin) & (m1copy <= xmax))
        m1copy = m1copy[m1copy_NotNanIndex]
        m2copy = m2copy[m1copy_NotNanIndex]
        m2copy_notNanIndex = np.where((m2copy >= ymin) & (m2copy <= ymax))
        m1copy = m1copy[m2copy_notNanIndex]
        m2copy = m2copy[m2copy_notNanIndex]

    x = np.arange(xmin, xmax, dx)
    y = np.arange(ymin, ymax, dy)
    
    X, Y = np.meshgrid(x, y)
    positions = np.vstack([X.T.ravel(), Y.T.ravel()])
    values = np.vstack([m1copy, m2copy])
    
    kernel = stats.gaussian_kde(values)
    Z = np.reshape(kernel(positions).T, X.T.shape)

    # Zgaussian = kernel()

#     import matplotlib.pyplot as plt
    cmap = plt.cm.get_cmap("rainbow")
    cmap.set_under("white")
    cmap.set_over("red")
#     cmap.min(0.5)
    fig = plt.figure(figsize=(14, 8))
    fig.tight_layout()
    ax = fig.add_subplot(111)
    ax.axis('equal')
    ax.axis('tight')
    image = ax.imshow(np.rot90(Z), cmap=cmap, #plt.cm.rainbow, #coolwarm,  ##gist_earth_r,
               extent=[xmin, xmax, ymin, ymax], interpolation='nearest', vmin=0.005)
# Supported values are 'none', 'nearest', 'bilinear', 'bicubic',
#     'spline16', 'spline36', 'hanning', 'hamming', 'hermite', 'kaiser',
#     'quadric', 'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc',
#     'lanczos'.

    ax.plot(m1, m2, 'k.', markersize=2, alpha = 0.1)


#     C = plt.contour(X, Y, Z, 10, colors = 'red', linewidth = 0.5)
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    plt.colorbar(mappable=image, ax=ax)
    plt.title('PSD analysis', fontsize=12)
    plt.xlabel('Conductance logG', fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylabel('Noise PSD/G log(PSD/G)', fontsize=12)
    plt.show()


'''example: 

PSD_Cluster4_All_10_24 = ClusterPSD(ConduTraces, -5, -2, 20000, 100, 1000)

print(len(PSD_Cluster4_All_10_24.dataSelection), PSD_Cluster4_All_10_24.dataSmps)##### 10 

PSD_Interploate2(PSD_Cluster4_All_10_24, [-6,-1], [-7, -1])
del PSD_Cluster4_All_10_24

'''



### band filter function'''

from scipy import signal

# fs = 2000 ## 20K Hz 
def fband(x_1,Fstop1,Fstop2, fs): #（输入的信号，截止频率下限，截止频率上限）
    b, a = signal.butter(8, [2.0*Fstop1/fs,2.0*Fstop2/fs], 'bandpass')
    filtedData = signal.filtfilt(b,a,x_1)
    return filtedData


def fhp(x_1,Fc, fs):  #（输入的信号，限制频率）
    b, a = signal.butter(8,2.0*Fc/fs, 'highpass')  
    filtedData = signal.filtfilt(b, a,x_1) 
    return filtedData

def flp(x_1,Fc,fs):  #（输入的信号，限制频率）
    b, a = signal.butter(8,2.0*Fc/fs, 'lowpass')  
    filtedData = signal.filtfilt(b, a,x_1) 
    return filtedData



def fda_hp(sig, Fc, fs):
    sos = signal.butter(8, Fc, 'hp', fs=fs, output='sos')
    filtered = signal.sosfilt(sos, sig)
    return filtered

def fda_lp(sig, Fc,fs):
    sos = signal.butter(8, Fc, 'lp', fs=fs, output='sos')
    filtered = signal.sosfilt(sos, sig)
    return filtered

def fda_bp(sig, Fc, Fstop,fs):
    sos = signal.butter(8, [Fc, Fstop], 'bp', fs=fs, output='sos')
    filtered = signal.sosfilt(sos, sig)
    return filtered



a = np.array([[1, 3,5, 6, 7],[1,3,4,5,6]]).T

a.reshape(len(a), -1)

a[:, -1]
