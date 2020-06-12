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

colorsSeq=['r','b','y','m','dodgerblue', 'c', 'gold', 'coral', 'darkorange', 'violet','indigo', 'fuchsia','g','k']

large = 26; med = 24; small = 18
params = {'axes.titlesize': large,
          'legend.fontsize': small,
          'figure.figsize': (10, 8),
          'axes.labelsize': large,
          'axes.titlesize': large,
          'xtick.labelsize': med,
          'ytick.labelsize': med,
          'figure.titlesize': large}
plt.rcParams.update(params)

#### plot functions: 
def plateau_length2(dataCut0, peak_start = - 0.5,peak_end = - 6.5):
#     peak_start = - 2
#     peak_end = - 5.5
    m,  = dataCut0.shape
    d = np.zeros((m))

    for i in range(m):
        DataConduI = dataCut0[i][:, 1]
        index, = np.where(DataConduI[:] >= peak_start)

        indexn, = np.where(DataConduI[:] <= peak_end)
        if(index.shape[0] < 1):
            index = [0]
        if indexn.shape[0] < 1:
            indexn = [-1]                      ####### 
        
        DatadistI = dataCut0[i][:, 0]           
        
        if(indexn[0] < index[-1]):
            d[i] = 0
        else:
            d[i] = np.abs(DatadistI[index[-1]] - DatadistI[indexn[0]])
        
    zerosValueIndex = np.where(d <= 0.025)[0]#####magic valus 0.025;;;
    d = np.delete(d, zerosValueIndex)
    return d

def plot_plateau_length(label_junc_or_not, dataCut, peak_start = - 0.5,peak_end = - 6.5):
        
    labelsmax = len(set(label_junc_or_not))
    j=0
    plt.figure(labelsmax, figsize=(6*labelsmax, 5))
    for i in set(label_junc_or_not):
        

        index_i, = np.where(label_junc_or_not == i)
#         print(index_i)
        if index_i.shape[0] < 3:
            continue
        
        ClusterI_Cut = dataCut[index_i]
        
#         ClusterI_x = dataDist[index_i]
#         ClusterI = dataCondu[index_i]  
        
        plt.subplot(1, labelsmax, j+1)
        j += 1
        plateau_data2 = plateau_length2(ClusterI_Cut, peak_start, peak_end)
        
        
        gkde = stats.gaussian_kde(plateau_data2)
        ind = np.linspace(0.0,plateau_data2.max()+0.5, 100)
        kdepdf = gkde.evaluate(ind)
#         print(gkde.)
        
        n, bins, patches = plt.hist(plateau_data2, bins = 100, 
                                    color = 'gray', edgecolor = 'blue', density=True)
        
        mu = np.mean(plateau_data2)
#         sigma = np.std(plateau_data2)
#         y = stats.norm.pdf(bins, mu, sigma)
        
        plt.axvline(x = mu, linewidth = 1.5, color = 'black')
        plt.plot(ind, kdepdf, label='KDE', color = 'r')
        plt.xlim(0.0, np.max(plateau_data2)+0.5)##### 1.5 need modification according the max length of plateau.
        
        plt.title('Length')
        plt.xlabel('Length /nm')
        plt.ylabel('Counts')
    plt.show()
    return labelsmax


#####functions of  plot 1d and 2d histograms;;;

def concate_to_one_row(dataCut):
    tempData = np.vstack((x for x in dataCut))
    dataDist = tempData[:, 0]
    dataCondu = tempData[:, -1]

    return dataDist, dataCondu


def plothistofClusters(label_junc_or_not, dataCut):
    labelsmax = len(set(label_junc_or_not))
    j=0
    plt.figure(labelsmax, figsize=(6*labelsmax, 5))
    for i in set(label_junc_or_not):
        

        index_i = np.array(np.where(label_junc_or_not == i)).reshape(-1)

        if index_i.shape[0] < 3:
            continue
        
        ClusterI_Cut = dataCut[index_i]
        
        ClusterI_x, ClusterI = concate_to_one_row(ClusterI_Cut)
        

        plt.subplot(1, labelsmax, j+1)
        j += 1
        plt.title('ClusterData %s %s' %(i, index_i.shape[0]))
        ClusterI_2dHist = plt.hist(ClusterI, bins = 300, range=(-7, 0.5),
                                   histtype = 'stepfilled', alpha = 0.8, 
                                   label='Cluster:%d'%(i), density=True)
#         plt.legend(fontsize='x-large')
        plt.title('Histogram')
        plt.ylabel('Counts')
        plt.xlabel('Conductance /$\mathrm{\log(G/G_0)}$')
    plt.show()
    
    
def plot2dCloudofClusters(Label_junc, dataCut):    
    labelsmax = len(set(Label_junc)) 
    j = 0
#     print(labelsmax)
    plt.figure(labelsmax, figsize=(4*labelsmax, 5))
    for i in set(Label_junc):
        

        index_i, = np.array(np.where(Label_junc == i))
    #     print(i)
        if index_i.shape[0] < 3:
            continue

        ClusterI_Cut = dataCut[index_i]
        ClusterI_x, ClusterI = concate_to_one_row(ClusterI_Cut)

        plt.subplot(1, labelsmax, j+1)
        j += 1
        plt.title('ClusterData %d %s' %(i, index_i.shape[0]))
        plt.xlabel('Distance /nm')
        if j==1:
            plt.ylabel('Conductance /$\mathrm{\log(G/G_0)}$')
        ClusterI_2dHist = plt.hist2d(ClusterI_x.reshape(-1), 
                                       ClusterI.reshape(-1), 100, range=((-0.5, 2), (-7, 1)),
                             cmap=cm.coolwarm, normed = True,vmax=0.3, vmin=0.00)
    plt.show() 
    return labelsmax


def plot_hist2(labels1Cluster, dataCut):
    plt.figure(figsize=(12, 8))
    for i in set(labels1Cluster):
        index_i = np.array(np.where(labels1Cluster == i)).reshape(-1)

        ClusterI_Cut = dataCut[index_i]
        ClusterI_x, ClusterI = concate_to_one_row(ClusterI_Cut)

       
        plt.title('ClusterData %s %s' %(i, index_i.shape[0]))
        ClusterI_2dHist = plt.hist(ClusterI.reshape(-1), bins = 730, range=(-7.5, 0.5),
                                   histtype = 'stepfilled', alpha = 0.6, 
                                   label='Cluster:%d'%(i))

        plt.title('Histogram')
        plt.ylabel('Counts')
        plt.xlabel('Conductance /$\mathrm{\log(G/G_0)}$')
    plt.show()
    return ClusterI_2dHist

def plot2dCloudofClusters_ax(Label_junc, dataCut, rangeH=((-0.5, 2), (-7, 1)), vmax=0.5, vmin=0.0, save=0,filename_data=None):    
    '''if save != 0, please set the filename..'''
    
    labelsmax = len(set(Label_junc)) 
    j = 0

    xticks = np.arange(np.int(rangeH[0][0]), np.int(rangeH[0][1])+1, 1)
    if rangeH[0][1] < 1.1:
        xticks = np.arange(np.int(rangeH[0][0]), rangeH[0][1]+0.5, 0.5)
        
    yticks = np.arange(np.int(rangeH[1][0]), np.int(rangeH[1][1])+1, 1)
    
    xticks = xticks[xticks <= rangeH[0][1]]
    yticks = yticks[yticks <= rangeH[1][1]]
    fig, ax = plt.subplots(figsize=(4*labelsmax, 5.5), ncols=labelsmax)

    for i in set(Label_junc):
        

        index_i, = np.array(np.where(Label_junc == i))
    
        if index_i.shape[0] < 3:
            continue

        ClusterI_Cut = dataCut[index_i]
        ClusterI_x, ClusterI = concate_to_one_row(ClusterI_Cut)
        j += 1
        ax[j-1].plot()
        
        ax[j-1].set_title('Cluster %d: %s' %(i, index_i.shape[0]))
        ax[j-1].set_xlabel('Distance /nm')
        if j==1:
            ax[j-1].set_ylabel('Conductance /$\mathrm{\log(G/G_0)}$')
        ClusterI_2dHist = ax[j-1].hist2d(ClusterI_x.reshape(-1), 
                                       ClusterI.reshape(-1), (100, 300), range=rangeH,
                                       cmap=cm.coolwarm, normed = True,vmax=vmax, vmin=vmin)
        ax[j-1].set_xticks(xticks)
        ax[j-1].set_yticks(yticks)

    fig.subplots_adjust(left=0.07, right=0.87)
    box = ax[-1].get_position()
    pad, width = 0.13, 0.02
    cax = fig.add_axes([box.xmax + pad, box.ymin+0.06, width, box.height-0.04])
    fig.colorbar(ClusterI_2dHist[3], cax=cax)
    fig.tight_layout()
    fig.show()

    if(save == 1):
        plt.savefig(filename_data[:-4]+'LLC_Fig.tif', dpi=600, format='tif')
    if(save == 2):
        plt.savefig(filename_data[:-4]+'_%dClusters_LLC_Fig.png'%labelsmax,bbox_inches='tight', format='png')
    return labelsmax

def plot2dCloudofClusters_2(Label_junc, dataCut, rangeH=((-0.5, 2), (-7, 1)), vmax=0.5, vmin=0.0, save=0,filename_data=None):    
    '''if save != 0, please set the filename..'''
    
    labelsmax = len(set(Label_junc)) 
    j = 0

    plt.figure(labelsmax, figsize=(4*labelsmax, 5.5))
    for i in set(Label_junc):
        

        index_i, = np.array(np.where(Label_junc == i))
    #     print(i)
        if index_i.shape[0] < 3:
            continue

        ClusterI_Cut = dataCut[index_i]
        ClusterI_x, ClusterI = concate_to_one_row(ClusterI_Cut)

        plt.subplot(1, labelsmax, j+1)
        j += 1
        plt.title('Cluster %d: %s' %(i, index_i.shape[0]))
        plt.xlabel('Distance /nm')
        if j==1:
            plt.ylabel('Conductance /$\mathrm{\log(G/G_0)}$')
        ClusterI_2dHist = plt.hist2d(ClusterI_x.reshape(-1), 
                                       ClusterI.reshape(-1), (100, 300), range=rangeH,
                             cmap=cm.coolwarm, normed = True,vmax=vmax, vmin=vmin)
        plt.xticks(list(range(np.int(rangeH[0][0]),np.int(rangeH[0][1])+1)))
        plt.yticks(list(range(np.int(rangeH[1][0]),np.int(rangeH[1][1])+1)))
    plt.tight_layout()
    plt.show()
#         plt.colorbar()
    if(save == 1):
        plt.savefig(filename_data[:-4]+'LLC_Fig.tif', dpi=600, format='tif')
    if(save == 2):
        plt.savefig(filename_data[:-4]+'_%dClusters_LLC_Fig.png'%labelsmax,bbox_inches='tight', format='png')
    return labelsmax

def plotCorrelation(dataHistAll3_5, dataHistAll_x, cmapColor=cm.coolwarm):
    histcorrcoef3_5 =  np.corrcoef(dataHistAll3_5.T)
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111)
    ax.axis('equal')
    image = ax.imshow(np.rot90(histcorrcoef3_5), cmap=cmapColor, # cmap, #plt.cm.rainbow, #coolwarm,  ##gist_earth_r,
           extent=[dataHistAll_x[0], dataHistAll_x[-1],dataHistAll_x[0], dataHistAll_x[-1]],  aspect='auto', vmax=0.1, vmin=-0.2)
    plt.colorbar(image)
    ax.ylabel('Conductance /$\mathrm{\log(G/G_0)}$')
    ax.xlabel('Conductance /$\mathrm{\log(G/G_0)}$')
    plt.title('Correlation')
    plt.show()
    return dataHistAll_x[0], dataHistAll_x[-1]

import time
def plotCurve(datacut):
    
    plt.figure(figsize=(10, 8))
    t_start1 = time.clock()
    # list_pab = list(map(getCurve, [i for i in range(dataCondu1Cluster_SP.shape[0])]))
    for i in range(datacut.shape[0]):
        indexbetween = np.where((datacut[i][:, 1] < 0.3) & (datacut[i][:, 1] > -6.9))
        startIndex = indexbetween[0][0]
        endIndex = indexbetween[0][-1]
        plt.plot(datacut[i][startIndex:endIndex, 0], datacut[i][startIndex:endIndex, 1])

    plt.xlabel('Distance /nm')
    plt.ylabel('Conductance /$\mathrm{log(G/G_0)}$')
    t_end1 = time.clock() - t_start1
    plt.show()
    print('******************************')
    print("* Plot Time " + str(round(t_end1, 6)) + " s！*" )
    print('******************************')

def plot(ConduTraces, index):
    plt.figure(figsize=(4, 6))
    plt.plot(ConduTraces[index][:, 0], ConduTraces[index][:, 1], lw=3)
    plt.ylim(-8., 1)
    plt.yticks(np.arange(-8,1))
    plt.title('')
    plt.xlabel('Distance /nm')
    plt.ylabel('Conductance /$\mathrm{log(G/G_0)}$')
    plt.show()

#cmCollor = ['r','b', 'coral', 'cyan',  'y', 'gold', 'hotpink', 'indigo', 'fuchsia'] #### need to be more ...Changed Changed 20190904;;
cmCollor = colorsSeq
len_cmCollor = len(cmCollor)


def plot_hist3(lowlogG, highlogG, labels1Cluster, dataHistAll, dataHistAll_x):
    
    
    cbins = np.where((dataHistAll_x >= lowlogG) & (dataHistAll_x <= highlogG))[0]
    
    dataHistAll_6 = dataHistAll[:, cbins]
#     plt.figure(figsize=(12, 8))
    for i in set(labels1Cluster):
        index_i = np.array(np.where(labels1Cluster == i)).reshape(-1)

        Cluster_i_hist = dataHistAll_6[index_i]
        
        y2, = plt.plot(dataHistAll_x[cbins], Cluster_i_hist.mean(axis=0), 
                       color=cmCollor[i%len_cmCollor], lw=0)
#         plt.fill_between()
        plt.fill_between(dataHistAll_x[cbins], 0, Cluster_i_hist.mean(axis=0), color=cmCollor[i%len_cmCollor],label='Cluster %d:%d ' %(i,index_i.shape[0])) #, alpha = 0.9, color=[1,0,1.0])#, color = 'b')
#         ClusterI_2dHist = plt.hist(ClusterI.reshape(-1), bins = 300, range=(-6.3, 0.3),
#                                    histtype = 'stepfilled', alpha = 0.9, density=True, 
#                                    label='Cluster:%d'%(i))
        plt.legend(fontsize = 18)
        plt.title('Histogram')
        plt.ylabel('Counts per trace')
        plt.xlabel('Conductance /$\mathrm{log(G/G_0)}$')
    plt.show()

def plot_hist3Line(lowlogG, highlogG, labels1Cluster, dataHistAll, dataHistAll_x):
    
    
    cbins = np.where((dataHistAll_x >= lowlogG) & (dataHistAll_x <= highlogG))[0]
    
    dataHistAll_6 = dataHistAll[:, cbins]
#     plt.figure(figsize=(12, 8))
    for i in set(labels1Cluster):
        index_i = np.array(np.where(labels1Cluster == i)).reshape(-1)

        Cluster_i_hist = dataHistAll_6[index_i]
        
        y2, = plt.plot(dataHistAll_x[cbins], Cluster_i_hist.mean(axis=0), 
                       color=cmCollor[i%len_cmCollor], lw=2, label='Cluster %d:%d ' %(i,index_i.shape[0]))
#         plt.fill_between()
#         plt.fill_between(dataHistAll_x[cbins], 0, Cluster_i_hist.mean(axis=0), color=cmCollor[i%len_cmCollor],) #, alpha = 0.9, color=[1,0,1.0])#, color = 'b')
#         ClusterI_2dHist = plt.hist(ClusterI.reshape(-1), bins = 300, range=(-6.3, 0.3),
#                                    histtype = 'stepfilled', alpha = 0.9, density=True, 
#                                    label='Cluster:%d'%(i))
        plt.legend(fontsize = 18)
        plt.title('Histogram')
        plt.ylabel('Counts per trace')
        plt.xlabel('Conductance /$\mathrm{log(G/G_0)}$')
    plt.show()

def plotHistofClusters_perTrace(label_junc_or_not, lowlogG, highlogG, dataHistAll, dataHistAll_x):
    
    cbins = np.where((dataHistAll_x >= lowlogG) & (dataHistAll_x <= highlogG))[0]
    labelsmax = len(set(label_junc_or_not))
    j=0
    labels = label_junc_or_not

    y1, = plt.plot(dataHistAll_x[cbins], dataHistAll[:, cbins].mean(axis=0), 'b', label='Original Data')
    plt.ylabel('Counts per trace')
    plt.xlabel('Conductance /$\mathrm{log(G/G_0)}$')
    plt.title('%0.1f Original data and Cluster %0.1f'%(lowlogG, highlogG) )

    for i in range(labelsmax):
        nClusterI = (labels == i).sum()
        index_i = np.array(np.where(labels == i)).reshape(-1)
        Cluster_i_hist = dataHistAll[index_i][:, cbins]
        y2, = plt.plot(dataHistAll_x[cbins], Cluster_i_hist.mean(axis=0), label='Cluster %d_%d/%d ' %(i+1,nClusterI,labelsmax))
    plt.legend(bbox_to_anchor = (1.3, 0.5), ncol = 1, loc = 7, borderaxespad=0)
    plt.show()
    
def plot_hist_of_Traces(ConduTraces, cbins = 730, rangeG = (-7.5, 0.5)):
    
    NumTraces = len(ConduTraces)
    
    rangeCondu = rangeG
    
    dataHistAll = np.zeros((cbins))
    
    for i in range(NumTraces):
        data_hist_i = np.histogram(ConduTraces[i][:,-1], cbins, range = rangeCondu, density=None) 
        dataHistAll += data_hist_i[0]
    dataHistAll_x = data_hist_i[1][1:]
    dataHistAll /= NumTraces
    
    plt.plot(dataHistAll_x, dataHistAll, label="histogram")
#     plt.fill(dataHistAll_x, dataHistAll, )
    plt.legend(fontsize=18)
    plt.title('Histogram')
    plt.ylabel('Counts')
    plt.xlabel('Conductance /$\mathrm{log(G/G_0)}$')
    plt.show()
    return dataHistAll_x, dataHistAll

def plotLabel(labels1Cluster_SP, index = 1):
    plt.figure(1, figsize=(3, 20))
    lab1 = labels1Cluster_SP == index
    #     index, = np.where(labels1Cluster_SP == i)
    x_range = np.arange(labels1Cluster_SP.shape[0])
    # plt.barh(x_range[lab0], labels1Cluster_SP[labels1Cluster_SP == 0]+1, 1)
    plt.barh(x_range[lab1], 1, 1.)
    plt.show()

def TimeSeq(labelList_5_1, Kopt_1):
    listlabels = []
    for i in range(Kopt_1):
        listlabels.append(np.where(labelList_5_1 == i)[0])
    plt.figure(1, figsize=(28, Kopt_1+3))
    plt.eventplot(listlabels, lineoffsets=1.01, colors=colorsSeq[:Kopt_1],
              linelengths=1,linewidths=0.7)
    plt.show()  
