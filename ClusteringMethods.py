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

from sklearn import metrics

###For preparation: feature choose 1d or 2d histgoram..
def getHist1D(ConduTraces, cbins = 300, ROI = (-8.0, 0.3), density=False):
 #######bin size seems independant???, need more test...#####
    NumTraces = len(ConduTraces)
    rangeCondu = ROI
    dataHistAll = np.zeros((NumTraces, cbins)).astype(np.float32)###save the all traces' histogram....

    for i in range(NumTraces):
        data_hist_i = np.histogram(ConduTraces[i][:,1], cbins, range = rangeCondu, density=density) 
        if(data_hist_i[0].max() != 0): 
            dataHistAll[i, :] = data_hist_i[0]
        
    dataHistAll_x = data_hist_i[1][1:] - (data_hist_i[1][1] - data_hist_i[1][0])/2
    return dataHistAll, dataHistAll_x


def getHist2D(ConduTraces, cbins2d_x = 100, cbins2d_y = 200, ROI=[[0.1, 2], [-7, -1]], density=False):
    TracesSelection = ConduTraces
    NumTraces = TracesSelection.shape[0]
    dataHistAll2d = np.zeros((NumTraces, cbins2d_x , cbins2d_y))

    for i in range(NumTraces):
        data_hist2d_i = np.histogram2d(TracesSelection[i][:, 0], TracesSelection[i][:, 1], 
                                       (cbins2d_x, cbins2d_y), range=ROI, density=density)
        if(data_hist2d_i[0].max() != 0):
            dataHistAll2d[i] = data_hist2d_i[0] / data_hist2d_i[0].max()


    dataHistAll2d_x = data_hist2d_i[1][1:] + (data_hist2d_i[1][1] - data_hist2d_i[1][0]) / 2
    dataHistAll2d_y = data_hist2d_i[2][1:] + (data_hist2d_i[2][1] - data_hist2d_i[2][0]) / 2
    return dataHistAll2d, dataHistAll2d_x, dataHistAll2d_y


###spectral clustering####
def clusteringBtweenSP(lowlogG, highlogG, dataHistAll_x, dataHistAll, nCluster = 3):

    cbins = np.where((dataHistAll_x >= lowlogG) & (dataHistAll_x <= highlogG))[0]
    
    dataHistAll_6 = dataHistAll[:, cbins]
        
    histcorrcoef =  np.corrcoef(dataHistAll_6)
    histcorrcoef[np.isnan(histcorrcoef)] = 0             ####need warning???
    histcorrcoefNorm_6 = histcorrcoef * 0.5 + 0.5
#   /  nCluster = 6
    labelsList_6 = []
    
    for nCl in range(2, nCluster+1):
        SpectralFun = cluster.SpectralClustering(n_clusters=nCl, random_state=429, 
                                                 n_jobs=-1, affinity='precomputed')
        SpectralFun.fit(histcorrcoefNorm_6)  ###choose 4000 curve 
        labels = SpectralFun.labels_
        labelsList_6.append(labels)
        plt.figure(nCl)
        y1, = plt.plot(dataHistAll_x, dataHistAll.mean(axis=0), 'b', label='Original Data')
        plt.ylabel('Counts per trace')
        plt.xlabel('Conductance /$\mathrm{log(G/G_0)}$')
        plt.title('%0.1f Original data and Cluster %0.1f'%(lowlogG, highlogG) )

        for i in range(nCl):
            nClusterI = (labels == i).sum()
            index_i = np.array(np.where(labels == i)).reshape(-1)
            Cluster_i_hist = dataHistAll[index_i]
            y2, = plt.plot(dataHistAll_x, Cluster_i_hist.mean(axis=0), label='Cluster %d_%d/%0.2f ' %(i+1,nClusterI,nClusterI/len(dataHistAll)))
        plt.legend(bbox_to_anchor = (1.3, 0.5), ncol = 1, loc = 7, borderaxespad=0)
        plt.show()
    return labelsList_6, dataHistAll_6

### data----histogram,  spectral clustering between [lowlogG, highlogG] rigion of interesting(ROI).

def clusteringBtweenSP_M(lowlogG, highlogG, dataHistAll_x, dataHistAll, nCluster = 3): ###Modified...

    cbins = np.where((dataHistAll_x >= lowlogG) & (dataHistAll_x <= highlogG))[0]
    
    dataHistAll_6 = dataHistAll[:, cbins]
    
    zerosRows = np.where(np.all(dataHistAll_6 == 0, axis=1))
    dataHistAll_6 = np.delete(dataHistAll_6,zerosRows, axis=0)
    dataHistAll = np.delete(dataHistAll,zerosRows, axis=0)
#     labels = np.delete(labels, zerosRows)
        
    histcorrcoef =  np.corrcoef(dataHistAll_6)
    histcorrcoef[np.isnan(histcorrcoef)] = 0             ####need warning???
    histcorrcoefNorm_6 = histcorrcoef * 0.5 + 0.5
#   /  nCluster = 6
    labelsList_6 = []
    
    for nCl in range(2, nCluster+1):
        SpectralFun = cluster.SpectralClustering(n_clusters=nCl, random_state=429, 
                                                 n_jobs=-1, affinity='precomputed')
        SpectralFun.fit(histcorrcoefNorm_6)  ###choose 4000 curve 
        labels = SpectralFun.labels_
        labelsList_6.append(labels)
        plt.figure(nCl)
        y1, = plt.plot(dataHistAll_x, dataHistAll.mean(axis=0), 'b', label='Original Data')
        plt.ylabel('Counts per trace')
        plt.xlabel('Conductance /$\mathrm{log(G/G_0)}$')
        plt.title('%0.1f Original data and Cluster %0.1f'%(lowlogG, highlogG) )

        for i in range(nCl):
            nClusterI = (labels == i).sum()
            index_i = np.array(np.where(labels == i)).reshape(-1)
            Cluster_i_hist = dataHistAll[index_i]
            y2, = plt.plot(dataHistAll_x, Cluster_i_hist.mean(axis=0), label='Cluster %d_%d/%d ' %(i+1,nClusterI,nCl))
        plt.legend(bbox_to_anchor = (1.3, 0.5), ncol = 1, loc = 7, borderaxespad=0)
        
    return labelsList_6, dataHistAll_6, zerosRows

def clusteringBtweenKmeans(lowlogG, highlogG,dataHistAll_x, dataHistAll, nCluster = 3):

    cbins = np.where((dataHistAll_x >= lowlogG) & (dataHistAll_x <= highlogG))[0]
    
    dataHistAll_6 = dataHistAll[:, cbins]
        
    labelsList_6 = []
    
    for nCl in range(2, nCluster+1):
        SpectralFun = cluster.KMeans(n_clusters=nCl, random_state=429, 
                                             n_jobs=-1)
        SpectralFun.fit(dataHistAll_6)  ###choose 4000 curve 
        labels = SpectralFun.labels_
        labelsList_6.append(labels)
        plt.figure(nCl)
        y1, = plt.plot(dataHistAll_x, dataHistAll.mean(axis=0), 'b', label='Original Data')
        plt.ylabel('Counts per trace')
        plt.xlabel('Conductance /$\mathrm{log(G/G_0)}$')
        plt.title('%0.1f Kmean Original data and Cluster %0.1f'%(lowlogG, highlogG) )
        for i in range(nCl):
            nClusterI = (labels == i).sum()
            index_i = np.array(np.where(labels == i)).reshape(-1)
            Cluster_i_hist = dataHistAll[index_i]

            y2, = plt.plot(dataHistAll_x, Cluster_i_hist.mean(axis=0), 
                           label='Cluster %d_%d/%d ' %(i+1,nClusterI,nCl))
        plt.legend(bbox_to_anchor = (1.3, 0.5), ncol = 1, loc = 7, borderaxespad=0)
        
    return labelsList_6, dataHistAll_6

def clusteringBtweenSP_2d(dataHistAll2d, nCluster = 3): 
    '''dataHistAll2d must be 2d histogram....
    3d array, [traceIndex, dataDistance, dataConductanc] '''

    dataHistAll_6 = dataHistAll2d.reshape(dataHistAll2d.shape[0], -1)
    histcorrcoef =  np.corrcoef(dataHistAll_6)
    histcorrcoef[np.isnan(histcorrcoef)] = 0             ####need warning???
    histcorrcoefNorm_6 = histcorrcoef * 0.5 + 0.5
#   /  nCluster = 6
    labelsList_6 = []
    
    for nCl in range(2, nCluster+1):
        SpectralFun = cluster.SpectralClustering(n_clusters=nCl, random_state=429, 
                                                 n_jobs=-1, affinity='precomputed')
        SpectralFun.fit(histcorrcoefNorm_6)  ###choose 4000 curve 
        labels = SpectralFun.labels_
        labelsList_6.append(labels)

        
    return labelsList_6, dataHistAll_6

def clusteringBtweenKmeans_2d(dataHistAll2d, nCluster = 3):
    '''dataHistAll2d must be 2d histogram....
    3d array, [traceIndex, dataDistance, dataConductanc] '''
    dataHistAll_6 = dataHistAll2d.reshape(dataHistAll2d.shape[0], -1)
    
       
    labelsList_6 = []
    
    for nCl in range(2, nCluster+1):
        SpectralFun = cluster.KMeans(n_clusters=nCl, random_state=429, 
                                             n_jobs=-1)
        SpectralFun.fit(dataHistAll_6)  ###choose 4000 curve 
        labels = SpectralFun.labels_
        labelsList_6.append(labels)

        
    return labelsList_6, dataHistAll_6

### SVD method !!


def clusteringSVD(lowlogG, highlogG,dataHistAll_x, dataHistAll, nCluster = 6):

    cbins = np.where((dataHistAll_x >= lowlogG) & (dataHistAll_x <= highlogG))[0]
    dataHistAll_6 = dataHistAll[:, cbins]
    histcorrcoef =  np.corrcoef(dataHistAll_6)
    histcorrcoef[np.isnan(histcorrcoef)] = 0             ####need warning???
    histcorrcoefNorm_6 = histcorrcoef * 0.5 + 0.5
    
    Affinity_Matrix = histcorrcoefNorm_6
    D = np.sum(Affinity_Matrix, axis=1)
    L = np.matmul((np.matmul(np.diag(np.sqrt(1/D)), (Affinity_Matrix))), np.diag(np.sqrt(1/D)))
    
    Usvd, Ssvd, Vsvd = np.linalg.svd(L)
    del Ssvd, Vsvd, L, D, histcorrcoef, Affinity_Matrix
    
    labelsList_6 = []
    
    for nCl in range(2, nCluster+1):
        KmeansMethod = cluster.KMeans(n_clusters=nCl, random_state=429, n_jobs=-1)
        KmeansMethod.fit(Usvd[:,:nCl])
        labels = KmeansMethod.labels_
        labelsList_6.append(labels)
        
        plt.figure(nCl)
        y1, = plt.plot(dataHistAll_x, dataHistAll.mean(axis=0), 'b', label='Original Data')
        plt.ylabel('Counts per trace')
        plt.xlabel('Conductance /$\mathrm{log(G/G_0)}$')
        plt.title('%0.1f SVD_SP Original data and Cluster %0.1f'%(lowlogG, highlogG) )
        for i in range(nCl):
            nClusterI = (labels == i).sum()
            index_i = np.array(np.where(labels == i)).reshape(-1)
            Cluster_i_hist = dataHistAll[index_i]

            y2, = plt.plot(dataHistAll_x, Cluster_i_hist.mean(axis=0), 
                           label='Cluster %d_%d/%d ' %(i+1,nClusterI,nCl))
        plt.legend(bbox_to_anchor = (1.3, 0.5), ncol = 1, loc = 7, borderaxespad=0)
        
        
    return labelsList_6,dataHistAll_6

######  modify based on scikit-learning...
import sklearn.metrics.pairwise as skmp
def calinski_harabasz_score_M(X, labels, metric = 'correlation'):
    ####where a higher Calinski-Harabasz score relates to a model with better defined clusters.
    """Compute the Calinski and Harabasz score.
    It is also known as the Variance Ratio Criterion.
    The score is defined as ratio between the within-cluster dispersion and
    the between-cluster dispersion.
    Read more in the :ref:`User Guide <calinski_harabasz_index>`.
    Parameters
    ----------
    X : array-like, shape (``n_samples``, ``n_features``)
        List of ``n_features``-dimensional data points. Each row corresponds
        to a single data point.
    labels : array-like, shape (``n_samples``,)
        Predicted labels for each sample.
    Returns
    -------
    score : float
        The resulting Calinski-Harabasz score.
    References
    ----------
    .. [1] `T. Calinski and J. Harabasz, 1974. "A dendrite method for cluster
       analysis". Communications in Statistics
       <https://www.tandfonline.com/doi/abs/10.1080/03610927408827101>`_
    """
    
    if metric == 'correlation':
        zerosRows = np.where(np.all(X == 0, axis=1))
        X = np.delete(X,zerosRows, axis=0)
        labels = np.delete(labels, zerosRows)
        
    n_samples, _ = X.shape
    n_labels = len(set(labels))

    extra_disp, intra_disp = 0., 0.
    
    mean = np.mean(X, axis=0)
    
    for k in range(n_labels):
        cluster_k = X[labels == k]  ####cluster 
        mean_k = np.mean(cluster_k, axis=0)
        extra_disp += len(cluster_k) *  (skmp.pairwise_distances(np.array(mean_k, ndmin=2),
                                                                np.array(mean, ndmin=2), 
                                                                metric=metric, n_jobs = -1)**2)[0][0]
        
        intra_disp += np.sum(skmp.pairwise_distances(cluster_k, 
                                                   np.array(mean_k, ndmin=2),
                                                   metric=metric, n_jobs = -1)**2)
    return (1. if intra_disp == 0. else
            extra_disp * (n_samples - n_labels) /
            (intra_disp * (n_labels - 1.)))

def davies_bouldin_score_M(X, labels, metric = 'correlation'):   #####
    ###Zero is the lowest possible score. Values closer to zero indicate a better partition.
    """Computes the Davies-Bouldin score.
    The score is defined as the average similarity measure of each cluster with
    its most similar cluster, where similarity is the ratio of within-cluster
    distances to between-cluster distances. Thus, clusters which are farther
    apart and less dispersed will result in a better score.
    The minimum score is zero, with lower values indicating better clustering.
    Read more in the :ref:`User Guide <davies-bouldin_index>`.
    Parameters
    ----------
    X : array-like, shape (``n_samples``, ``n_features``)
        List of ``n_features``-dimensional data points. Each row corresponds
        to a single data point.
    labels : array-like, shape (``n_samples``,)
        Predicted labels for each sample.
    Returns
    -------
    score: float
        The resulting Davies-Bouldin score.
    References
    ----------
    .. [1] Davies, David L.; Bouldin, Donald W. (1979).
       `"A Cluster Separation Measure"
       <https://ieeexplore.ieee.org/document/4766909>`__.
       IEEE Transactions on Pattern Analysis and Machine Intelligence.
       PAMI-1 (2): 224-227
    """
    n_samples, _ = X.shape
    n_labels = len(set(labels))

#     print(n_samples, n_labels)
    intra_dists = np.zeros(n_labels)
    centroids = np.zeros((n_labels, len(X[0])), dtype=np.float)
    for k in range(n_labels):
        cluster_k = X[labels == k]
        centroid = cluster_k.mean(axis=0)
        
      
        centroids[k] = centroid

            
        intra_dists[k] = np.mean(skmp.pairwise_distances(
        cluster_k, [centroid], metric=metric, n_jobs = -1))
        
    centroid_distances = skmp.pairwise_distances(centroids, metric=metric, n_jobs = -1)
            
    if np.allclose(intra_dists, 0) or np.allclose(centroid_distances, 0):
        return 0.0

    centroid_distances[centroid_distances <= 0.00001] = np.inf
    combined_intra_dists = intra_dists[:, None] + intra_dists
    scores = np.max(combined_intra_dists / centroid_distances, axis=1)
    return np.mean(scores)

def Cluster_Indicator_LLC(X, labels, metric = 'correlation'):
    ### average compactness / average separation.??? the lower the better...
    n_samples, _ = X.shape
    n_labels = len(set(labels))

    extra_disp, intra_disp = 0., 0.
    
    DistanceAll = skmp.pairwise_distances(X, metric=metric, n_jobs = -1)**2
    
    similarityInCluster = 0
    
    for k in range(n_labels):
        IndexLabel = np.where(labels == k)[0]
        
        IndexInMatrix = np.ix_(IndexLabel, IndexLabel)
        
        NumWeight = len(IndexLabel) * len(IndexLabel-1) / 2
        
        intra_disp = intra_disp + \
                              np.sum(np.tril(DistanceAll[IndexInMatrix], -1)) / NumWeight
        
        
        IndexLabelOut = np.where(labels != k)[0]
        
        IndexInMatrix2 = np.ix_(IndexLabel, IndexLabelOut)
        
        extra_disp = extra_disp + \
                              np.mean((DistanceAll[IndexInMatrix2]))  
        
    return intra_disp / extra_disp

# The more dissimilar two objects are, the farther they would
# be from each other
def VNC(X,labels, metric='correlation'):  ###S. Zhou, Z. Xu / Applied Soft Computing 71 (2018) 78–88
    n_samples, NumFeature = X.shape
    n_labels = len(set(labels))
    mu_k = np.ndarray((n_labels, NumFeature))
    bd = np.ones((n_labels)) * np.inf
    wd = np.ones((n_labels))
    VCN = np.ones((n_labels))
#     distance = np.ones((n_labels-1))
    for k in range(n_labels):
        cluster_k = X[labels == k]  ####cluster 
        mean_k = np.mean(cluster_k, axis=0)  ##(1) mu_k,,,(11)
        mu_k[k] = mean_k
        
        for j  in range(n_labels):
            if k == j:
                continue
            cluster_j = X[labels == j]
            distance = np.mean(skmp.pairwise_distances(cluster_j, np.array(mean_k, ndmin=2), 
                                                       metric, n_jobs = -1))
#             bd[k] = distance if distance < bd[k] else bd[k]
            bd[k] = np.min([distance, bd[k]])
    
        wd[k] = np.mean(skmp.pairwise_distances(cluster_k, np.array(mean_k, ndmin=2), 
                                               metric, n_jobs = -1))
        VCN[k] = (bd[k]-wd[k]) / np.max([bd[k], wd[k]])
    return np.mean(VCN)

def plot_validity_index(LLC_Index, davies_bouldin_Index, calinski_harabasz_Index,
                       Silhouette_Index):
    
    NumCluster = len(LLC_Index)+2
    rangx = range(2, NumCluster)
    plt.figure(figsize=(16, 12))
    ax = plt.subplot(221)
    y2, = ax.plot(rangx, LLC_Index, 'r', label='VCN score')
    plt.legend()
    plt.ylabel('VCN score ')
    plt.xlabel('Clusters Number')
    plt.title('VCN score, higher better' )

    ax2 = plt.subplot(222)
    y2, = plt.plot(rangx, davies_bouldin_Index, 'r', label='davies bouldin score')
    plt.legend()
    plt.ylabel('davies bouldin score ')
    plt.xlabel('Clusters Number')
    plt.title('davies bouldin score, lower better' )

    ax3 = plt.subplot(223)
    y2, = plt.plot(rangx, calinski_harabasz_Index, 'r', label='Calinski-Harabasz')
    plt.legend()
    plt.ylabel('Calinski-Harabasz score ')
    plt.xlabel('Clusters Number')
    plt.title('Calinski-Harabasz Index, higher better' )

    ax4 = plt.subplot(224)
    y1, = plt.plot(rangx, Silhouette_Index, 'r', label='silhouette score')
    plt.legend()
    plt.ylabel('silhouette score')
    plt.xlabel('Clusters Number')
    plt.title('silhouette score, higher better' )
    
    plt.tight_layout()

def ComputeIndex(dataHistAll_6, labelsList, functionIndex, metric='euclidean'):
    numClusters = len(labelsList)
    LLC_Index = np.ndarray(numClusters)
    for i in range(numClusters):
        LLC_Index[i] = functionIndex(dataHistAll_6, labelsList[i], metric)
    return LLC_Index

def plotIndex(labelsList_7_1_Kdesity, metric='euclidean'):
    LLC_Index = ComputeIndex(labelsList_7_1_Kdesity[1], labelsList_7_1_Kdesity[0], 
                             VNC, metric)

    davies_bouldin_Index = ComputeIndex(labelsList_7_1_Kdesity[1], labelsList_7_1_Kdesity[0], 
                                        davies_bouldin_score_M, metric)

    calinski_harabasz_Index = ComputeIndex(labelsList_7_1_Kdesity[1], labelsList_7_1_Kdesity[0], 
                                           calinski_harabasz_score_M, metric)

    Silhouette_Index = ComputeIndex(labelsList_7_1_Kdesity[1], labelsList_7_1_Kdesity[0],
                                    metrics.silhouette_score, metric)

    plot_validity_index(LLC_Index, davies_bouldin_Index, calinski_harabasz_Index,
                           Silhouette_Index)
    return LLC_Index, davies_bouldin_Index, calinski_harabasz_Index, Silhouette_Index

def plotCHI(labelsList_7_2, metric='correlation'):
    
    labelsList = labelsList_7_2[0]
    numClusters = len(labelsList)
    calinski_harabasz_Index = np.ndarray(numClusters)
    rangx = range(2,numClusters+2)
    for i in range(numClusters):
        calinski_harabasz_Index[i] = calinski_harabasz_score_M(labelsList_7_2[1],
                                                               labelsList[i], metric=metric)

    y2, = plt.plot(rangx, calinski_harabasz_Index, 'r', label='Calinski-Harabasz')
    plt.legend()
    plt.ylabel('Calinski-Harabasz score ')
    plt.xlabel('Clusters Number')
    plt.title('Calinski-Harabasz Index, higher better' )
    plt.xticks(rangx)
    return calinski_harabasz_Index



def Distance(hist1, hist2):
    Acoef = np.corrcoef(hist1, hist2)
    return 1 - (Acoef[1, 0] * 0.5 + 0.5)

def calinski_harabasz_score(X, labels, Distance=Distance):
    ####where a higher Calinski-Harabasz score relates to a model with better defined clusters.
    """Compute the Calinski and Harabasz score.
    It is also known as the Variance Ratio Criterion.
    The score is defined as ratio between the within-cluster dispersion and
    the between-cluster dispersion.
    Read more in the :ref:`User Guide <calinski_harabasz_index>`.
    Parameters
    ----------
    X : array-like, shape (``n_samples``, ``n_features``)
        List of ``n_features``-dimensional data points. Each row corresponds
        to a single data point.
    labels : array-like, shape (``n_samples``,)
        Predicted labels for each sample.
    Returns
    -------
    score : float
        The resulting Calinski-Harabasz score.
    References
    ----------
    .. [1] `T. Calinski and J. Harabasz, 1974. "A dendrite method for cluster
       analysis". Communications in Statistics
       <https://www.tandfonline.com/doi/abs/10.1080/03610927408827101>`_
    """
#     X, labels = check_X_y(X, labels)
#     le = LabelEncoder()
#     labels = le.fit_transform(labels)

    n_samples, _ = X.shape
    n_labels = len(set(labels))

#     print(n_samples, n_labels)
#     check_number_of_labels(n_labels, n_samples)

    extra_disp, intra_disp = 0., 0.
    
    mean = np.mean(X, axis=0)
    
    for k in range(n_labels):
        cluster_k = X[labels == k]  ####cluster 
        mean_k = np.mean(cluster_k, axis=0)
        extra_disp += len(cluster_k) *  Distance(mean_k, mean)**2 ##np.sum((mean_k - mean) ** 2)
        for cluster_in_k in cluster_k:
            intra_disp +=  Distance(cluster_in_k, mean_k)**2 ##np.sum((cluster_k - mean_k) ** 2) 
        ###change the distance

    return (1. if intra_disp == 0. else
            extra_disp * (n_samples - n_labels) /
            (intra_disp * (n_labels - 1.)))
# rangx = range(2, 18)
# plt.plot(range(2, 18), CHI396/np.max(CHI396), 'r', label='CH Index')
# plt.plot(range(2, 18), 1-DBI396/np.max(DBI396), 'b', label = 'DB Index')
# plt.plot(range(2, 18), SILI396/np.max(SILI396), 'black', label = 'Sil')
# plt.plot(range(2, 18), VNCI396/np.max(VNCI396), 'cyan', label='VCN')
# plt.xlabel("Cluster number")
# plt.ylabel("Score")
# plt.legend(loc=1)
# plt.xticks(rangx)
# plt.show()

#  VNCI396,DBI396, CHI396, SILI396 = plotIndex(labelsList_7_2, 'correlation')

#### kmeans clustering with users' defined distance
# for test 
# import numpy as np
import random
random.seed(429)
def cluster_points(X, mu, DistanceFun):
    clusters  = {}
    labels = np.zeros(len(X)).astype(int)
    ilabels = 0
    for x in X:
        bestmukey = min([(i[0], DistanceFun(x, mu[i[0]])) \
                    for i in enumerate(mu)], key=lambda t:t[1])[0]
        try:
            clusters[bestmukey].append(x)
#             labels[bestmukey].append(ilabels)
        except KeyError:
            clusters[bestmukey] = [x]
#             labels[bestmukey] = [ilabels]
        labels[ilabels] = np.int(bestmukey)
        ilabels += 1
    return clusters, labels

 
def reevaluate_centers(mu, clusters):
    newmu = []
    keys = sorted(clusters.keys())
    for k in keys:
        newmu.append(np.mean(clusters[k], axis = 0))
    return newmu

def has_converged(mu, oldmu):
    return set([tuple(a) for a in mu]) == set([tuple(a) for a in oldmu])

def find_centers(X, K, DistanceFun, maxiter = 100):
    # Initialize to K random centers
    oldmu = random.sample(list(X), K)
    mu = random.sample(list(X), K)
    iterNum = 0
    while not has_converged(mu, oldmu):
        oldmu = mu
        # Assign all points in X to clusters
        clusters, labels = cluster_points(X, mu, DistanceFun)
        # Reevaluate centers
        mu = reevaluate_centers(oldmu, clusters)
        iterNum += 1
        if iterNum > maxiter:
            break
    return(mu, labels)



###new clustering distance....
from sklearn import preprocessing
# 8, 16, 32, 64, 128 ###5 scale;;
def getNewHist(ConduTraces,  distRange, ConduRange, cbins = 10, scaleList=[8, 16, 32, 64, 128]):
    sumList = sum(scaleList)
    NumTraces = len(ConduTraces)
    dataHistAllAug = np.zeros((NumTraces, cbins*sumList)).astype(np.float32)
    dataHistAllAugScale = np.zeros((NumTraces, cbins*sumList)).astype(np.float32)
    j = 0

    xbinsAll = [np.linspace(distRange[0], distRange[1], n+1) for n in scaleList]
    
    for ConduTrace in ConduTraces:
        for n in range(len(scaleList)):
            it = 0
            xbins = xbinsAll[n]
            for ibins in range(len(xbins)-1):
                indexTmp = np.where((ConduTrace[:, 0] >= xbins[ibins]) & (ConduTrace[:, 0] <= xbins[ibins+1]))[0]

                data_hist_ibins = np.histogram(ConduTrace[indexTmp, 1], cbins, range = ConduRange, density=None)

                dataHistAllAug[j, cbins*it:cbins*(it+1)] = data_hist_ibins[0] #####may need scaling???
                dataHistAllAugScale[j, cbins*it:cbins*(it+1)] = preprocessing.scale(data_hist_ibins[0].astype(np.float64))
                it += 1
        j += 1
    return dataHistAllAug, dataHistAllAugScale