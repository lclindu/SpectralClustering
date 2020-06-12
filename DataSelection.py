import numpy as np
def TracesSelect(hightlogG, lowlogG,ConduTraces, nanoGapLimit=0.1):
    
    TracesSelection = []

    delTraceIndex=[]
    for i, trace in enumerate(ConduTraces):
        start_Index = np.where(trace[:, 1] < hightlogG)[0] ##first, make sure exist...


        if start_Index.shape[0] < 1:
            delTraceIndex.append(i)
            continue
        end_Index = np.where(trace[:, 1] >= lowlogG)[0] ###last, make sure exist
        if end_Index.shape[0] < 1:
            delTraceIndex.append(i)
            continue

        start_Index_I = np.where(trace[:, 1] >= hightlogG)[0] ##last

        start = start_Index[0]
        if start_Index_I.shape[0] > 1:
            if trace[start_Index_I[-1],0] < nanoGapLimit:
                start = max(start_Index[0], start_Index_I[-1])

        end_Index_I = np.where(trace[:, 1] < lowlogG)[0]  ###first

        end = end_Index[-1]
        if end_Index_I.shape[0] > 1:
            if trace[end_Index_I[0], 0] > nanoGapLimit:
                end = min(end_Index[-1], end_Index_I[0])

        if trace[end, 0] - trace[start, 0] < nanoGapLimit:
            delTraceIndex.append(i)
            continue

        TracesSelection.append(trace[start:end, :])
        
    return np.array(TracesSelection), np.array(delTraceIndex)
#     next()

def saveTrace(label_junc_or_not, dataCut,filename_data):
    labelsmax = len(set(label_junc_or_not))
    j=0
    for i in set(label_junc_or_not):

        index_i = np.array(np.where(label_junc_or_not == i)).reshape(-1)
        
        ClusterI_Cut = dataCut[index_i]
        
        ClusterI = np.vstack((x for x in ClusterI_Cut))
        
        np.savetxt(filename_data[:-4] + 'LLC_Cluster%d_%d_%d.txt'
                   %(len(index_i),labelsmax, i), ClusterI, fmt="%0.3f")
        del ClusterI

def saveTraceHist(label_junc_or_not, dataHistAll, dataHistAll_x, filename_data):
    labelsmax = len(set(label_junc_or_not))
    j=0
    NumTraces = len(dataHistAll_x)
    ClusterI = np.vstack((dataHistAll_x, dataHistAll.mean(axis=0))).T
    np.savetxt(filename_data[:-4] + 'ClusterHistAllCurve%d.txt'%NumTraces
                   , ClusterI, fmt="%0.5f")
    
    for i in set(label_junc_or_not):

        index_i = np.array(np.where(label_junc_or_not == i)).reshape(-1)
        
        Cluster_i_hist = dataHistAll[index_i]
        
        ClusterI = np.vstack((dataHistAll_x, Cluster_i_hist.mean(axis=0))).T
        np.savetxt(filename_data[:-4] + 'ClusterHist_%d_%dCurve%d.txt'
                   %(labelsmax, i+1, index_i.shape[0]), ClusterI, fmt="%0.5f")
        del ClusterI


        
def meanlize(dataCut, step, window=1):
    NumTraces = len(dataCut)
    dataMean = []
    halfStep = np.int(step/2)
    for curve in dataCut:
        NumPoints = len(curve)
        dataMean.append(np.array([np.mean(curve[i:i+step], axis=0) for i in range(halfStep, NumPoints-halfStep, window)]))
    return np.array(dataMean)

def RelatvieDist(DataSel):
    '''Change the DataSel actually..'''
    for Curve in DataSel:
        Curve[:, 0] = Curve[:, 0]-Curve[0, 0]
    return DataSel
