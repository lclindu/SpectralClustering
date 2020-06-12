import numpy as np
#from numpy.core.umath_tests import inner1d

A = np.array([[1,2],[3,4],[5,6],[7,8]])
B = np.array([[2,3],[4,5],[6,7],[8,9],[10,11]])

def inner1d(A, B):
    return np.einsum('ij, ij->i', A, B)

# Hausdorff Distance
def HausdorffDist(A,B):
    # Hausdorf Distance: Compute the Hausdorff distance between two point
    # clouds.
    # Let A and B be subsets of metric space (Z,dZ),
    # The Hausdorff distance between A and B, denoted by dH(A,B),
    # is defined by:
    # dH(A,B) = max(h(A,B),h(B,A)),
    # where h(A,B) = max(min(d(a,b))
    # and d(a,b) is a L2 norm
    # dist_H = hausdorff(A,B)
    # A: First point sets (MxN, with M observations in N dimension)
    # B: Second point sets (MxN, with M observations in N dimension)
    # ** A and B may have different number of rows, but must have the same
    # number of columns.
    #
    # Edward DongBo Cui; Stanford University; 06/17/2014

    # Find pairwise distance
    D_mat = np.sqrt(inner1d(A,A)[np.newaxis].T + inner1d(B,B)-2*(np.dot(A,B.T)))
    # Find DH
    dH = np.max(np.array([np.max(np.min(D_mat,axis=0)),np.max(np.min(D_mat,axis=1))]))
    return(dH)

def ModHausdorffDist(A,B):
    #This function computes the Modified Hausdorff Distance (MHD) which is
    #proven to function better than the directed HD as per Dubuisson et al.
    #in the following work:
    #
    #M. P. Dubuisson and A. K. Jain. A Modified Hausdorff distance for object
    #matching. In ICPR94, pages A:566-568, Jerusalem, Israel, 1994.
    #http://ieeexplore.ieee.org/xpls/abs_all.jsp?arnumber=576361
    #
    #The function computed the forward and reverse distances and outputs the
    #maximum/minimum of both.
    #Optionally, the function can return forward and reverse distance.
    #
    #Format for calling function:
    #
    #[MHD,FHD,RHD] = ModHausdorffDist(A,B);
    #
    #where
    #MHD = Modified Hausdorff Distance.
    #FHD = Forward Hausdorff Distance: minimum distance from all points of B
    #      to a point in A, averaged for all A
    #RHD = Reverse Hausdorff Distance: minimum distance from all points of A
    #      to a point in B, averaged for all B
    #A -> Point set 1, [row as observations, and col as dimensions]
    #B -> Point set 2, [row as observations, and col as dimensions]
    #
    #No. of samples of each point set may be different but the dimension of
    #the points must be the same.
    #
    #Edward DongBo Cui Stanford University; 06/17/2014

    # Find pairwise distance
    D_mat = np.sqrt(inner1d(A,A)[np.newaxis].T + inner1d(B,B)-2*(np.dot(A,B.T)))
    # Calculating the forward HD: mean(min(each col))
    FHD = np.mean(np.min(D_mat,axis=1))
    # Calculating the reverse HD: mean(min(each row))
    RHD = np.mean(np.min(D_mat,axis=0))
    # Calculating mhd
    MHD = np.max(np.array([FHD, RHD]))
    return(MHD, FHD, RHD)



import math
import numpy as np

# Euclidean distance.
def euc_dist(pt1,pt2):
    return math.sqrt((pt2[0]-pt1[0])*(pt2[0]-pt1[0])+(pt2[1]-pt1[1])*(pt2[1]-pt1[1]))

def _c(ca,i,j,P,Q):
    if ca[i,j] > -1:
        return ca[i,j]
    elif i == 0 and j == 0:
        ca[i,j] = euc_dist(P[0],Q[0])
    elif i > 0 and j == 0:
        ca[i,j] = max(_c(ca,i-1,0,P,Q),euc_dist(P[i],Q[0]))
    elif i == 0 and j > 0:
        ca[i,j] = max(_c(ca,0,j-1,P,Q),euc_dist(P[0],Q[j]))
    elif i > 0 and j > 0:
        ca[i,j] = max(min(_c(ca,i-1,j,P,Q),_c(ca,i-1,j-1,P,Q),_c(ca,i,j-1,P,Q)),euc_dist(P[i],Q[j]))
    else:
        ca[i,j] = float("inf")
    return ca[i,j]

"""
Computes the discrete frechet distance between two polygonal lines
Algorithm: http://www.kr.tuwien.ac.at/staff/eiter/et-archive/cdtr9464.pdf
P and Q are arrays of 2-element arrays (points)
"""
def frechetDist(P,Q):
    ca = np.ones((len(P),len(Q)))
    ca = np.multiply(ca,-1)
    return _c(ca,len(P)-1,len(Q)-1,P,Q)





def DTWDistance(s1, s2):
    DTW={}
    
    for i in range(len(s1)):
        DTW[(i, -1)] = float('inf')
    for i in range(len(s2)):
        DTW[(-1, i)] = float('inf')
    DTW[(-1, -1)] = 0

    for i in range(len(s1)):
        for j in range(len(s2)):
            dist= (s1[i]-s2[j])**2
            DTW[(i, j)] = dist + min(DTW[(i-1, j)],DTW[(i, j-1)], DTW[(i-1, j-1)])
    return np.sqrt(DTW[len(s1)-1, len(s2)-1])

def DTWDistance_fast(s1, s2,w):
    DTW={}
    
    w = max(w, abs(len(s1)-len(s2)))
    
    for i in range(-1,len(s1)):
        for j in range(-1,len(s2)):
            DTW[(i, j)] = float('inf')
    DTW[(-1, -1)] = 0
  
    for i in range(len(s1)):
        for j in range(max(0, i-w), min(len(s2), i+w)):
            dist= (s1[i]-s2[j])**2
            DTW[(i, j)] = dist + min(DTW[(i-1, j)],DTW[(i, j-1)], DTW[(i-1, j-1)])
		
    return np.sqrt(DTW[len(s1)-1, len(s2)-1])

def LB_Keogh(s1,s2,r):  ###new DTW distance
    LB_sum=0
    for ind,i in enumerate(s1):
        
        lower_bound=min(s2[(ind-r if ind-r>=0 else 0):(ind+r)])
        upper_bound=max(s2[(ind-r if ind-r>=0 else 0):(ind+r)])
        
        if i>upper_bound:
            LB_sum=LB_sum+(i-upper_bound)**2
        elif i<lower_bound:
            LB_sum=LB_sum+(i-lower_bound)**2
    
    return np.sqrt(LB_sum)

###kmeans  for DTW ditances...
import random

def k_means_clust(data,num_clust,num_iter,w=5):
    centroids = random.sample(list(data),num_clust)
    counter = 0
    for n in range(num_iter):
        counter += 1
        print (counter)
        assignments = {}
        #assign data points to clusters
        for ind,i in enumerate(data):
            min_dist = float('inf')
            closest_clust = None
            #assign each data point to the closest cluster
            for c_ind,j in enumerate(centroids):
                if LB_Keogh(i,j,5) < min_dist:
                    cur_dist = DTWDistance(i,j,w)
                    if cur_dist<min_dist:
                        min_dist=cur_dist
                        closest_clust=c_ind
            if closest_clust in assignments:
                assignments[closest_clust].append(ind)
            else:
                assignments[closest_clust]=[]
    
        #recalculate centroids of clusters or update centroids 
        for key in assignments:
            clust_sum=0
            for k in assignments[key]:
                clust_sum=clust_sum+data[k]
            centroids[key]=[m/len(assignments[key]) for m in clust_sum]
    
    return centroids, assignments

####for any distance kmeans
 
def cluster_points(X, mu, DistanceFun):
    clusters  = {}
    for x in X:
        bestmukey = min([(i[0], DistanceFun(x, mu[i[0]])) \
                    for i in enumerate(mu)], key=lambda t:t[1])[0]
        try:
            clusters[bestmukey].append(x)
        except KeyError:
            clusters[bestmukey] = [x]
    return clusters
 
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
        clusters = cluster_points(X, mu, DistanceFun)
        # Reevaluate centers
        mu = reevaluate_centers(oldmu, clusters)
        iterNum += 1
        if iterNum > maxiter:
            break
    return(mu, clusters)


def FrechetDist(ca,DistMatrix, i, j): ###DistMatrix  === parewise distance matrix.
    if ca[i, j] > -1:
        return ca[i, j]
    elif i == 0 and j == 0:
        ca[i, j] = DistMatrix[i, j]
    elif i > 0 and j == 0:
        ca[i, j] = max(DistMatrix[i, 0], FrechetDist(ca, DistMatrix, i-1, 0))
    elif i == 0 and j > 0:
        ca[i, j] = max(DistMatrix[0, j], FrechetDist(ca, DistMatrix, 0, j-1))
    elif i > 0 and j > 0:
        ca[i,j] = max(min(FrechetDist(ca, DistMatrix, i-1, j), FrechetDist(ca, DistMatrix, i-1, j-1), 
                         FrechetDist(ca, DistMatrix, i, j-1)), DistMatrix[i, j])
    else:
        ca[i, j] = float('inf')
    return ca[i, j]


def FrechetDistanc_L(P, Q):
    ca = np.ones((len(P), len(Q)))
    ca = -ca
    DistMaxtrix = skmp.euclidean_distances(P, Q)
    return FrechetDist(ca, DistMaxtrix, len(P)-1, len(Q)-1)

###for example
import sklearn.metrics.pairwise as skmp

def DistanceFun(x, y):
    x = np.array(x, ndmin=2)
    y = np.array(y, ndmin=2)
    return skmp.pairwise_distances(x,y, metric='correlation')

def ManhattanDist(x,y):
    '''曼哈顿距离(Manhattan Distance)'''
    x = np.array(x, ndmin=2)
    y = np.array(y, ndmin=2)
    return skmp.pairwise_distances(x,y, metric='cityblock')

def seuclideanDist(x,y):
    x = np.array(x, ndmin=2)
    y = np.array(y, ndmin=2)
    return skmp.pairwise_distances(x,y, metric='seuclidean')    

def HellingerDist(p, q):
   ''' Hellinger distance'''
   return np.sqrt(1-np.sum(np.sqrt(p*q)))

def BusDist(p,q):
    '''巴氏距离（Bhattacharyya Distance）'''
    BC=np.sum(np.sqrt(p*q))
    b=-np.log(BC)
    return b

def DistanceFunCorre(x, y):
    return np.abs(np.argmax(np.correlate(x, y, 'full')) - (len(y)-1))

def FindConst(zeroAll):
    index = []
    for i in range(len(zeroAll)):
        if(np.all(zeroAll[i] - zeroAll[i].mean() == 0) ):
            index.append(i)
    return np.array(index)
