import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.mixture import GaussianMixture as GMM
from sklearn.cluster import KMeans
import os
from sklearn import metrics

#The program to calcuate Dunn index is used by the following source.
#https://gist.github.com/douglasrizzo/cd7e792ff3a2dcaf27f6?permalink_comment_id=2914816

DIAMETER_METHODS = ['mean_cluster', 'farthest']
CLUSTER_DISTANCE_METHODS = ['nearest', 'farthest']

def inter_cluster_distances(labels, distances, method='nearest'):
    """Calculates the distances between the two nearest points of each cluster.
    :param labels: a list containing cluster labels for each of the n elements
    :param distances: an n x n numpy.array containing the pairwise distances between elements
    :param method: `nearest` for the distances between the two nearest points in each cluster, or `farthest`
    """
    if method not in CLUSTER_DISTANCE_METHODS:
        raise ValueError(
            'method must be one of {}'.format(CLUSTER_DISTANCE_METHODS))

    if method == 'nearest':
        return __cluster_distances_by_points(labels, distances)
    elif method == 'farthest':
        return __cluster_distances_by_points(labels, distances, farthest=True)


def __cluster_distances_by_points(labels, distances, farthest=False):
    n_unique_labels = len(np.unique(labels))
    cluster_distances = np.full((n_unique_labels, n_unique_labels),
                                float('inf') if not farthest else 0)

    np.fill_diagonal(cluster_distances, 0)

    for i in np.arange(0, len(labels) - 1):
        for ii in np.arange(i, len(labels)):
            if labels[i] != labels[ii] and (
                (not farthest and
                 distances[i, ii] < cluster_distances[labels[i], labels[ii]])
                    or
                (farthest and
                 distances[i, ii] > cluster_distances[labels[i], labels[ii]])):
                cluster_distances[labels[i], labels[ii]] = cluster_distances[
                    labels[ii], labels[i]] = distances[i, ii]
    return cluster_distances


def diameter(labels, distances, method='farthest'):
    """Calculates cluster diameters
    :param labels: a list containing cluster labels for each of the n elements
    :param distances: an n x n numpy.array containing the pairwise distances between elements
    :param method: either `mean_cluster` for the mean distance between all elements in each cluster, or `farthest` for the distance between the two points furthest from each other
    """
    if method not in DIAMETER_METHODS:
        raise ValueError('method must be one of {}'.format(DIAMETER_METHODS))

    n_clusters = len(np.unique(labels))
    diameters = np.zeros(n_clusters)

    if method == 'mean_cluster':
        for i in range(0, len(labels) - 1):
            for ii in range(i + 1, len(labels)):
                if labels[i] == labels[ii]:
                    diameters[labels[i]] += distances[i, ii]

        for i in range(len(diameters)):
            diameters[i] /= sum(labels == i)

    elif method == 'farthest':
        for i in range(0, len(labels) - 1):
            for ii in range(i + 1, len(labels)):
                if labels[i] == labels[ii] and distances[i, ii] > diameters[
                        labels[i]]:
                    diameters[labels[i]] = distances[i, ii]
    return diameters


def dunn(labels, distances, diameter_method='farthest',
         cdist_method='nearest'):
    """
    Dunn index for cluster validation (larger is better).
    
    .. math:: D = \\min_{i = 1 \\ldots n_c; j = i + 1\ldots n_c} \\left\\lbrace \\frac{d \\left( c_i,c_j \\right)}{\\max_{k = 1 \\ldots n_c} \\left(diam \\left(c_k \\right) \\right)} \\right\\rbrace
    
    where :math:`d(c_i,c_j)` represents the distance between
    clusters :math:`c_i` and :math:`c_j`, and :math:`diam(c_k)` is the diameter of cluster :math:`c_k`.
    Inter-cluster distance can be defined in many ways, such as the distance between cluster centroids or between their closest elements. Cluster diameter can be defined as the mean distance between all elements in the cluster, between all elements to the cluster centroid, or as the distance between the two furthest elements.
    The higher the value of the resulting Dunn index, the better the clustering
    result is considered, since higher values indicate that clusters are
    compact (small :math:`diam(c_k)`) and far apart (large :math:`d \\left( c_i,c_j \\right)`).
    :param labels: a list containing cluster labels for each of the n elements
    :param distances: an n x n numpy.array containing the pairwise distances between elements
    :param diameter_method: see :py:function:`diameter` `method` parameter
    :param cdist_method: see :py:function:`diameter` `method` parameter
    
    .. [Kovacs2005] Kovács, F., Legány, C., & Babos, A. (2005). Cluster validity measurement techniques. 6th International Symposium of Hungarian Researchers on Computational Intelligence.
    """

    labels = LabelEncoder().fit(labels).transform(labels)

    ic_distances = inter_cluster_distances(labels, distances, cdist_method)
    min_distance = min(ic_distances[ic_distances.nonzero()])
    max_diameter = max(diameter(labels, distances, diameter_method))

    return min_distance / max_diameter


#dataPath = ['ORB_features.npy', 'SIFT_features.npy']
#FeatureImgName = ['ORB_FeaturesImgName.npy', 'SIFT_FeaturesImgName.npy']
#dataPath = ['ORB_featuresNoShuffle.npy', 'SIFT_featuresNoShuffle.npy']
#FeatureImgName = ['ORB_FeaturesImgNameNoShuffle.npy', 'SIFT_FeaturesImgNameNoShuffle.npy']
#models = ['ORB','SIFT']
dataPath = ['ORB_250Features.npy']
FeatureImgName = ['ORB_250FeaturesImgName.npy']
models = ['ORB']
for p in range(len(dataPath)):
    #Input data
    data = np.load(dataPath[p], allow_pickle=True)
    #data = data.reshape((len(data), len(data[0])))
    imgName = np.load(FeatureImgName[p])
    #Normailize Min-max
    scaler = MinMaxScaler() ###
    scaler.fit(data) ###
    NorData = scaler.transform(data) ###
    #NorData = data ###
    
    minNb = 2
    maxNb = 9
    dunnScores = []
    folderName = models[p]+'_Res250Features'
    if not os.path.exists(folderName):
        os.makedirs(folderName)
    for Km in range(minNb,maxNb):
        clusterRes = {}
        plt.cla()
        plt.clf()
        #Cluster learning by GMM
        #gmm = GMM(n_components=Km).fit(Tdata)
        gmm = KMeans(n_clusters=Km, random_state=None).fit(NorData)
        labels = 0
        labels = gmm.predict(NorData)
        dis = euclidean_distances(NorData)
        dunnIdx = dunn(labels, dis)
        dunnScores.append(dunnIdx)
        print('Number of clusers:', Km)
        print('Dunn Index:', dunnIdx)
        InvData = scaler.inverse_transform(NorData) ###
        #InvData = NorData ###
        plt.scatter(InvData[:, 0], InvData[:, 1], c=labels, s=40, cmap='viridis');
        plt.savefig('./'+folderName+'/'+str(Km)+'.png')
        #plt.show()
        clusterRes['Name']=imgName
        clusterRes['Label']=labels
        pd.DataFrame(clusterRes).to_excel('./'+folderName+'/'+models[p]+'_'+str(Km)+".xlsx") 
        
    plt.cla()
    plt.clf()
    plt.plot(range(minNb,maxNb), dunnScores, 'go-', linewidth=2)
    plt.xlabel("Number of clusters")
    plt.ylabel("Dunn Index")
    plt.savefig('./'+folderName+'/Dunn Index.png')
    
    indxRes = {}
    indxRes['Dunn Index']=dunnScores
    pd.DataFrame(indxRes).to_excel('./'+folderName+'/'+models[p]+'_IndexRes.xlsx')
    #plt.show()

