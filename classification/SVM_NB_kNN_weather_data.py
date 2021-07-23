#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 20:02:26 2020

@author: a
"""

#Classification of weather data
#Using support vector machines (SVM, naive Bayes (NB, and k-nearest neighbours (kNN)
#Includes evaluation of methods used.

## Libraries

import numpy as np
import scipy as sp
import scipy.cluster as sc
import scipy.linalg

from sklearn import metrics, svm
from sklearn.cluster import MeanShift, estimate_bandwidth, AffinityPropagation
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import warnings

## Functions

def outlier_removal(data, threshold):
    tmp = data.copy()
    outlier_flag = [False]*data.shape[0]
    
    for j in range(data.shape[1]):
        meanVal = np.mean(data[:, j])
        stdevVal = np.std(data[:, j])
        maxVal = meanVal + threshold*stdevVal
        minVal = meanVal - threshold*stdevVal
        
        for i in range(data.shape[0]):
            if data[i, j] < minVal or data[i, j] > maxVal:
                outlier_flag[i] = True
            
    outliers_removed = [tmp[i] for i in range(len(outlier_flag)) if not outlier_flag[i]]
    outliers_removed = np.asarray(outliers_removed)
    return outliers_removed

def normalise(data):
    normData = data.copy()

    for j in range(data.shape[1]):
        maxElement = np.amax(data[:,j])
        minElement = np.amin(data[:,j])

        for i in range(data.shape[0]):
            normData[i,j] = (data[i,j] - minElement) / (maxElement - minElement)

    return normData

def centralise(data):
    centralisedData = data.copy()
    
    for j in range(data.shape[1]):
        mean = np.mean(data[:,j])
        
        for i in range(data.shape[0]):
            centralisedData[i,j] = (data[i,j] - mean)
            
    return centralisedData

def distance(p1, p2): #calculates Euclidean distance
    return np.linalg.norm(p1 - p2)
            
def loadData(filePath):
    dataRaw = []
    DataFile = open(filePath, "r")

    while True:
        theline = DataFile.readline()

        if len(theline) == 0:
            break

        theline = theline.rstrip()

        readData = theline.split(",")
        for i in range(len(readData)):
            readData[i] = float(readData[i]);

        dataRaw.append(readData)

    DataFile.close()

    data = np.array(dataRaw)

    return data

# to make nC2  plots where n = number of features:
# (can include centroids in case of K-means plots)
def plotter(data, title, centroids = []):

    tmp = [] #selects principal component axes to plot against
    for i in range(data.shape[1]):
        for j in range(i+1, data.shape[1]):
            if (j < data.shape[1]):
                tmp.append([i, j])
    
    cmap = cm.get_cmap("Spectral")
    cmap2 = cm.get_cmap("Paired")
    colours = cmap(np.linspace(0, 1, num = len(tmp)))
    
    #checking if a centroids parameter was included:
    if not np.array_equal(centroids, []): 
        #colours of centroids:
        cc = cmap2(np.linspace(0, 1, num = len(centroids)))
    
    for i in range(len(tmp)):
        
        plt.plot(data[:,tmp[i][0]], data[:,tmp[i][1]],
                 marker = ".", color = colours[i], linestyle = "None")
        
        if not np.array_equal(centroids, []):
            for j in range(len(centroids)):
                plt.plot(centroids[j,tmp[i][0]],centroids[j,tmp[i][1]],
                         marker = "x", ms = 15, mew = 4, color = cc[j])
        
        plt.title("{}".format(title))
        plt.xlabel("Principal Component: {}".format(tmp[i][0] + 1))
        plt.ylabel("Principal Component: {}".format(tmp[i][1] + 1))
        plt.show()
        plt.close()

# used to make silhouette plots for k-means clustering:
def silhouette_maker(data, n_clusters):

    for k in range(2, n_clusters + 1): #k is the number of clusters
    
        #creating two subplots, the silhouette plot and the plot of the cluster
        plt.figure(figsize = (6,4))
        ax = plt.axes()
        
        #setting x-axis limits for silhouette plot
        ax.set_xlim([-0.2, 1])
        #setting y-axis limits for silhouette plot, with a margin
        ax.set_ylim([0, data.shape[0] + (k+1)*10])
        
        #kmeans2 is preferred for the ease of accessing labels
        centroids, labels = sc.vq.kmeans2(data, k)
        
        #calculating the silhouette score
        silhouette_avg = metrics.silhouette_score(data, labels)
        print("For {} clusters, the average silhouette score is {}.".format\
              (k, silhouette_avg))
        
        #calculating silhouette score for each sample
        sample_silhouette_vals = metrics.silhouette_samples(data, labels)    
        
        y_lower = 10 #for spacing between silhoette plots
        for i in range(k):
            #aggregating and sort silhouette scores belonging to cluster i
            cluster_i_silhouette_vals = sample_silhouette_vals\
                [labels == i]
                
            cluster_i_silhouette_vals.sort()
            
            #find size of the ith cluster
            size_cluster_i = cluster_i_silhouette_vals.shape[0]
            y_upper = y_lower + size_cluster_i
            
            colour = cm.nipy_spectral(float(i)/k)
            plt.fill_betweenx(np.arange(y_lower, y_upper),
                              0, cluster_i_silhouette_vals,
                              facecolor=colour, edgecolor=colour, alpha=0.7)
            
            #label silhouette plots with cluster numbers
            plt.text(-0.05, y_lower + 0.5*size_cluster_i, str(i+1))
            
            #compute new lower bound for next plot
            y_lower = y_upper + 10
            
        ax.set_title("Silhouette plot for {} clusters".format(k))
        ax.set_xlabel("Silhouette coefficient values")
        ax.set_ylabel("Cluster label")
        
        #vertical line showing average silhouette score for all values
        ax.axvline(x=silhouette_avg, color="black", linestyle="--")
        
        #don't need y_ticks as cluster numbers are already marked
        ax.set_yticks([])
        ax.set_xticks(np.arange(-0.2, 1, 0.2))
    
    print("\n")
    plt.show()
    plt.close()
    
#to estimate the optimal C value for SVM:
#WARNING: this has considerable runtime for a high number of runs
def C_val_estimation(X, y, target_names, runs):
    
    #to prevent warnings arising from calculating precision/f1 score
    warnings.filterwarnings("ignore")
    
    #checking C values 10^-5, 10^-4, .... , 10^6, 10^7
    Cvals = np.logspace(-5, 7, 13)
    C_accuracy = []
    
    for C in Cvals:
        
        C_accuracy_temp = []
        
        for i in range(runs):
            
            n_sample = len(X) #number of samples = number of rows
        
            #randomly rearranging data
            order = np.random.permutation(n_sample)
            X = X[order]
            y = y[order].astype(np.float)
            
            #creating training and test datasets:
            X_train = X[:int(0.9*n_sample)]
            y_train = y[:int(0.9*n_sample)]
            
            X_test = X[int(0.9*n_sample):]
            y_test = y[int(0.9*n_sample):]
    
            clf = svm.SVC(kernel = "linear", C = C)
            clf.fit(X_train, y_train)
            
            y_pred = clf.predict(X_test)
            
            report = metrics.classification_report(y_test, y_pred,
                                                   output_dict=True)
            
            C_accuracy_temp.append(report["accuracy"])
        
        C_accuracy.append(np.mean(C_accuracy_temp))
    
    #creating table of C-value to average accuracy
    plt.figure(figsize = (6,3))
    col_label = ("C value", "Avg. accuracy")
    
    plt.title("Average accuracies for different C values over {} runs"
              .format(runs))
    plt.axis("off")
    C_full = np.array([Cvals, C_accuracy]).T
    plt.table(cellText=C_full, colLabels=col_label, loc="center")
    
    plt.show()
    plt.close()

def kNN_distance_metric(X, y, target_names, K_val, runs):

    #metrics to check:
    Dvals = ["manhattan", "euclidean", "chebyshev"]
    
    D_accuracy = []
    
    for D in Dvals:
        
        D_accuracy_temp = []
        
        #10 cross validation runs:
        for i in range(runs):
            
            n_sample = len(X) #number of samples = number of rows
        
            #randomly rearranging data
            order = np.random.permutation(n_sample)
            X = X[order]
            y = y[order].astype(np.float)
            
            #creating training and test datasets:
            X_train = X[:int(0.9*n_sample)]
            y_train = y[:int(0.9*n_sample)]
            
            X_test = X[int(0.9*n_sample):]
            y_test = y[int(0.9*n_sample):]
    
            clf = KNeighborsClassifier(n_neighbors = K_val, metric = D)
            clf.fit(X_train, y_train)
            
            y_pred = clf.predict(X_test)
            
            report = metrics.classification_report(y_test, y_pred,
                                                   output_dict=True)
            
            D_accuracy_temp.append(report["accuracy"])
        
        D_accuracy.append(np.mean(D_accuracy_temp))
    
    #creating table of C-value to average accuracy
    plt.figure(figsize = (6,3))
    col_label = ("Dist. metric", "Avg. accuracy")
    
    plt.title("Average accuracies for kNN for\ndifferent distance metrics over 100 runs")
    plt.axis("off")
    C_full = np.array([Dvals, D_accuracy]).T
    plt.table(cellText=C_full, colLabels=col_label, loc="center")
    
    plt.show()
    plt.close()

# plotting function for classifiers; nC2 plots where n = no. of features
def class_plotter(clf, X, y, X_test, method):

    tmp = [] #selects principal component axes to plot against
    for i in range(X.shape[1]):
        for j in range(i+1, X.shape[1]):
            if (j < X.shape[1]):
                tmp.append([i, j])
    
    for i in range(len(tmp)):
        
        plt.figure()
        plt.clf()
        
        #plottng all the data:
        plt.scatter(X[:, tmp[i][0]], X[:, tmp[i][1]], c = y, zorder = 10,
                    cmap = plt.cm.Spectral, edgecolor = "k", s = 20)
        
        #circling out the test data:
        plt.scatter(X_test[:, tmp[i][0]], X_test[:, tmp[i][1]], s=80, 
                facecolors="none", zorder=10, edgecolor="k")
        
        plt.axis("tight")
        
        x_min = X[:, tmp[i][0]].min() - 1
        x_max = X[:, tmp[i][0]].max() + 1
        y_min = X[:, tmp[i][1]].min() - 1
        y_max = X[:, tmp[i][1]].max() + 1
        
        #putting the result into a colour plot:
        XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
        pred = np.array([XX.ravel(), YY.ravel()] +
                        [np.repeat(0, XX.ravel().size) for _ in range(2)]).T
    
        # # 
        
        Z = clf.predict(pred).reshape(XX.shape)
    
        plt.pcolormesh(XX, YY, Z > 0, cmap=plt.cm.Spectral, shading = "auto")
        plt.contourf(XX, YY, Z, cmap = plt.cm.Spectral, alpha = 0.8)
        
        plt.title("{}".format(method))
        plt.xlabel("Principal Component: {}".format(tmp[i][0] + 1))
        plt.ylabel("Principal Component: {}".format(tmp[i][1] + 1))
        plt.show()
        plt.close()

baseData = loadData("WeatherData.txt")

#### Pre-Processing

## Creating a histogram of every feature:

for i in range(baseData.shape[1]):
    X = baseData[:, i]
    plt.hist(X, bins = 20)
    plt.title("Feature: {}".format(i+1))
    plt.show()

## It can be observed that the data is not normal, because several features
## have multiple peaks

## Removing outliers outside 6 standard deviations
noOutliersData = outlier_removal(baseData, 6)

## The data are normalised, then centralised about feature means
normData = normalise(noOutliersData)
data = centralise(normData)

### Principal Component Analysis:
## For dimensionality reduction

cov = np.cov(data, rowvar = False)
eigVals, eigVectors = sp.linalg.eig(cov)

orderedEigVectors = np.empty(eigVectors.shape, dtype = complex)

tmp = eigVals.copy()

maxValue = float("-inf")
maxValuePos = -1

for i in range(len(eigVectors)):

    maxVal = float("-inf")
    maxValPos = -1
        
    for n in range(len(eigVectors)):
        if (tmp[n] > maxValue):
            maxVal = tmp[n]
            maxValPos = n

    orderedEigVectors[:,i] = eigVectors[:,maxValuePos]
    tmp[maxValuePos] = float("-inf")

## An eigenvalue tells you how much variance is captured along its eigenvector.
## Therefore, plotting eigenvalues shows how rapidly the variance drops.
## (Elbow method)


## To calculate percentage of variance explained by each eigenvalue:
eigValRatio = (eigVals*100)/sum(eigVals)

plt.figure(figsize = (6,4))

## x-axis counts no. of eigenvalues/principal components

plt.plot([i+1 for i in range(len(eigVals))], eigValRatio, "--ko")
plt.xlabel("No. of Principal Components")
plt.xticks(np.arange(1, 20, 1))
plt.ylabel("Eigenvalue %")
plt.show()

## From the obtained graph, there are diminishing reductions in variations
## after 4 eigenvalues, and these 4 eigenvalues account for over 90% variance.
## Therefore, for this PCA, the dimensions are reduced to 4.

k = 4

projectionMatrix = eigVectors[:,0:k]

pcaData = data.dot(projectionMatrix)
        
plotter(pcaData, "Principal Component Analysis")

## The PCA shows how there are no drastic divisions in the data.

#### Clustering

## Density-based clustering methods are chosen because

### Clustering Method 1
### K-means method

## The optimal number of clusters is identified, using silhouette analysis.
## 2 - 7 clusters are considered.

silhouette_maker(pcaData, 7)

## From the silhouette analysis, it is seen that all plotted clusters in all
## cluster combinations have silhouette score above the average score of the
## cluster. A high silhouette coefficient implies good cohesion (i.e. how
## similar a point is to its own cluster), whereas a negative score implies
## that the point has been incorrectly clustered.

## Another point to consider is the thickness of the silhouette plot clusters.
## Similar thicknesses are better as the points are clustered evenly.

## Therefore, given these considerations, clustering with 2 clusters is
## considered the optimal choice as it has the highest average silhouette
## score, the silhouette plot clusters have similar thicknesses, and
## proportionately fewer data points are clustered wrongly.

## However, 2 clusters will not provide enough classes for our analysis. Going
## for 4 clusters as it has a good silhouette score, etc.

## Therefore, proceeding to obtain the clusters with k = 4:
    
centroids, kMeansLabels = sc.vq.kmeans2(pcaData, 4)

plotter(pcaData, "k-means Clusters", centroids)

#calculating Calinski and Harabasz score
kMeans_ch_score = metrics.calinski_harabasz_score(pcaData, kMeansLabels)

### Clustering Method 2
### Mean-shift method

## Code adapted from scikit-learn documentation

## (info)

bandwidth = estimate_bandwidth(pcaData, quantile = 0.2)

msModel = MeanShift(bandwidth = bandwidth, bin_seeding = True)
msModel.fit(pcaData)
msLabels = msModel.labels_
cluster_centers = msModel.cluster_centers_

#calculating Calinski and Harabasz score
ms_ch_score = metrics.calinski_harabasz_score(pcaData, msLabels)

msUniqueLabels = np.unique(msLabels)
msN = len(msUniqueLabels) #number of clusters

#plotting mean shift clusters in 2D across all principal component axes:
print("Number of estimated clusters by mean-shift method: {}".format(msN))

plt.figure(figsize = (10,10))
# plt.suptitle("2D visualisation of mean-shift clusters with marked centroids",
#               fontsize = 18, fontweight = "bold")
plt.tight_layout()

ax1 = plt.subplot(3, 2, 1)
ax2 = plt.subplot(3, 2, 2)
ax3 = plt.subplot(3, 2, 3)
ax4 = plt.subplot(3, 2, 4)
ax5 = plt.subplot(3, 2, 5)
ax6 = plt.subplot(3, 2, 6)
      
colours = ["b", "g", "r", "c", "m", "y"]
for k, col in zip(range(msN), colours):
    select = msLabels == k
    cluster_center = cluster_centers[k]
    ax1.plot(pcaData[select, 0], pcaData[select, 1], col + ".", alpha = 0.3)
    ax1.plot(cluster_center[0], cluster_center[1], "x",
          markerfacecolor = col, markeredgecolor = "k", ms = 30, mew = 7)
    ax2.plot(pcaData[select, 0], pcaData[select, 2], col + ".", alpha = 0.3)
    ax2.plot(cluster_center[0], cluster_center[2], "x",
          markerfacecolor = col, markeredgecolor = "k", ms = 30, mew = 7)
    ax3.plot(pcaData[select, 0], pcaData[select, 3], col + ".", alpha = 0.3)
    ax3.plot(cluster_center[0], cluster_center[3], "x",
          markerfacecolor = col, markeredgecolor = "k", ms = 30, mew = 7)
    ax4.plot(pcaData[select, 1], pcaData[select, 2], col + ".", alpha = 0.3)
    ax4.plot(cluster_center[1], cluster_center[2], "x",
          markerfacecolor = col, markeredgecolor = "k", ms = 30, mew = 7)
    ax5.plot(pcaData[select, 1], pcaData[select, 3], col + ".", alpha = 0.3)
    ax5.plot(cluster_center[1], cluster_center[3], "x",
          markerfacecolor = col, markeredgecolor = "k", ms = 30, mew = 7)
    ax6.plot(pcaData[select, 2], pcaData[select, 3], col + ".", alpha = 0.3)
    ax6.plot(cluster_center[2], cluster_center[3], "x",
          markerfacecolor = col, markeredgecolor = "k", ms = 30, mew = 7)
    

ax1.set_xlabel("1st Principal Component", fontweight = "bold")
ax1.set_ylabel("2nd Principal Component", fontweight = "bold")
ax1.set_xticks([])
ax1.set_yticks([])

ax2.set_xlabel("1st Principal Component", fontweight = "bold")
ax2.set_ylabel("3rd Principal Component", fontweight = "bold")
ax2.set_xticks([])
ax2.set_yticks([])

ax3.set_xlabel("1st Principal Component", fontweight = "bold")
ax3.set_ylabel("4th Principal Component", fontweight = "bold")
ax3.set_xticks([])
ax3.set_yticks([])

ax4.set_xlabel("2nd Principal Component", fontweight = "bold")
ax4.set_ylabel("3rd Principal Component", fontweight = "bold")
ax4.set_xticks([])
ax4.set_yticks([])

ax5.set_xlabel("2nd Principal Component", fontweight = "bold")
ax5.set_ylabel("4th Principal Component", fontweight = "bold")
ax5.set_xticks([])
ax5.set_yticks([])

ax6.set_xlabel("3rd Principal Component", fontweight = "bold")
ax6.set_ylabel("4th Principal Component", fontweight = "bold")
ax6.set_xticks([])
ax6.set_yticks([])

plt.show()
plt.close()

### Clustering Method 3
### Affinity propagation method

## (info)

apModel = AffinityPropagation(random_state = None)
apModel.fit(pcaData)

apCentroidIndices = apModel.cluster_centers_indices_
apLabels = apModel.labels_

#calculating Calinski and Harabasz score
ap_ch_score = metrics.calinski_harabasz_score(pcaData, apLabels)

apN = len(apCentroidIndices) #number of potential exemplars

## Plotting the affinity propagation clusters in 2D for every principal component
plt.figure(figsize = (10,10))
# plt.suptitle("2D visualisation of affinity propagation clusters with marked centroids",
#               fontsize = 18, fontweight = "bold")
plt.tight_layout()

ax1 = plt.subplot(3, 2, 1)
ax2 = plt.subplot(3, 2, 2)
ax3 = plt.subplot(3, 2, 3)
ax4 = plt.subplot(3, 2, 4)
ax5 = plt.subplot(3, 2, 5)
ax6 = plt.subplot(3, 2, 6)
      
colours = ["b", "g", "r", "c", "m", "y"]
for k, col in zip(range(msN), colours):
    select = apLabels == k
    cluster_center = pcaData[apCentroidIndices[k]]
    ax1.plot(pcaData[select, 0], pcaData[select, 1], col + ".", alpha = 0.3)
    ax1.plot(cluster_center[0], cluster_center[1], "o",
          markerfacecolor = col, markeredgecolor = "k", ms = 14, mew = 3)
    ax2.plot(pcaData[select, 0], pcaData[select, 2], col + ".", alpha = 0.3)
    ax2.plot(cluster_center[0], cluster_center[2], "o",
          markerfacecolor = col, markeredgecolor = "k", ms = 14, mew = 3)
    ax3.plot(pcaData[select, 0], pcaData[select, 3], col + ".", alpha = 0.3)
    ax3.plot(cluster_center[0], cluster_center[3], "o",
          markerfacecolor = col, markeredgecolor = "k", ms = 14, mew = 3)
    ax4.plot(pcaData[select, 1], pcaData[select, 2], col + ".", alpha = 0.3)
    ax4.plot(cluster_center[1], cluster_center[2], "o",
          markerfacecolor = col, markeredgecolor = "k", ms = 14, mew = 3)
    ax5.plot(pcaData[select, 1], pcaData[select, 3], col + ".", alpha = 0.3) 
    ax5.plot(cluster_center[1], cluster_center[3], "o",
          markerfacecolor = col, markeredgecolor = "k", ms = 14, mew = 3)
    ax6.plot(pcaData[select, 2], pcaData[select, 3], col + ".", alpha = 0.3)        
    ax6.plot(cluster_center[2], cluster_center[3], "o",
          markerfacecolor = col, markeredgecolor = "k", ms = 14, mew = 3)
    
    for x in pcaData[select]:
        ax1.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]],
                  col, alpha = 0.1)
        ax2.plot([cluster_center[0], x[0]], [cluster_center[2], x[2]],
                  col, alpha = 0.1)
        ax3.plot([cluster_center[0], x[0]], [cluster_center[3], x[3]],
                  col, alpha = 0.1)
        ax4.plot([cluster_center[1], x[1]], [cluster_center[2], x[2]],
                  col, alpha = 0.1)
        ax5.plot([cluster_center[1], x[1]], [cluster_center[3], x[3]],
                  col, alpha = 0.1)    
        ax6.plot([cluster_center[2], x[2]], [cluster_center[3], x[3]],
                  col, alpha = 0.1)    

ax1.set_xlabel("1st Principal Component", fontweight = "bold")
ax1.set_ylabel("2nd Principal Component", fontweight = "bold")
ax1.set_xticks([])
ax1.set_yticks([])

ax2.set_xlabel("1st Principal Component", fontweight = "bold")
ax2.set_ylabel("3rd Principal Component", fontweight = "bold")
ax2.set_xticks([])
ax2.set_yticks([])

ax3.set_xlabel("1st Principal Component", fontweight = "bold")
ax3.set_ylabel("4th Principal Component", fontweight = "bold")
ax3.set_xticks([])
ax3.set_yticks([])

ax4.set_xlabel("2nd Principal Component", fontweight = "bold")
ax4.set_ylabel("3rd Principal Component", fontweight = "bold")
ax4.set_xticks([])
ax4.set_yticks([])

ax5.set_xlabel("2nd Principal Component", fontweight = "bold")
ax5.set_ylabel("4th Principal Component", fontweight = "bold")
ax5.set_xticks([])
ax5.set_yticks([])

ax6.set_xlabel("3rd Principal Component", fontweight = "bold")
ax6.set_ylabel("4th Principal Component", fontweight = "bold")
ax6.set_xticks([])
ax6.set_yticks([])

plt.show()
plt.close()

######

####Classification

##Using the k-means clusters to proceed to the classification stage.

###Classification method 1
###Support Vector Machines

X = pcaData
y = kMeansLabels
target_names = ["Class 1", "Class 2", "Class 3", "Class 4"]

## WARNING: THE FOLLOWING CODE HAS A SIGNIFICANT RUNTIME
## Estimating optimal C value by gauging accuracy over 100 cross-validation runs
C_val_estimation(X, y, target_names, 100)
## The best C value is estimated to be 10000

n_sample = len(X) #number of samples = number of rows

#randomly rearranging data
order = np.random.permutation(n_sample)
X = X[order]
y = y[order].astype(np.float)

#creating training and test datasets:
X_train = X[:int(0.9*n_sample)]
y_train = y[:int(0.9*n_sample)]

X_test = X[int(0.9*n_sample):]
y_test = y[int(0.9*n_sample):]

clf = svm.SVC(kernel = "linear", C = 100000)
clf.fit(X_train, y_train)

#printing classification report
y_pred = clf.predict(X_test)

print("SVM with PCs 1, 2, 3 & 4\nClassification report:\n",
      metrics.classification_report(y_test, y_pred,
                                    target_names=target_names))

print("Confusion matrix:\n", metrics.confusion_matrix(y_test, y_pred))

class_plotter(clf, X, y, X_test, "SVM")

###Classification method 2
###Naive Bayes

clf2 = GaussianNB()
clf2.fit(X_train, y_train)

#printing classification report
y_pred = clf2.predict(X_test)

print("\nNaive Bayes with PCs 1, 2, 3 & 4\nClassification report:\n",
      metrics.classification_report(y_test, y_pred,
                                    target_names=target_names))

print("Confusion matrix:\n", metrics.confusion_matrix(y_test, y_pred))

class_plotter(clf2, X, y, X_test, "Naive Bayes")

### Classification method 3
### k-nearest neighbours

## Estimating the k value for the kNN algorithm with the thumb-rule that
## k equals the square root of the number of samples.

K_val = round(np.sqrt(X.shape[0]))

## Estimating the best distance metric to use for this dataset:
## Considering Manhattan, Euclidean, Chebyshev and Mahalanobis metrics  


## Estimating the best distance metric from Manhattan, Euclidean and Chebyshev
## to use by computing accuracy over 100 cross-validation runs for 
kNN_distance_metric(X, y, target_names, K_val, 100)
## Euclidean and Manhattan are similarly very accurate. Proceeding with
## Euclidean distance

clf3 = KNeighborsClassifier(n_neighbors = K_val, metric = "euclidean")
clf3.fit(X_train, y_train)

#printing classification report
y_pred = clf3.predict(X_test)

print("\nk Nearest Neighbours with PCs 1, 2, 3 & 4\nClassification report:\n",
      metrics.classification_report(y_test, y_pred,
                                    target_names=target_names))

print("Confusion matrix:\n", metrics.confusion_matrix(y_test, y_pred))

class_plotter(clf3, X, y, X_test, "K Nearest Neighbours")

###Cross-validation (10 runs)

# For tabulation purposes:

CV_run = []

SVM_macro_precision = []
SVM_macro_recall = []
SVM_macro_f1 = []
SVM_accuracy = []

NB_macro_precision = []
NB_macro_recall = []
NB_macro_f1 = []
NB_accuracy = []

kNN_macro_precision = []
kNN_macro_recall = []
kNN_macro_f1 = []
kNN_accuracy = []

for i in range(10):
    CV_run.append(i+1)
    X = pcaData
    y = kMeansLabels
    target_names = ["Class 1", "Class 2", "Class 3", "Class 4"]
    
    n_sample = len(X) #number of samples = number of rows
    
    #randomly rearranging data
    order = np.random.permutation(n_sample)
    X = X[order]
    y = y[order].astype(np.float)
    
    #creating training and test datasets:
    X_train = X[:int(0.9*n_sample)]
    y_train = y[:int(0.9*n_sample)]
    
    X_test = X[int(0.9*n_sample):]
    y_test = y[int(0.9*n_sample):]
    
    ###SVM
    
    clf = svm.SVC(kernel = "linear", C = 100000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    SVM_report = metrics.classification_report(y_test, y_pred, output_dict=True)
    
    SVM_macro_precision.append(round(SVM_report["macro avg"]["precision"], 3))
    SVM_macro_recall.append(round(SVM_report["macro avg"]["recall"], 3))
    SVM_macro_f1.append(round(SVM_report["macro avg"]["f1-score"], 3))
    SVM_accuracy.append(round(SVM_report["accuracy"], 3))
    
    ###Naive Bayes
    
    clf2 = GaussianNB()
    clf2.fit(X_train, y_train)
    y_pred = clf2.predict(X_test)
    
    NB_report = metrics.classification_report(y_test, y_pred, output_dict=True)
    
    NB_macro_precision.append(round(NB_report["macro avg"]["precision"], 3))
    NB_macro_recall.append(round(NB_report["macro avg"]["recall"], 3))
    NB_macro_f1.append(round(NB_report["macro avg"]["f1-score"], 3))
    NB_accuracy.append(round(NB_report["accuracy"], 3))
    
    ###k Nearest Neighbours
    
    clf3 = KNeighborsClassifier(n_neighbors = K_val, metric = "euclidean")
    clf3.fit(X_train, y_train)
    y_pred = clf3.predict(X_test)
    
    kNN_report = metrics.classification_report(y_test, y_pred, output_dict=True)
    
    kNN_macro_precision.append(round(kNN_report["macro avg"]["precision"], 3))
    kNN_macro_recall.append(round(kNN_report["macro avg"]["recall"], 3))
    kNN_macro_f1.append(round(kNN_report["macro avg"]["f1-score"], 3))
    kNN_accuracy.append(round(kNN_report["accuracy"], 3))

##tabulating results from cross-validation:

#creating final row of mean values:
CV_run.append("Average")    

SVM_macro_precision.append(round(np.mean(SVM_macro_precision), 3))    
SVM_macro_recall.append(round(np.mean(SVM_macro_recall), 3))
SVM_macro_f1.append(round(np.mean(SVM_macro_f1), 3))
SVM_accuracy.append(round(np.mean(SVM_accuracy), 3))
    
NB_macro_precision.append(round(np.mean(NB_macro_precision), 3))    
NB_macro_recall.append(round(np.mean(NB_macro_recall), 3))
NB_macro_f1.append(round(np.mean(NB_macro_f1), 3))
NB_accuracy.append(round(np.mean(NB_accuracy), 3))

kNN_macro_precision.append(round(np.mean(kNN_macro_precision), 3))    
kNN_macro_recall.append(round(np.mean(kNN_macro_recall), 3))
kNN_macro_f1.append(round(np.mean(kNN_macro_f1), 3))
kNN_accuracy.append(round(np.mean(kNN_accuracy), 3))

#creating table of metrics:

plt.figure(figsize = (12,3))
col_label = ("Cross-validation Run", "Macro Precision", "Macro Recall",
             "Macro f1-score", "Accuracy")

plt.title("Cross-validation metrics for SVM")
plt.axis("off")
SVM_full = np.array([CV_run, SVM_macro_precision, SVM_macro_recall,
                     SVM_macro_f1, SVM_accuracy]).T
SVM_table = plt.table(cellText=SVM_full, colLabels=col_label, loc="center")

plt.show()
plt.close()

plt.figure(figsize = (12,3))
col_label = ("Cross-validation Run", "Macro Precision", "Macro Recall",
              "Macro f1-score", "Accuracy")

plt.title("Cross-validation metrics for Naive Bayes")
plt.axis("off")
NB_full = np.array([CV_run, NB_macro_precision, NB_macro_recall,
                     NB_macro_f1, NB_accuracy]).T
NB_table = plt.table(cellText=NB_full, colLabels=col_label, loc="center")

plt.show()
plt.close()

plt.figure(figsize = (12,3))
col_label = ("Cross-validation Run", "Macro Precision", "Macro Recall",
              "Macro f1-score", "Accuracy")

plt.title("Cross-validation metrics for k Nearest Neighbours")
plt.axis("off")
kNN_full = np.array([CV_run, kNN_macro_precision, kNN_macro_recall,
                     kNN_macro_f1, kNN_accuracy]).T
kNN_table = plt.table(cellText=kNN_full, colLabels=col_label, loc="center")

plt.show()
plt.close()

#creating comparative metrics:

comparative_precision = np.array([SVM_macro_precision[-1],
                                  NB_macro_precision[-1],
                                  kNN_macro_precision[-1]])
comparative_recall = np.array([SVM_macro_recall[-1],
                                  NB_macro_recall[-1],
                                  kNN_macro_recall[-1]])
comparative_f1 = np.array([SVM_macro_f1[-1],
                                  NB_macro_f1[-1],
                                  kNN_macro_f1[-1]])
comparative_accuracy = np.array([SVM_accuracy[-1],
                                  NB_accuracy[-1],
                                  kNN_accuracy[-1]])
    
#creating table of comparative metrics:
    
method_names = ("SVM", "Naive Bayes", "k Nearest Neighbours")

plt.figure(figsize = (12,1))
col_label = ("Method", "Avg. Macro Precision", "Avg. Macro Recall",
             "Avg. Macro f1-score", "Avg. Accuracy")

plt.title("Comparative average metrics for 10 cross-validation runs")
plt.axis("off")
comparative_full = np.array([method_names, comparative_precision,
                             comparative_recall, comparative_f1, 
                             comparative_accuracy]).T

comparative_table = plt.table(cellText=comparative_full, colLabels=col_label,
                              loc="center")










