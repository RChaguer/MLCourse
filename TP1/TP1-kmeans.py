#!/usr/bin/env python
# coding: utf-8

# # IF240 - Apprentissage et deep learning
# 
# ## Practice 1: Kmeans algorithm
# 
# By Aurélie Bugeau
# Credits:  Chris Piech and Andrew Ng

# K-Means is an algorithm that takes aims at clustering a given dataset into $k$ groups (called clusters) of data similar to each other. Each cluster is characterized by its centroid. 
# In this practice, you are going to implement the kmeans algorithm and apply it on different dataset.
# 
# 1. Read and understand the kmeans function below
# 2. Complete the following functions
# 3. Experiment and validate on a toy dataset

# In[ ]:


import numpy as np

# Function: K Means
# -------------
def kmeans(dataSet, k, MAX_ITERATIONS=5):
    
    centroids = getRandomCentroids(dataSet, k)
    iterations = 0
    oldCentroids = None
    
    while not shouldStop(oldCentroids, centroids, iterations, MAX_ITERATIONS):
        oldCentroids = centroids
        iterations += 1
        labels = getLabels(dataSet, centroids)
        centroids = getCentroids(dataSet, labels, k)
    
    return centroids, labels


# In[ ]:


# Function: getRandomCentroids
# -------------
# Initialize centroids by choosing randomly k points from the dataset
def getRandomCentroids(dataSet, numClusters):
    numPoints, _ = dataSet.shape
    centroids = dataSet[np.random.randint(numPoints, size =  numClusters), :]
    return centroids


# In[ ]:


# Function: shouldStop
# -------------
# Returns True or False if k-means if the maximum number of iterations is reached 
# or if the centroids do not change anymore
def shouldStop(oldCentroids, centroids, iterations, MAX_ITERATIONS):
    #COMPLETE





# In[ ]:


# Function: getLabels
# -------------
# Returns the label for each point in the dataSet. The label is the one of the closest centroid
def getLabels(dataSet, centroids):
    #COMPLETE

    
    


# In[ ]:


# Function: getCentroids
# -------------
# Returns the centroids of the clusters. Each centroid is the geometric mean of the points that
# have that centroid's label. Important: If a centroid is empty (no points have
# that centroid's label) you should randomly re-initialize it.
def getCentroids(dataSet, labels, k):
    #COMPLETE


# ### Toy dataset
# 
# Experiment your algorithm on different sets of points with different values of $k$. <span style="color:red">Analyze you results</span>.

# In[ ]:


x1 = np.array([1, 1])
x2 = np.array([2, 2])
x3 = np.array([3, 3])
x4 = np.array([4, 4])
testX = np.vstack((x1, x2, x3, x4))
result = kmeans(testX, 1)


# ### 2D Point Cloud
# You are now going to test and plot the result on a 2D point cloud

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  # for plot styling


# In[ ]:


# dataset generation
from sklearn.datasets.samples_generator import make_blobs
X, y_true = make_blobs(n_samples=300, centers=4,
                       cluster_std=0.60, random_state=0)
plt.scatter(X[:, 0], X[:, 1], s=50);


# In[ ]:


# Apply kmeans
#COMPLETE

#plot the labels
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')

#plot the centers
#COMPLETE
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)


# #### Scikit-learn library
# The Scikit-learn library proposes many functions for machine learning. You are going to compare the results obtained by your implementation with the ones from this library. 
# 
# * Study the documentation of the KMeans function and apply it to the 2D dataset.
# * Compare

# In[ ]:


from sklearn.cluster import KMeans
#COMPLETE


# ### 2 moons problem
# Observe and explain the clustering in 2 classes obtained on the following dataset

# In[ ]:


from sklearn.datasets import make_moons
X, y = make_moons(200, noise=.05, random_state=0)
#COMPLETE


# ### 3D Point Cloud
# Observe and explain the clustering on a 3D point cloud (<span style="color:red">Analyze you results</span>)

# In[ ]:


from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = Axes3D(fig)
X = np.random.rand(50,3)
ax.scatter(X[:, 0], X[:, 1],X[:, 2], s=20)
plt.show()

#COMPLETE

