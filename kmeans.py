import numpy as np
import pandas as pd
import math
import random
from sklearn.metrics import confusion_matrix
from PIL import Image
import matplotlib.pyplot as plt
import sys

#this function returns key, objects pairs where the
#key is the predicted class and the objects are the centroids
#the prediction is based on the highest frequency class in
#each cluster
def getCentroidsPredictions(classes, clusters, centroids):
  clusterClasses = []
  res = {}
  for i in range(len(clusters)):
    frequencies = classes.loc[clusters[i]].value_counts()
    classType = frequencies.index[0]
    # if class type is already in set, just continue to next
    #cluster
    if classType in clusterClasses:
      continue
    clusterClasses.append(classType)
    res[classType] = centroids[i]
  return res

#gets new cluster centroids based on the members of the clusters
#the mean of each dimension values
def recalculateCentroids(df, clusters):
  centroids = [[] for _ in range(len(clusters))]
  for i in range(len(clusters)):
    centroids[i] = np.array(df.loc[clusters[i]].mean().values)
  return np.array(centroids)

#gets the accuracy based on the actual and predicted classes
def getAccuracy(actualClass, predictedClass):
  correct = 0.0
  for i in range(len(actualClass)):
    if actualClass[i] == predictedClass[i]:
      correct += 1
  return correct/len(actualClass)

#calculates the entropy of each cluster based on the classes of their memebers
def calculateEntropy(classes, cluster):
  clusterClasess = classes.loc[cluster]
  entropySum = 0
  mi = len(clusterClasess)
  for classType in clusterClasess.unique():
    mij = len(np.nonzero(clusterClasess == classType)[0])
    entropySum += (mij/mi) * math.log2(mij/mi)
    
  return entropySum * -1 
    
#calculates the mean entropy based on the entropy of each closses
def calculateMeanEntropy(classes, clusters):
  meanSum = 0
  for i in range(len(clusters)):
    meanSum+= (len(clusters[i])/len(classes)) * calculateEntropy(classes, clusters[i])
  return meanSum

#generates all possible pars of a list
def generatePairs(lst):
  pairs = []
  for i in range(len(lst)):
    for j in range(i):
      if i != j:
        pairs.append([i,j])
  return pairs

#calculate the mss of the centroids
def calculateMSS(centroids):
  pairs = generatePairs(centroids)
  mssSum = 0
  k = len(centroids)
  for pair in pairs:
    mssSum += getEuclidianDistancePoints(centroids[pair[0]], centroids[pair[1]])**2
  return mssSum/(k*(k-1)/2)

#calcualtes the mse of a given cluster and centroid
def calculateMSECluster(cluster, centroid):
  return np.sum(np.sum(np.power((cluster- centroid), 2), axis=1))/len(cluster)
  
#calcualte the average mse based on the mse of each cluster and centroids  
def calculateAverageMSE(df, clusters, centroids):   
  k = len(clusters)
  mseSum = 0
  for i in range(k):
    mseSum += calculateMSECluster(df.iloc[clusters[i]], centroids[i])
  return mseSum/k

#calculates the euclidian distance based on the cluster and a centroid
def getEuclidianDistance(cluster, centroid):
  return np.power(np.sum(np.power((cluster - centroid), 2), axis=1), 0.5)

#calculates euclidian distance for two points of any dimension   
def getEuclidianDistancePoints(p1, p2):
  return math.sqrt(np.sum(np.power((p1 - p2), 2), axis=0))

#generates random centroid values based on the number of clusters, dimensions
#with maximum and minimum values
def getRandomCentroids(k, dim, minimum, maximum, seed=2814):
  random.seed(seed)
  random.randint(1, 10)
  centroids = []
  for i_ in range(k):
    randomCenter = np.array([random.randint(minimum, maximum) for _ in range(dim)])
    centroids.append(randomCenter)
    
  return np.array(centroids)

def main(k=10):
  #generate the headers for the sets
  headers = ["x", "y"]
  #define number of clusters
  trainSet = pd.read_csv(sys.argv[1], names=headers)
  testSet = pd.read_csv(sys.argv[2], names=headers)
  
  trainingPeriods = 5
  trainCentroids = []
  trainClusters = []
  minMSE = 999999;
  for _ in range(trainingPeriods):
    #get random centroids from the train data
    newCentroids =  trainSet.drop('class', axis=1).sample(k).values
    #get random centroids to compare against
    oldCentroids =  getRandomCentroids(k=k, dim=64, minimum=0, maximum=16)
    #repeat until the mean value of the centroid does not change
    while(np.mean(newCentroids - oldCentroids) != 0):
      clusters = [[] for _ in range(len(newCentroids))]
      clustersFinal = []
      distances = []
      #get the distance of all train data to each centroid
      for i in range(len(newCentroids)):
        distances.append(getEuclidianDistance(trainSet.drop('class', axis=1), newCentroids[i]))
      #get which data point is closer to which centroid
      closerCentroids = np.argmin(np.array(distances), axis=0)
      #set each cluster's members to the data points they are closer with
      for i in range(len(clusters)):
        clusters[i] = np.where(closerCentroids == i)[0]
        
      #remove any empty clusters from calculations since they are not going
      #to be updated anymore
      emptyClusters = []
      for i in range(len(clusters)):
        if len(clusters[i]) > 0:
          clustersFinal.append(clusters[i])
        else:
          emptyClusters.append(i)
      newCentroids = np.delete(newCentroids,emptyClusters, axis=0) 
      #save the centroids to calculate the man for next loop
      oldCentroids = newCentroids.copy()
      #recalculate new cnetroids based on the clusters
      newCentroids = recalculateCentroids(trainSet.drop('class', axis=1), clustersFinal)
      #if the average mse is the current minimum, saved that clustering data
      avgMse = calculateAverageMSE(trainSet.drop('class', axis=1), clustersFinal, newCentroids)
      if avgMse < minMSE:
        minMSE = avgMse
        trainCentroids = newCentroids.copy()
        trainClusters = clustersFinal.copy()
  #with the lowest average mse clusters, calculate metrics
  avgMse = calculateAverageMSE(trainSet.drop('class', axis=1), trainClusters, trainCentroids)
  mss = calculateMSS(trainCentroids)
  meanEntropy = calculateMeanEntropy(trainSet['class'], trainClusters)
  print("average MSE", avgMse, "MSS", mss, "Mean Entropy", meanEntropy)
  #get the predicted class based on each cluster class frequency
  predictionCentroids = getCentroidsPredictions(trainSet['class'], trainClusters, trainCentroids)
  predictionClasses = []
  testData = testSet.drop('class', axis=1)
  #classify the test data based on its distance to each centroid
  for index in testData.index:
    minDist = 999999
    classType = -1
    for key in predictionCentroids:
      dist =  np.power(np.sum(np.power((testData.iloc[index] - predictionCentroids[key]), 2)), 0.5)
      if dist < minDist:
        minDist = dist
        classType = key
    predictionClasses.append(classType)
  #get confusion matrix
  cm = confusion_matrix(testSet['class'].values, predictionClasses)
  print(cm)
  #get accuracy of predictions
  print(getAccuracy(testSet['class'].values, predictionClasses))
  #show the graphical representation of each centroid
  plt.gray()
  for key in predictionCentroids:
    t = (predictionCentroids[key] * 15)
    t.resize((8, 8))
    im = Image.fromarray(t)
    plt.figure()
    plt.title(key)
    plt.imshow(im)
  plt.show()
  
  
if __name__ == '__main__':
  if len(sys.argv) > 3:
    main(int(sys.argv[3]))
  else:
    print("Usage", sys.argv[0], "optdigits.train optdigits.test 10")