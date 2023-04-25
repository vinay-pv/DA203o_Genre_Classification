# DA203o_Genre_Classification

The automatic classification of music genres has been a popular research topic recently. The ability to classify music into genres provides valuable information for music streaming platforms and recommendation systems.

## Setup and Execution instructions:

Setup: Run genre_list_augmenter.ipynb to generate features_augmented.csv to be used for further model training.

The remaining Python notebooks (GenreClustering.ipynb, baselines.ipynb, etc.) can be run immediately afterwards to test different algorithms on the prepared data.

## DATA PROCESSING & CLEANUP ​
  
We are using FMA (Free Music Archive, Small dataset): 8,000 tracks of 30 sec each, 7.2 GB
https://github.com/mdeff/fma

FMA metadata contains ~520 spectral data parameters generated from music samples (via Python's librosa module)
Also contains more human-understandable attributes (energy, tempo, danceability, etc.) generated from EchoNest's proprietary tools.

Data cleanup and augmentation was required to map the numerical genre IDs (ground truth) listed for each track to the corresponding feature vector and filter out the tracks with invalid/non-existent genres.
Resultant file contains 520 audio extracted features for 104343 datapoints


## DATA ANALYTICS​
Following represents the Tract distribution according to genre

<img width="639" alt="Track_distribution" src="https://user-images.githubusercontent.com/120098895/234274610-80cd1d63-673d-4904-8ec5-bd61da593885.png">

Following represents the Inter-genre correlation (pairs of genres most often linked with each other in terms of frequency of occurrence)

<img width="887" alt="inter-genre-correlation" src="https://user-images.githubusercontent.com/120098895/234274849-7b57e39b-d3e4-452f-96de-f2c1196645a9.png">

## AUDIO FEATURES EXTRACTION

2 popular libraries to extract features from Audio are Echonest and Librosa. Librosa has more features compared from Echonest.

 
Echonest Features :​
  1. acousticness​
  2. danceability​
  3. energy​
  4. instrumentalness​
  5. liveness​
  6. speechiness​
  7. tempo​
  8. valence​

 
Librosa Features :​

  1. Spectral Features:  
      a. Spectral Centroid:  
      b. Spectral Contrast:  
      c. Spectral Bandwidth:  
      
  2. Tonal Features:  
      a. Chroma:  
      b. CENS:  
      c. Tonnetz:  
      
  3. Rhythmic Features:  
      a. Tempo. 
      b. Beat Synchronous Chroma:  

  4. Root Mean Square (RMS) Energy  

  5. Zero Crossing Rate:  


## DIMENSIONALITY REDUCTION​

We have followed 2 separate methods to reduce the dimensionality.   
1.  Feature Correlation based dimesion reduction. 
  We have used feature correlation matrix to compute the correlation index of different features. Those features which have correlation index of >=0.7 are dropped from the cleaned dataset.
  Random Forest model is trained on this feature set (genre_classification_using_random_forest.ipynb)
  
  
2. Dimension reduction using PCA
  We have Principal Component Analysis (PCA) to reduce highly correlated features. 
  Random forest classifier is trained on this to obtain the results (genre_classification_using_random_forest_with_PCA.ipynb)

## MODELING ​

We followed incremental approach to understand the problem statement, the data, various machine learning techniques to arrive mullti-label classification.

1. Initially, we executed the baseline model provided by FMA dataset itself to understand the concept of music genre classification. We observed that these baseline models predict single genre for a given audio features. They do not perform multi-label classification.
(baselines.ipynb)
  
 Results of baseline model for single music genre classification is as follows -
   
   
<img width="423" alt="image" src="https://user-images.githubusercontent.com/120098895/234277991-7ebbdae7-2dac-4499-a251-f021dc4a6c38.png">

  
  
  
2. We extended our experiment towards unsupervised learning, since there is large amount of unlabelled music data available and it can be used for music genre classification.  
On experiementing with K-Means clustering algorithm, we saw that the clustering is resulting in multiple genres getting overalapped. Thus, creating undesired outcome.
<img width="453" alt="image" src="https://user-images.githubusercontent.com/120098895/234278286-68372a25-3e80-499b-b81e-8016ac69d088.png">

This indicates that traditional human genres are not very clear-cut and are more arbitrary when compared to genres divided by the actual audio characteristics. Further manual clustering them would help in better classification outcome as below.

<img width="456" alt="image" src="https://user-images.githubusercontent.com/120098895/234278362-a6bb04f9-fc63-4679-95e5-2448134c7a3f.png">

However, these results are not much useful w.r.t the known genre set.

3. We then moved towards supervised learning using Random Forest Classifier, which provides provision for multi-label classification with inherent randomness in data selecting by means of bagging. One vs Rest classifier is used to perform multi-label classification.

![Picture3](https://user-images.githubusercontent.com/120098895/234280182-9f11a3b8-bb2c-461f-828d-1bf469496c7b.png)

  
 We also experiemented with deep neural network (DNN) model with drop-out for dimension reductionality to observe the multi-lable classification output.
 
 ![Picture5](https://user-images.githubusercontent.com/120098895/234280211-5fe81026-436c-4d87-b4e2-9b56b93a5b75.png)


## RESULTS

The observed result of our experimentation with supervised learning is tabulated as below -

![Results](https://user-images.githubusercontent.com/120098895/234280064-0cf3cc6c-54f6-4ea1-b56f-cb9474b83023.png)

## Conclusion

We observed that the audio extracted feature does help in performing multi-label classification with reasonable accuracy/F1 score. DNN model seem to outperform Random Forest model w.r.t Coverage Error indicating that DNN model is able to cover many genres compared to Random Forest model.

However, this work can further be extended by experimenting with latest of machine learning techniques to obtain results matching with SOTA.
