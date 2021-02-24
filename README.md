# Hybrid_Travel_Recommender_System-HTRS

This project is conducting under the supervision of Omar Nada at the Sharpest Minds fellowship program.

# Introduction and Motivation

One of the first things to do while planning a trip is to book a good place to stay. Booking a hotel online can be an overwhelming task with thousands of hotels to choose from, for every destination. Motivated by the importance of these situations, we decided to work on the task of recommending hotels to users. We used Expedia’s hotel recommendation dataset, which has a variety of features that helped us achieve a deep understanding of the process that makes a user choose certain hotels over others. The aim of this project is to to use different recommendation models -2 collaborative techniques and 1 hybrid technique. 


# Dataset

We have used the Expedia Hotel Recommendation dataset from Kaggle. The dataset, which had been collected in the 2013-2014 time-frame, consists of a variety of features that could provide us great insights into the process user go through for choosing hotels. The training set consists of 37,670,293 entries and the test set contains 2,528,243 entries. Apart from this, the dataset also provide some latent features for each of the destinations recorded in the train and test sets. The data is anonymized and almost all the fields are in numeric format. The goal is to predict 5 hotel clusters where a user is more likely to stay, out of a total of 100 hotel clusters. The problem has been modeled as a ranked multi-class classification task. Missing data, ranking requirement, and the curse of dimensionality are the main challenges posed by this dataset. 

# Notebooks Files

## 1. Data_processing-Feature-eng.ipynb
Our first step was to clean and pre-process the data and perform exploratory analysis to get some interesting insights into the process of choosing a hotel.We identified the searches by each user belonging to a specific type of destination. This gave us some useful information about which hotel cluster was finally chosen over other hotel clusters explored by the user. One important observation to note is that few users might be travel agents and could explore multiple type of destinations at the same time. Feature engineering was performed and many new features such as duration, importance features, solo trip or family trip etc. were extracted from dataset. Moreover,we have 149 latent features for each destination; we have applied PCA to extract the most relevant dimensions.

## 2. Hotel_Baseline_Recommender.ipynb
The two main families of Recommendation system are Content-Based and Collaborative-Filtering models. Collaborative filtering methods are based on similarity from users and items interaction and content-based filtering methods calculate the similarity of attributes of an items and items. 

We have used Collaborative-Filtering model as the baseline model. Collaborative Filtering can be divided into Memory-Based Collaborative Filtering and Model-Based Collaborative filtering. First of all, We have performed some data analysis for removing duplicates data and understand better user item rating distribution. We have created user-item matrices for train and test set. Then performed some data analysis on the rating matrix and transformed rating matrix into average rating matrix (per user/row). Then, two collaborative filtering models were implemented from scratch.

            ** Memory-Based CF by computing cosine similarity: Memory-Based Collaborative Filtering approaches can be divided into two main sections: user-item filtering   and item-item filtering. In both cases, We created a rating matrix that builds from the entire dataset in order to make recommendations.

            ** Model-Based CF by using singular value decomposition (SVD) and Alternating Least Squares (ALS) method: Model-Based CF models are developed using machine learning algorithms to predict a user's rating of unrated items.  As per my understanding, the algorithms in this approach can further be broken down into 3 sub-types such as Clustering based algorithm, Matrix Factorization, Deep Learning. For example, we can mention few of these algorithms SVD, NMF, KNN etc.

