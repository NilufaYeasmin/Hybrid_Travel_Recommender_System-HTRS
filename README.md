# Hybrid_Travel_Recommender_System-HTRS

This project is conducting under the supervision of Omar Nada at the Sharpest Minds fellowship program.

# Introduction and Motivation

One of the first things to do while planning a trip is to book a good place to stay. Booking a hotel online can be an overwhelming task with thousands of hotels to choose from, for every destination. Motivated by the importance of these situations, we decided to work on the task of recommending hotels to users. We used Expedia’s hotel recommendation dataset, which has a variety of features that helped us achieve a deep understanding of the process that makes a user choose certain hotels over others. The aim of this project is to to use different recommendation models such as collaborative techniques and hybrid technique. 


# Dataset

We have used the Expedia Hotel Recommendation dataset from Kaggle. The dataset, which had been collected in the 2013-2014 time-frame, consists of a variety of features that could provide us great insights into the process user go through for choosing hotels. The training set consists of 37,670,293 entries and the test set contains 2,528,243 entries. Apart from this, the dataset also provide some latent features for each of the destinations recorded in the train and test sets. The data is anonymized and almost all the fields are in numeric format. The goal is to predict 5 hotel clusters where a user is more likely to stay, out of a total of 100 hotel clusters. The problem has been modeled as a ranked multi-class classification task. Missing data, ranking requirement, and the curse of dimensionality are the main challenges posed by this dataset. 

# Notebooks Files

## 1. Data_processing-Feature-eng.ipynb
Our first step was to clean and pre-process the data and perform exploratory analysis to get some interesting insights into the process of choosing a hotel. We identified the searches by each user belonging to a specific type of destination. This gave us some useful information about which hotel cluster was finally chosen over other hotel clusters explored by the user. One important observation to note is that few users might be travel agents and could explore multiple type of destinations at the same time. Feature engineering was performed and many new features such as duration, importance features, solo trip or family trip etc. were extracted from dataset. Moreover,we have 149 latent features for each destination; we have applied PCA to extract the most relevant dimensions. We observe that geographical location and the distance between user and the hotel (which we calculated using distance matrix completion method) are the most important features. Next, we visualize the correlation matrix between the features of the training set in Figure 1 and observe following things: 


![Co-relation](https://user-images.githubusercontent.com/26486681/112173212-9a0e4c80-8bb2-11eb-8c0d-ca986ce79812.PNG)



-  hotel_cluster does not seem to have a strong (positive or negative) correlation with any other feature. Thus, methods which model linear relationship between features might not be very successful.

- orig_destination_distance has a positive correlation with duration (constructed using srch_ci and srch _co), which means people who are planning for a long trip tend to go far away from the place of origin.

- hotel_continent and posa_continent (which is from where the booking is done) are negatively correlated. This means that people tend to go to continents different from theirs for vacations.


- duration seems to have a strong positive correlation with is_package. This means that people who tend to book hotel for longer duration usually choose hotels as a part of a package.

- srch_destination_id has a strong correlation with srch_destination_type_id. This is expected as each destination would have an associated type; for example, vacation spot, city, etc.

- duration is also positively correlated with hotel_continent which means certain continents are preferred for longer duration of stay.

- srch_rm_cnt has a very strong correlation with srch_adults_cnt, and to an extent, with srch_children_cnt also. This is expected as people tend to take rooms based on how many families/ couples are there.


## 2. Hotel_Baseline_Recommender.ipynb
The two main families of Recommendation system are Content-Based and Collaborative-Filtering models. Collaborative filtering methods are based on similarity from users and items interaction and content-based filtering methods calculate the similarity of attributes of an items and items. 

 

Collaborative-based filter           |  Content-based filter
:-------------------------:|:-------------------------:
<img src="https://user-images.githubusercontent.com/26486681/112215239-5f6dd980-8bdd-11eb-8ada-8e338c6d2b49.png" width="500" />  | <img src="https://user-images.githubusercontent.com/26486681/112215279-6e548c00-8bdd-11eb-9832-d5e2e42662c0.png" width="300" />  


We have used Collaborative-Filtering model as the baseline model. Collaborative Filtering can be divided into Memory-Based Collaborative Filtering and Model-Based Collaborative filtering. First of all, We have performed some data analysis for removing duplicates data and understand better user item rating distribution. We have created user-item matrices for train and test set. Then performed some data analysis on the rating matrix and transformed rating matrix into average rating matrix (per user/row). Then, two collaborative filtering models were implemented from scratch.

  -    Memory-Based CF by computing cosine similarity: Memory-Based Collaborative Filtering approaches can be divided into two main sections: user-item filtering   and item-item  filtering. In both cases, We created a rating matrix that builds from the entire dataset in order to make recommendations.

   -   Model-Based CF by using singular value decomposition (SVD) and Alternating Least Squares (ALS) method: Model-Based CF models are developed using machine learning algorithms to predict a user's rating of unrated items.  As per my understanding, the algorithms in this approach can further be broken down into 3 sub-types such as Clustering based algorithm, Matrix Factorization, Deep Learning. For example, we can mention few of these algorithms SVD, NMF, KNN etc.

## 3. Hotel_Artificial_Neural_Network_Recommender.ipynb
The aim of this notebook is to build a Recommendation System using with Artificial Neural Network by using Keras and TensorFlow library. First of all, we have created Embedding layer for both user and item and then combine’s embeddings using a dot product. In an embedding model the embeddings are the weights that are learned during training. Then, we have created a fully connected neural network with dense layer and we recommend hotel cluster to a user.

## 4. Hotel_Wide&Deep_Recommender.ipynb 
In the past few decades, deep learning has been tremendously successful in a wide range of applications and has shown state-of-the-art results in recommender architectures as well. In our baseline model (collaborative-filtering recommender system), we used only three features of our large dataset.Therefore, we decided to build hybrid recommender systems so that we can use more features of our large dataset which colud gives us more better recommendations. After going through some research paper, we decided to work on Wide & Deep Learning for Recommender Systems. Wide & Deep Learning [paper](https://arxiv.org/abs/1606.07792)
presents a new framework for recommandation tasks, which combines the advantages of two different type of models: 

  -    Generalized Linear Models (logistic regression): efficient for large scale regression and classification tasks with sparse data, provide nice interpretations;
  -    Deep Neural Networks: extract more complicate feature interactions and less feature engineering.
  
  The spectrum of Wide & Deep models given below.
  
  
![Wide Deep_Model](https://user-images.githubusercontent.com/26486681/112028732-bb5d3300-8af5-11eb-945b-8dfbc53cdaac.png)

 
The wide and deep learning has two individual components. The wide network is a linear estimator or a single layer feed-forward network which assigns weights to each features and adds bias to them to model the matrix factorization method, as illustrated in above figure (left). These feature set includes raw input features and transformed. The deep component is a feed-forward neural network, as shown in above figure (right). The high dimensional categorical features are first convert into a low dimensional and dense real-valued vector, often referred as embeddings. The embedding vectors are initialized randomly and then the values are trained to minimize the final loss function. Then they are fed into the hidden layers with feed forward step. By jointly training the wide and deep network, it takes the weighted sum of the outputs from both wide model and deep model as the prediction value. However, we implemented the wide and deep model by using [shuoranly
/
DeepCTR-1](https://github.com/shuoranly/DeepCTR-1) package.

## 5. Hotel_Deep_Factorization_Machine_Recommender.ipynb 
As an extension of the Wide and Deep Learning approach, [DeepFM: A Factorization-Machine Based Neural Network for CTR Prediction](https://arxiv.org/pdf/1703.04247.pdf)is an end-to-end model that seamlessly integrates Factorization Machine (the wide component) and Multi-Layer Perceptron (the deep component). Compared to the Wide and Deep Model from Google, DeepFM does not require tedious feature engineering.


<img src="https://user-images.githubusercontent.com/26486681/112199932-edd95f80-8bcb-11eb-90ce-7922c47b3e88.png" width="1000" />     

Wide & deep architecture of the DeepFM framework ( figure left). The wide and deep component share the same input raw feature vector, which enables DeepFM to learn low-and high-order feature interactions simultaneously from the input raw features. The wide component of DeepFM is an FM layer and the Deep Component of DeepFM can be any neural network. 

## 6. Evaluation Metrics for Recommender Systems
There are many methods for evaluating a Recommender Systems, but in this research, the model performance is judged by the most commonly used measures.

•	Mean Absolute Error (MAE)- The mean of the absolute value of the errors.   

•	Mean Squared Error (MSE)- The mean of the squared errors. 

•	Root Mean Squared Error (RMSE) - The square root of the mean of the squared errors. 

•	Area under the ROC Curve (AUC) - AUC measures the entire two-dimensional area underneath the entire ROC curve. 

• Precision and Recall are binary metrics used to evaluate models with binary output. Thus we need a way to translate our numerical problem (ratings usually from 1 to 5) into a binary problem (relevant and not relevant items). To do the translation we will assume that any true rating above 3.5 corresponds to a relevant item and any true rating below 3.5 is irrelevant. We are most likely interested in recommending top-N items to the user. So it makes more sense to compute precision and recall metrics in the first N items instead of all the items. Thus the notion of precision and recall at k where k is a user definable integer that is set by the user to match the top-N recommendations objective.

• P@K - Precision at k is the proportion of recommended items in the top-k set that are relevant. 

• R@K - Recall at k is the proportion of relevant items found in the top-k recommendations.

• MRR (Mean Reciprocal Rank) - Average reciprocal hit ratio (ARHR). The relevance score is either 0 or 1, for items not bought or bought (not clicked or clicked, etc.).




## 7. Making Recommendations 
Finally, we recommended top 5 hotel cluster each of the users. The following table, shows the recommendation for user_id= 1048.

<!-- TABLE_GENERATE_START -->

| User ID       | Hotel Cluster (item_id) |
| ------------- | ------------- |
| User_id 1048 recommended  | hotel cluster:  45 |
| User_id 1048 recommended  | hotel cluster:  17 |
| User_id 1048 recommended  | hotel cluster:  40 |
| User_id 1048 recommended  | hotel cluster:  11 |
| User_id 1048 recommended  | hotel cluster:  98 |

<!-- TABLE_GENERATE_END -->
