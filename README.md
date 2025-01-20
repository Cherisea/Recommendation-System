# Introduction
Recommendation system lies at the forefront of modern digital age characterized by information overload. 
It's also an essentail tool for social media plaforms to enhance user engagement and retention. 
However, developing an effective recommendation algorithm is presented with several major challenges:

- Cold start: there is little interaction data on new users or items;
- Data sparsity: most users don't rate or provide explicit feedback on items they iteract with, resulting in a sparse user-item matrix;
- Diversity VS accuracy: how to strike the right balance between recommending relevant items and a diverse range of items;
- Scalability: computational efficiency is key with millions or even billions of users and items;

This project aims to explore this topic by implementing a two-stage framework that integrates with the latest two-tower DNN model. 
Since real-time data on social platforms is usually considered an invaluable asset, thus charging a substantial use fee, 
we'll use the [MovieLens](https://grouplens.org/datasets/movielens/32m/) for our experiments. To put the performance of our two-stage algorithm into context, 
we'll also conduct experiments with two well-known baseline algorithms -- content-based filtering and collaborative filtering.

# Content-based Filtering(CBF)
An early technique of recommendation systems, CBF relies on attributes of items to make recommendations that are similar to what a user has liked before. The underlining assumption is that a user is quite likely to prefer an item that resembles another item he liked by nature. 

## Workflow
- Select a feature or features that define an item;
- Provide an item that user liked before;
- Calculate the pair-wise cosine similarity between all items in stock and the given item based on selected features;
- Rank cosine similarity scores in descreasing order;
- Return top n items;

# Collaborative Filtering(CF) with ALS
CF operates under the assumption that if two users have similar taste and one user likes an item that another hasn't interacted with yet, then it's highly likely that the other user will be interested in that item as well. This program focuses on one of the two types of CF -- model-based CF implemented with ALS matrix factorization algorithm, which enjoys two major advantages over the other type of CF -- memory-based CF:
- Data sparsity: designed to tackle the practical challenge of sparse user-item interaction matrix, ALS predicts user ratings by tuning the number of latent factors that decompose a higher dimension matrix into two lower dimension matrix;
- Scalability: unlike memory-based CF, ALS runs gradient descent across multiple partitions of underlying datasets from a cluster of machines;

## Workflow
- Launch a Spark sessin;
- Initialize an ALSRecommender;
- Tune ALS model via gridsearch or set ALS model hyperparameters;
- Ask ALSRecommender to make recommendations based on user input;
- Return top n items;
- Terminate Spark;