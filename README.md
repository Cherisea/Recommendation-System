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