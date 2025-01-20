"""
    Implement collaborative filtering with alternating least squares algorithm 
    for matrix factorization provided by Spark
    
    ============================ Program Workflow ============================
    1, Launch a spark session
    2, Initialize an ALSRecommender
    3, Set ALS model hyperparameters
    4, Invoke a chain of methods in ALSRecommender class:
        make_recommendations() => _infer() => _match_movie() => _append_rating() => _create_inference_data()
        => model.transform()
    5, Report recommendations of movie titles
    6, Terminate spark
"""
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import SparkSession, Row
from pyspark.sql.functions import lower, col

import argparse
import os
import time
import gc


class ALSRecommender():
    """
        A recommending class based on ALS model from Spark
    """

    def __init__(self, spark_session, movie_path, rating_path) -> None:
        """
            spark_session: a SparkSession object for interacting with Spark clusters
            movie_path: path to movie file
            rating_path: path to rating file
        """
        self.spark = spark_session
        self.spark_context = spark_session.sparkContext
        self.spark_context.setLogLevel('OFF')
        self.movie_df = self._load_file(
            movie_path).select(['movieId', 'title'])
        self.rating_df = self._load_file(rating_path).select(
            ['userId', 'movieId', 'rating']).sample(fraction=0.00255)
        self.model = ALS(userCol='userId', itemCol='movieId',
                         ratingCol='rating', coldStartStrategy='drop')
        self.next_userId = self.rating_df.agg(
            {'userId': 'max'}).collect()[0][0] + 1
        self.infer_time = 0

    def _load_file(self, filepath):
        """
            Function: Read file as a Spark dataframe

            filepath: path to csv file
            load_ration: fraction of rows in csv file to load
        """
        return self.spark.read.csv(filepath, header=True, inferSchema=True)

    def set_hyperparam(self, iter, rank, reg_para):
        """
            Manually set the hyperparameters of an ALS model

            iter: number of iterations
            rank: number of latent features
            reg_para: lambda coefficient of L2 regularization
        """
        self.model = self.model.setMaxIter(
            iter).setRank(rank).setRegParam(reg_para)

    def _match_movie(self, movie_name):
        """
            Retrieve movie ID from spark dataframe based on movie name

            movie_name: name of movie to search for
        """
        # Check if given movie_name is legit
        if not movie_name or not isinstance(movie_name, str):
            print("Supplied movie name is invalid. Try agin.")
            exit(1)

        print(f"You have entered: {movie_name}")
        movie_match = self.movie_df.filter(
            lower(col('title')).contains(movie_name.lower())
        ).select('movieId', 'title')

        # Return empty list if no match found
        if not len(movie_match.take(1)):
            print("Oops, no match is found!")
            return []
        else:
            movie_ids = movie_match.rdd.map(lambda x: x[0]).collect()
            titles = movie_match.rdd.map(lambda x: x[1]).collect()
            print(f"Possible Matches: {[title for title in titles]}\n")
            return movie_ids

        # Ask user to confirm if there are multiple fuzzy matches
        # if len(titles) > 1:
        #     print("Multiple matches found\n")
        #     for idx, title in enumerate(titles):
        #         print(f"{idx}: {title}")
        #     choice = int(input("Select a number: "))
        #     return ids[choice]

        # Otherwise return its id
        # print(f"Possible match: {titles}")
        # return ids

    def _append_rating(self, user_id, movie_id):
        """
            Function: append new user rating to existing rating dataframe

            user_id: id of new user
            movie_id: id of an input movie
        """
        # Create a user RDD
        user_rdd = self.spark_context.parallelize(
            [(user_id, m_id, 5.0) for m_id in movie_id])

        # Convert RDD into spark dataframe row
        user_row = user_rdd.map(lambda x: Row(
            userId=int(x[0]),
            movieId=int(x[1]),
            rating=float(x[2])
        )
        )

        # Create spark dataframe from row
        user_df = self.spark.createDataFrame(
            user_row).select(self.rating_df.columns)

        # Append user dataframe to the existing one
        self.rating_df = self.rating_df.union(user_df)

    def _create_inference_data(self, user_id, movie_id):
        """
            Function: generate data to predict user rating on unwatched movies 

            movie_id: id of movie user already watched
        """
        # Gather ids of movies user hasn't watched
        other_movieIds = self.movie_df \
            .filter(~col('movieId').isin(movie_id))    \
            .select('movieId') \
            .rdd.map(lambda x: x[0])   \
            .collect()

        # Generate dataframe rows of (userId, movieId)
        new_rdd_row = self.spark_context.parallelize(
            [(user_id, m_id) for m_id in other_movieIds]
        ).map(
            lambda x: Row(userId=int(x[0]), movieId=int(x[1]))
        )

        # Convert rows into dataframe
        infer_df = self.spark.createDataFrame(
            new_rdd_row).select(['userId', 'movieId'])

        return infer_df

    def _infer(self, model, fav_movie, top_k):
        """
            Function: compute preliminary recommendations of (movieID, predictions) pairs
        """
        # Assign a userId to new user and increment max userId
        new_userId = self.next_userId
        self.next_userId += 1

        # Get movieId of favorite movie
        fav_movidId = self._match_movie(fav_movie)

        # Append ratinggs of new user to rating dataframe
        self._append_rating(new_userId, fav_movidId)

        # Retrain model after updating rating dataframe (NECESSARY?)
        model = model.fit(self.rating_df)

        # Get data of (new_userId, movieId) pair to run inference on
        inference_df = self._create_inference_data(new_userId, fav_movidId)

        # Infer ratings
        return model.transform(inference_df) \
            .select(['movieId', 'prediction']) \
            .orderBy('prediction', ascending=False) \
            .rdd.map(lambda x: (x[0], x[1])) \
            .take(top_k)

    def make_recommendation(self, fav_movie, top_k):
        """
            Function: make movie recommendations based on given user-supplied movie

            fav_movie: name of favorite movie
            top_k: number of top recommendations
        """
        print("\n================= Recomendation System  Work in Progress =======================")
        start_time = time.time()

        # Get a list of recommended movie IDs and their predicted ratings
        raw_recommends = self._infer(self.model, fav_movie, top_k)

        # Retrieve predicted movie ids and their ratings
        movie_ids = [row[0] for row in raw_recommends]
        predict_ratings = [row[1] for row in raw_recommends]

        # Report time spent on inference
        self.infer_time = time.time() - start_time

        # Fetch movie titles from movie_ids
        movie_titles = self.movie_df \
            .filter(col('movieId').isin(movie_ids)) \
            .select('title') \
            .rdd.map(lambda x: x[0]) \
            .collect()

        # Report recommendations
        print(f"Top {top_k} Movie Recommendations Based on {fav_movie}")
        for i in range(len(movie_titles)):
            print(f"{i+1}: {movie_titles[i]}, rating {predict_ratings[i]:.3f}")
        self.display_statis()

    def display_statis(self):
        """
            Function: print out statistics on data and inference duration
        """
        print("\nInference Statistics")
        print(f"Rows in Data: {self.rating_df.count():,}")
        print(f"Total Time: {self.infer_time:.3f} seconds")


def tune_model(rating_data, max_iter, ranks, reg_paras, split_ratio=[0.6, 0.2, 0.2]):
    """
        Function: get the best ALS model from a predefined range of ranks and reg_paras

        rating_data: a spark dataframe containing rating data
        max_iter: number of iterations to run
        ranks: list of ints for the number of latent features
        reg_paras: list of floats for lambda of L2 regularization
        split_ratio: split ratio of (train, validate, test) for rating_data
    """
    # Split rating data based on split ratio
    train_data, validate_data, test_data = rating_data.randomSplit(
        split_ratio)

    # Train ALS model
    model = ALS(userCol='userId', itemCol='movieId',
                ratingCol='rating', coldStartStrategy='drop')
    best_model = tune_als(model, train_data, validate_data,
                          max_iter, ranks, reg_paras)

    # Evaluate its performance on out-of-sample data
    predictions = best_model.transform(test_data)
    evaluator = RegressionEvaluator(predictionCol='prediction',
                                    labelCol='rating',
                                    metricName='rmse')
    rmse = evaluator.evaluate(predictions)

    # Report result
    print(f"RMSE of the best model on test data: {rmse}")

    # clean up
    del train_data, validate_data, test_data, evaluator
    gc.collect()
    return best_model


def tune_als(model, train_data, validate_data, max_iter, ranks, reg_paras):
    """
        Function: train an ALS model on train_data and evaluate model on validate_data 
                  using RSME cost function 

        model: a spark ML ALS model 
        train_data: spark dataframe of the format (userId, itemId, rating)
        validate_data: spark dataframe of the format (userId, itemId, rating)
        max_iter: number of iterations to run
        ranks: list of ints for the number of latent features
        reg_paras: list of floats for lambda of L2 regularization

        Return: an optimal spark ALS model 
    """
    # Initialize starting values of hyperparameters
    min_error = float('inf')
    best_rank = -1
    best_reg = 0
    best_model = None

    for rank in ranks:
        for reg in reg_paras:
            # Set model parameter
            als = model.setMaxIter(max_iter).setRank(rank).setRegParam(reg)

            # Train model with train_data
            trained_model = als.fit(train_data)

            # Returns an updated df with a 'prediction' column
            predictions = trained_model.transform(validate_data)
            evaluator = RegressionEvaluator(predictionCol='prediction',
                                            labelCol='rating',
                                            metricName='rmse')
            # Evaluate predicted ratings with actual ratings
            rmse = evaluator.evaluate(predictions)
            print(
                f"{rank} latent features with regularization parameter {reg}: RMSE {rmse}")

            # Update hyperparameters after training with each combination of rank and reg
            if rmse < min_error:
                min_error = rmse
                best_rank = rank
                best_reg = reg
                best_model = trained_model

    print(
        f"Best ALS model: {best_rank} latent features, {best_reg} regularization parameter, RMSE {min_error}")
    return best_model


def parse_arg():
    """
        A user-friendly command-line parser

        Return: a namespace of user passed arguments
    """
    parser = argparse.ArgumentParser(
        prog="ALS Movie Recommder", description="Uncover movies you don't know you love")
    parser.add_argument("--file_path", nargs="?",
                        default="/Users/joeycherisea/Downloads/ml-latest/", help="base path of data files")
    parser.add_argument("--movie_file", nargs="?",
                        default="movies.csv", help="movie file name")
    parser.add_argument("--rating_file", nargs="?",
                        default="ratings.csv", help="rating file name")
    parser.add_argument("--movie_name", nargs="?", default="",
                        help="one of your favorite movies")
    parser.add_argument("--top_k", nargs=1, type=int, default=10,
                        help="number of movie recommendations")
    return parser.parse_args()


if __name__ == "__main__":
    # Retrieve all command-line arguments
    args = parse_arg()
    file_path = args.file_path
    movie_file = args.movie_file
    rating_file = args.rating_file
    fav_movie = args.movie_name
    top_k = args.top_k[0]

    # Launch a Spark session
    spark = SparkSession.builder    \
        .appName("Movie Recommender")   \
        .getOrCreate()

    # Initialize a recommender
    recommender = ALSRecommender(spark, os.path.join(file_path, movie_file),
                                 os.path.join(file_path, rating_file))

    # Set recommender hyperparameters
    recommender.set_hyperparam(10, 20, 0.05)

    # Make recommendations
    recommender.make_recommendation(fav_movie, top_k)

    # Terminate spark session
    spark.stop()
