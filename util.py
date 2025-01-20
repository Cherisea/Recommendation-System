"""
    A bunch of supporting functions for implementing content-based filtering and collaborative filtering
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

base_path = "/Users/joeycherisea/Downloads/ml-latest"
movie_filename = "movies.csv"
rating_filename = "ratings.csv"

movie_data = pd.read_csv(os.path.join(base_path, movie_filename),
                         usecols=["title", "movieId"],
                         dtype={"title": "str", "movieId": "int32"})

rating_data = pd.read_csv(os.path.join(base_path, rating_filename),
                          usecols=["userId", "movieId", "rating"],
                          dtype={"userId": "Int32", "movieId": "Int32", "rating": "float32"})

# Count frequency of rating of each movie in descending order
rating_freq = rating_data.groupby(
    "movieId").size().sort_values(ascending=False)


def draw_graph(narr, save_pic: bool = False):
    """
        Function: plot a graph from data and optionally save it as a picture

        narr: a numpy array 
        save_pic: whether to save the plot as a picture
    """
    plt.figure(figsize=(12, 6))

    # Plot a line chart
    plt.plot(narr.values, color="orange", linewidth=2)

    # Add a legend to chart
    plt.legend(["count"], loc="upper right")

    # Set label for x axis
    plt.xlabel("Movies ID (Ordered by Popularity)")

    # Set label for y axis
    plt.ylabel("Number of Ratings")

    # Set x ticks
    plt.xticks([10000, 20000, 30000, 40000, 50000])

    # Set graph title
    plt.title("Distribution of Movie Ratings")

    # (Optional) Stack statistics on the graph
    descript_text = rating_freq.describe().to_string()
    plt.figtext(0.75, 0.5, descript_text, ha="left", va="center", wrap=True,
                fontdict={"color": "red", "size": 10})

    # plt.tight_layout()

    # Optionally save plot as a picture
    if save_pic:
        plt.savefig("rating_distri.png")

    # Display the plot
    plt.show()
