"""
    Implementation of Content-Based Filtering(CBF) using sklearn
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import pandas as pd
import timeit


def read_data(filepath):
    """
        Read a csv file containing item information

        filepath: path to a csv file
    """
    try:
        file = pd.read_csv(filepath)
        return file
    except IOError as e:
        print(e)


def compute_similarity(feature_col, title_idx):
    """
        Calculate on-demand cosine similarity matrix for a given title

        feature_col: feature column
        title_idx: index of a previously watched movie
    """
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(feature_col)

    # Compute cosine similarity between one title and all titles
    cosine_similar = cosine_similarity(
        tfidf_matrix[title_idx], tfidf_matrix).flatten()
    return cosine_similar


def cbf_recommend(top_k: int, filepath: str, record: str, title_col: str, feature_col: str):
    """
        Function: recommend top k titles based on user' watch records

        top_k: number of recommendations
        filepath: path to a csv file containing all data
        record: previous title a user has watched
        title_col: column name of movie titles in file
    """
    data = read_data(filepath)

    # Get index of record
    title_idx = data[data[title_col].str.lower().str.contains(record, na=False, regex=False)].index

    # Ask user to select a best match if multiple results are found
    if len(title_idx) > 1:
        print("\nMultiple results found:")
        for idx, t_idx in enumerate(title_idx):
            print(f"{idx+1}: {data[title_col][t_idx]}")
        choice = int(input("Enter movie number: "))
        final_idx = title_idx[choice]
    elif len(title_idx) == 0:
        print("\nNo matches found. Exiting...")
        exit(1)
    else:
        final_idx = title_idx

    # Compute similarity score between record and all other movies
    simil_tuple = enumerate(compute_similarity(data[feature_col], final_idx))

    # Get a list of (title_index, score) tuple
    cosine_score = [(index, score)
                    for index, score in simil_tuple if index != final_idx]

    # Obtain top k recommendations based on cosine score
    top_recommend = sorted(
        cosine_score, key=lambda x: x[1], reverse=True)[:top_k]
    return [(i, data[title_col][i], score) for i, score in top_recommend]


def main():
    filepath = input("Movie Filepath: ")
    record = input("One movie you have watched: ").lower()
    title_column = input("Title Column: ")
    feature_column = input("Feature Column: ")
    top_k = input("Number of Recommendations: ")

    # Measure execution time
    start_time = timeit.default_timer()
    res = cbf_recommend(int(top_k), filepath, record,
                        title_column, feature_column)
    end_time = timeit.default_timer()

    # Display results
    print(
        f"\n===================== Top {top_k} Recommendations Based on Your View History =====================")
    for i, t, s in res:
        print(f"Movie ID: {i}\n\tTitle: {t}\n\tSimilarity Score: {s:.3f}\n")
    print(f"Execution Time: {end_time - start_time:.3f} seconds")


if __name__ == "__main__":
    main()
