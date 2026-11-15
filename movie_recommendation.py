# ==========================================
# MOVIE RECOMMENDATION SYSTEM
# Content-Based Filtering using TF-IDF
# ==========================================

import csv
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DATASET_PATH = Path(__file__).with_name("movies.csv")


def load_movies():
    with DATASET_PATH.open("r", encoding="utf-8", newline="") as dataset_file:
        reader = csv.DictReader(dataset_file)
        expected_columns = {"title", "genres", "description"}
        if not expected_columns.issubset(reader.fieldnames or []):
            raise ValueError(
                "movies.csv must contain title, genres, and description columns."
            )

        movies = [
            row
            for row in reader
            if row["title"].strip()
            and row["genres"].strip()
            and row["description"].strip()
        ]

    if not movies:
        raise ValueError("movies.csv does not contain any complete movie records.")
    return movies


def main():
    movies = load_movies()
    features = [
        f"{movie['genres']} {movie['description']}"
        for movie in movies
    ]

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(features)
    similarity = cosine_similarity(tfidf_matrix)

    print("Movie dataset loaded successfully!")
    print(f"\nTotal Movies: {len(movies)}")
    print(f"TF-IDF Matrix Shape: {tfidf_matrix.shape}")
    print("Cosine similarity calculated successfully!")
    print("\nAVAILABLE MOVIES:")
    for movie in movies:
        print(f"- {movie['title']}")

    while True:
        print("\nEnter 'exit' to close the system.")
        user_movie = input("\nEnter a movie name: ").strip()
        if user_movie.lower() == "exit":
            print("\nThank you for using Movie Recommendation System!")
            break

        matches = [
            index
            for index, movie in enumerate(movies)
            if movie["title"].strip().casefold() == user_movie.casefold()
        ]
        if not matches:
            print("\nMovie not found! Please enter a movie from the dataset.")
            continue

        movie_index = matches[0]
        sorted_scores = sorted(
            enumerate(similarity[movie_index]),
            key=lambda item: item[1],
            reverse=True,
        )

        print("\n================================")
        print("RECOMMENDED MOVIES")
        print("================================")
        recommendations = [
            (index, score)
            for index, score in sorted_scores
            if index != movie_index
        ][:5]
        for rank, (index, score) in enumerate(recommendations, start=1):
            movie = movies[index]
            print(f"{rank}. {movie['title']}")
            print(f"   Genres: {movie['genres']}")
            print(f"   Similarity: {score * 100:.2f}%\n")


if __name__ == "__main__":
    main()
