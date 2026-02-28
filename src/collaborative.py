import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from data_loader import load_raw_data


def build_user_item_matrix(interactions):
    return interactions.pivot_table(
        index="parent_id",
        columns="school_id",
        values="rating"
    ).fillna(0)


def recommend_collaborative(parent_id, top_n=5):
    schools, parents, interactions = load_raw_data()

    user_item_matrix = build_user_item_matrix(interactions)

    if parent_id not in user_item_matrix.index:
        return pd.DataFrame({"message": ["Parent not found in interactions."]})

    # Compute similarity between parents
    similarity_matrix = cosine_similarity(user_item_matrix)
    similarity_df = pd.DataFrame(
        similarity_matrix,
        index=user_item_matrix.index,
        columns=user_item_matrix.index
    )

    # Get similar parents
    similar_parents = similarity_df[parent_id].sort_values(ascending=False)

    # Remove self
    similar_parents = similar_parents.drop(parent_id)

    # Get top similar parents
    top_similar = similar_parents.head(10)

    # Weighted scoring
    parent_ratings = user_item_matrix.loc[parent_id]

    scores = {}

    for other_parent, similarity_score in top_similar.items():
        other_ratings = user_item_matrix.loc[other_parent]

        for school_id, rating in other_ratings.items():
            if parent_ratings[school_id] == 0 and rating > 0:
                scores.setdefault(school_id, 0)
                scores[school_id] += similarity_score * rating

    if not scores:
        return pd.DataFrame({"message": ["No collaborative recommendations available."]})

    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    top_school_ids = [school_id for school_id, _ in sorted_scores[:top_n]]

    recommended = schools[schools["school_id"].isin(top_school_ids)]

    return recommended


if __name__ == "__main__":
    recs = recommend_collaborative(parent_id=1, top_n=5)
    print(recs)