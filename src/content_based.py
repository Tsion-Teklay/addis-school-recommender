import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from config import SUBCITY_COORDINATES
from data_loader import load_raw_data


def euclidean_distance(coord1, coord2):
    return np.sqrt((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2)


def compute_proximity_score(parent_subcity, school_subcity):
    parent_coord = SUBCITY_COORDINATES[parent_subcity]
    school_coord = SUBCITY_COORDINATES[school_subcity]
    distance = euclidean_distance(parent_coord, school_coord)
    return max(0, 10 - distance)


def recommend_schools(parent_id, top_n=5):
    schools, parents, _ = load_raw_data()

    parent = parents[parents["parent_id"] == parent_id].iloc[0]

    candidate_schools = schools[schools["annual_fee"] <= parent["budget"]].copy()

    if candidate_schools.empty:
        return pd.DataFrame({"message": ["No schools within budget."]})

    scores = []

    for _, school in candidate_schools.iterrows():

        # Stream similarity
        stream_score = 1 if parent["preferred_stream"] == school["stream"] else 0.5

        # Type similarity
        type_score = 1 if parent["preferred_type"] == school["type"] else 0.5

        # Academic fit (child_avg vs ranking)
        academic_score = 1 - abs(parent["child_avg"] - school["ranking_score"]) / 100

        # Proximity score
        proximity_score = compute_proximity_score(
            parent["subcity"],
            school["subcity"]
        ) / 10

        # School quality
        quality_score = (
            school["ranking_score"] +
            school["facilities_score"] +
            school["teacher_quality"]
        ) / 300

        final_score = (
            0.25 * stream_score +
            0.20 * type_score +
            0.20 * academic_score +
            0.20 * proximity_score +
            0.15 * quality_score
        )

        scores.append(final_score)

    candidate_schools["score"] = scores

    recommendations = candidate_schools.sort_values(
        by="score", ascending=False
    ).head(top_n)

    return recommendations[[
        "school_id", "name", "subcity", "type",
        "annual_fee", "stream", "score"
    ]]


if __name__ == "__main__":
    recs = recommend_schools(parent_id=1, top_n=5)
    print(recs)