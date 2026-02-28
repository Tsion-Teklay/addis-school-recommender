import pandas as pd
import numpy as np
import random
from config import SUBCITY_COORDINATES, SCHOOL_TYPES, STREAMS
from pathlib import Path

np.random.seed(42)
random.seed(42)


# -----------------------------
# Utility Functions
# -----------------------------

def ensure_directory():
    Path("data/raw").mkdir(parents=True, exist_ok=True)


def euclidean_distance(coord1, coord2):
    return np.sqrt((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2)


# -----------------------------
# Generate Schools
# -----------------------------

def generate_schools(n_schools=300):
    schools = []

    for i in range(1, n_schools + 1):
        subcity = random.choice(list(SUBCITY_COORDINATES.keys()))
        school_type = random.choice(SCHOOL_TYPES)
        stream = random.choice(STREAMS)

        if school_type == "Public":
            annual_fee = random.randint(2000, 8000)
        elif school_type == "Private":
            annual_fee = random.randint(15000, 40000)
        else:
            annual_fee = random.randint(60000, 120000)

        ranking_score = random.randint(50, 100)
        facilities_score = random.randint(40, 100)
        teacher_quality = random.randint(50, 100)

        schools.append([
            i,
            f"School_{i}",
            subcity,
            school_type,
            annual_fee,
            stream,
            ranking_score,
            facilities_score,
            teacher_quality
        ])

    df = pd.DataFrame(schools, columns=[
        "school_id", "name", "subcity", "type",
        "annual_fee", "stream",
        "ranking_score", "facilities_score", "teacher_quality"
    ])

    return df


# -----------------------------
# Generate Parents
# -----------------------------

def generate_parents(n_parents=2000):
    parents = []

    for i in range(1, n_parents + 1):
        subcity = random.choice(list(SUBCITY_COORDINATES.keys()))
        budget = random.randint(5000, 80000)
        child_avg = random.randint(50, 100)
        preferred_stream = random.choice(STREAMS)
        preferred_type = random.choice(SCHOOL_TYPES)

        parents.append([
            i,
            subcity,
            budget,
            child_avg,
            preferred_stream,
            preferred_type
        ])

    df = pd.DataFrame(parents, columns=[
        "parent_id", "subcity", "budget",
        "child_avg", "preferred_stream", "preferred_type"
    ])

    return df


# -----------------------------
# Generate Interactions
# -----------------------------

def generate_interactions(parents_df, schools_df, n_interactions=10000):
    interactions = []

    for _ in range(n_interactions):
        parent = parents_df.sample(1).iloc[0]
        school = schools_df.sample(1).iloc[0]

        # Budget constraint
        if school["annual_fee"] > parent["budget"]:
            continue

        # Subcity proximity score
        parent_coord = SUBCITY_COORDINATES[parent["subcity"]]
        school_coord = SUBCITY_COORDINATES[school["subcity"]]
        distance = euclidean_distance(parent_coord, school_coord)

        proximity_score = max(0, 10 - distance)

        # Stream match
        stream_score = 10 if parent["preferred_stream"] == school["stream"] else 5

        # Quality score
        quality_score = (
            school["ranking_score"] +
            school["facilities_score"] +
            school["teacher_quality"]
        ) / 30

        raw_score = (0.4 * proximity_score +
                     0.3 * stream_score +
                     0.3 * quality_score)

        rating = min(5, max(1, round(raw_score / 2)))

        interactions.append([
            parent["parent_id"],
            school["school_id"],
            rating
        ])

    df = pd.DataFrame(interactions, columns=[
        "parent_id", "school_id", "rating"
    ])

    return df


# -----------------------------
# Main Execution
# -----------------------------

def main():
    ensure_directory()

    schools_df = generate_schools()
    parents_df = generate_parents()
    interactions_df = generate_interactions(parents_df, schools_df)

    schools_df.to_csv("data/raw/schools.csv", index=False)
    parents_df.to_csv("data/raw/parents.csv", index=False)
    interactions_df.to_csv("data/raw/interactions.csv", index=False)

    print("Dataset generated successfully!")


if __name__ == "__main__":
    main()