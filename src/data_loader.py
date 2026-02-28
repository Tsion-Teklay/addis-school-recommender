import pandas as pd


def load_raw_data():
    schools = pd.read_csv("data/raw/schools.csv")
    parents = pd.read_csv("data/raw/parents.csv")
    interactions = pd.read_csv("data/raw/interactions.csv")

    return schools, parents, interactions


def basic_validation(schools, parents, interactions):
    assert not schools.empty, "Schools dataset is empty!"
    assert not parents.empty, "Parents dataset is empty!"
    assert not interactions.empty, "Interactions dataset is empty!"

    assert schools["school_id"].is_unique, "Duplicate school IDs!"
    assert parents["parent_id"].is_unique, "Duplicate parent IDs!"

    print("Basic validation passed.")