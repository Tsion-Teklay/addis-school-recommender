import pandas as pd
import numpy as np
from content_based import recommend_schools as content_recommend
from collaborative import recommend_collaborative as collab_recommend
from data_loader import load_raw_data

# -----------------------------
# Hybrid Recommendation
# -----------------------------

def hybrid_recommend(parent_id, top_n=5, content_weight=0.5, collab_weight=0.4, quality_weight=0.1):
    """
    Combine content-based and collaborative recommendations.
    
    Parameters:
        parent_id: int - ID of the parent
        top_n: int - number of schools to recommend
        content_weight: float - weight of content-based score
        collab_weight: float - weight of collaborative score
        quality_weight: float - weight of school quality score
    """

    # Load schools for reference
    schools, _, _ = load_raw_data()

    # -----------------------------
    # Get content-based recommendations
    # -----------------------------
    content_df = content_recommend(parent_id, top_n=50)  # Take top 50 for scoring
    if "score" not in content_df.columns:
        content_df["score"] = 0
    content_df = content_df.set_index("school_id")

    # -----------------------------
    # Get collaborative recommendations
    # -----------------------------
    collab_df = collab_recommend(parent_id, top_n=50)
    if "score" not in collab_df.columns:
        collab_df = collab_df.copy()
        collab_df["score"] = 0
    collab_df = collab_df.set_index("school_id")

    # -----------------------------
    # Merge scores
    # -----------------------------
    combined = pd.DataFrame(index=schools["school_id"])
    combined = combined.join(content_df[["score"]].rename(columns={"score": "content_score"}))
    combined = combined.join(collab_df[["score"]].rename(columns={"score": "collab_score"}))

    combined.fillna(0, inplace=True)

    # -----------------------------
    # Add quality score (normalized)
    # -----------------------------
    schools_indexed = schools.set_index("school_id")
    combined["quality_score"] = (
        schools_indexed["ranking_score"] +
        schools_indexed["facilities_score"] +
        schools_indexed["teacher_quality"]
    ) / 300  # normalize to 0-1

    # -----------------------------
    # Compute final hybrid score
    # -----------------------------
    combined["hybrid_score"] = (
        combined["content_score"] * content_weight +
        combined["collab_score"] * collab_weight +
        combined["quality_score"] * quality_weight
    )

    # -----------------------------
    # Apply budget constraint
    # -----------------------------
    parent_budget = load_raw_data()[1].loc[load_raw_data()[1]["parent_id"] == parent_id, "budget"].values[0]
    affordable_schools = schools_indexed[schools_indexed["annual_fee"] <= parent_budget].index
    combined = combined.loc[combined.index.intersection(affordable_schools)]

    # -----------------------------
    # Return top-N recommendations
    # -----------------------------
    top_recommendations = combined.sort_values("hybrid_score", ascending=False).head(top_n)
    top_recommendations = top_recommendations.join(schools_indexed[[
        "name", "subcity", "type", "annual_fee", "stream"
    ]])

    return top_recommendations[[
        "name", "subcity", "type", "annual_fee", "stream", "hybrid_score"
    ]]


# -----------------------------
# Test the hybrid recommender
# -----------------------------
if __name__ == "__main__":
    parent_id_test = 1
    recs = hybrid_recommend(parent_id_test, top_n=5)
    print(recs)