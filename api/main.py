import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from fastapi import FastAPI, HTTPException, Query
from hybrid import hybrid_recommend
import uvicorn

app = FastAPI(
    title="Addis Ababa School Recommender",
    description="Hybrid recommendation API for parents choosing schools in Addis Ababa",
    version="1.0.0"
)

@app.get("/recommend")
def recommend(parent_id: int = Query(..., description="Parent ID"),
              top_n: int = Query(5, description="Number of recommendations")):
    """
    Return top-N recommended schools for a parent.
    """
    try:
        recs = hybrid_recommend(parent_id=parent_id, top_n=top_n)
        if recs.empty:
            return {"message": "No recommendations available for this parent."}
        return recs.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# -----------------------------
# Run locally
# -----------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)