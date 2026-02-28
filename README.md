

Markdown
# Addis Ababa School Recommender System

![Python](https://img.shields.io/badge/python-3.12+-blue)
![License](https://img.shields.io/badge/license-MIT-green)

**A modular hybrid recommendation system to help parents in Addis Ababa select the best schools for their children, combining both content-based and collaborative filtering methods. Built in pure Python and deployed via FastAPI.**

---

## 🚀 Features

- **Data Generation:** Synthetic datasets for ~300 schools, ~2000 parents, and ~10,000 parent–school interactions.
- **Flexible Preprocessing:** One-hot encoding for categories, normalization for numbers, and reusable preprocessor persistence.
- **Content-Based Filtering:** Ranks schools by parent preferences, proximity, streams, types, and school quality.
- **Collaborative Filtering:** Parent similarity via cosine distance and weighted scoring, using only Python and scikit-learn.
- **Hybrid Recommendation:** Combines multiple scores with adjustable weights, respects budgets.
- **FastAPI API:** Live, documented recommendations at `/recommend?parent_id=<id>&top_n=<n>`.
- **Production-Ready:** Modular, extendable, easily maintainable codebase.

---

## 📂 Project Structure

```bash 
addis-school-recommender/ 
├─ api/ 
│ └─ main.py # FastAPI app 
├─ data/ 
│ ├─ raw/ # Raw CSV datasets 
│ └─ processed/ # Processed files 
├─ models/ # Model artifacts or preprocessors 
├─ src/ 
│ ├─ content_based.py 
│ ├─ collaborative.py 
│ ├─ hybrid.py 
│ ├─ data_loader.py 
│ ├─ preprocessing.py 
│ └─ data_generator.py 
├─ requirements.txt 
└─ README.md

```

---

## ⚡ Quickstart

```bash
# 1. Clone the repository
git clone https://github.com/Tsion-Teklay/addis-school-recommender.git
cd addis-school-recommender

# 2. Create and activate a virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Generate synthetic data
python src/data_generator.py

# 5. Preprocess datasets
python src/preprocessing.py

# 6. Run the FastAPI server
uvicorn api.main:app --reload
Open http://127.0.0.1:8000/docs to explore and test the API.

📝 API Example
Request:
GET /recommend?parent_id=1&top_n=5

Sample Response:

```json
JSON
[
  {
    "school_id": 42,
    "school_name": "ABC Academy",
    "score": 0.95,
    "sub_city": "Bole",
    "type": "Private"
  },
  ...
]
```

💻 Usage in Python
```python
from src.hybrid import hybrid_recommend

# Get top 5 schools for parent with ID 1
recommendations = hybrid_recommend(parent_id=1, top_n=5)
print(recommendations)
```

🔧 Dependencies
Python 3.12+
pandas
numpy
scikit-learn
fastapi
uvicorn

📈 Optional Enhancements
- Add caching for repeated requests
- Sub-city filters for custom results
- Dockerize for containerized deployment
- Unit tests for all recommendation methods
- Support for real-world datasets

🤝 Contributing
Contributions are welcome! Feel free to open an issue, make a suggestion, or submit a PR.

📄 License
MIT License © 2026 Tsion-Teklay