# Addis Ababa School Recommender System

A **hybrid recommendation system** to help parents in **Addis Ababa** choose the best schools for their children based on:

- **Budget constraints**  
- **Sub-city proximity**  
- **School type & stream preference**  
- **School quality scores**  
- **Parent–school interaction patterns**

Built with **pure Python**, using a combination of **content-based** and **collaborative filtering** methods, and deployed with **FastAPI**.

---

## 📂 Project Structure
addis-school-recommender/
├─ api/                  # FastAPI application
│  └─ main.py
├─ data/
│  ├─ raw/               # Generated raw CSV datasets
│  └─ processed/         # Processed data files
├─ models/               # Saved preprocessors or model artifacts
├─ src/                  # Core Python modules
│  ├─ content_based.py
│  ├─ collaborative.py
│  ├─ hybrid.py
│  ├─ data_loader.py
│  ├─ preprocessing.py
│  └─ data_generator.py
├─ requirements.txt
└─ README.md


---

## ⚡ Features

1. **Data Generator**  
   Generates synthetic datasets:
   - ~300 schools across all sub-cities of Addis Ababa  
   - ~2000 parents with preferences  
   - ~10,000 parent–school interactions  

2. **Preprocessing Pipeline**  
   - One-hot encoding for categorical features  
   - Normalization for numeric features  
   - Saved preprocessor for reuse  

3. **Content-Based Recommender**  
   - Scores schools based on parent preferences  
   - Distance-based sub-city scoring  
   - Stream and type matching  
   - School quality and academic fit  

4. **Collaborative Filtering**  
   - User-based similarity (cosine similarity)  
   - Weighted scoring from similar parents’ ratings  
   - Fully Python + scikit-learn (no heavy dependencies)  

5. **Hybrid Recommender**  
   - Combines content + collaborative + quality scores  
   - Weighted final ranking  
   - Respects budget constraint  

6. **FastAPI Deployment**  
   - `/recommend?parent_id=<id>&top_n=<n>`  
   - Returns top-N school recommendations in JSON  
   - Auto-generated Swagger docs at `/docs`

---

## 🛠 Installation

1. **Clone the repository**

```bash
git clone https://github.com/<your-username>/addis-school-recommender.git
cd addis-school-recommender

2. **Create virtual environment**

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate # macOS/Linux

3. **Install dependencies**

```bash
pip install -r requirments.txt

4. **Generate dataset**

```bash
python src/data_generator.py

5. **Preprocess the dataset**

```bash
python src/preprocessing.py


## 🚀 Run the API

uvicorn api.main:app --reload

Visit in browser:

Swagger docs: http://127.0.0.1:8000/docs

Example endpoint: http://127.0.0.1:8000/recommend?parent_id=1&top_n=5


## ⚖️ Usage Example

from hybrid import hybrid_recommend

# Get top 5 schools for parent with ID 1
recommendations = hybrid_recommend(parent_id=1, top_n=5)
print(recommendations)


🔧 Dependencies

Python 3.12+

pandas, numpy, scikit-learn

fastapi, uvicorn


🏆 Key Highlights

Production-ready Python project

Modular, maintainable, and extendable code

Combines multiple recommendation strategies

Budget and proximity constraints applied

Fully deployable via FastAPI

📌 Optional Enhancements

Add caching for repeated requests

Add subcity filters for more customized results

Dockerize for deployment

Unit tests for all recommenders

Real dataset integration

📄 License

MIT License © 2026