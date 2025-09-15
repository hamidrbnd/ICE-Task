# 📚 Book Recommendation System  

This repository implements a **scalable recommendation system** using both **Collaborative Filtering (CF)** and a **Hybrid Neural Model**.  
It also includes an interactive **dashboard** (Streamlit or Gradio) for showcasing recommendations.  

---

## 🚀 Project Overview  

- **Goal:** Deliver personalized book recommendations using multiple approaches.  
- **Business Value:** Improve user engagement, retention, and discovery of new books.  
- **Techniques Used:**  
  - Collaborative Filtering (user–user & item–item)  
  - Hybrid deep neural re-ranker (user embeddings + demographics, item embeddings + metadata)  
  - Streamlit/Gradio dashboards for visualization  

---

## 📂 Dataset  

We use the **Book-Crossing dataset** (3 CSV files provided):  

- **Users.csv** → User-ID, Age, Location  
- **Books.csv** → ISBN, Title, Author, Publisher, Year, Cover URL  
- **Ratings.csv** → User-ID, ISBN, Book-Rating (0–10)  

### Preprocessing  
- Filter out users/items with <5 interactions (for stable CF).  
- Remove “uniform raters” (e.g., always rating 10 or always 0).  
- Convert ratings into **implicit feedback**:  
  - `>=5` → positive interaction  
  - `<5`  → negative/no interaction  
- Build:  
  - `uid2ix`, `isbn2ix` mappings  
  - Sparse **user–item matrix** for CF  
  - Metadata features (TF-IDF + SVD embeddings of titles, author, publisher)  
  - Demographic features (age bucket, country one-hot)  

---

## 🧠 Model Architectures  

### 1. Collaborative Filtering (CF)  
- **User–User CF** → finds similar users based on rating patterns.  
- **Item–Item CF (KNN)** → finds items co-liked by many users.  
- Used for **fast candidate generation**.  

### 2. Hybrid Model  
- **Inputs:**  
  - User embeddings (learned from ID) + demographic features  
  - Item embeddings (learned from ID) + content features (title embeddings, author, etc.)  
  - Dual-Pool Retrieval: combines Item+User CF scores (weighted merge)

- **Architecture:**  
  - Embeddings + projected features combined  
  - Concatenated → MLP re-ranker → relevance score  
- **Training:**  
  - Implicit feedback with negative sampling  
  - BCEWithLogitsLoss  
  - Early stopping on validation loss  
- **Output:** Personalized ranking of candidate books  

---

## 📊 Evaluation  

We evaluated the models on **100 sample users** using Precision@10, Recall@10, and NDCG@10.  

| Model            | Precision@10 | Recall@10 | NDCG@10 |
|------------------|--------------|-----------|---------|
| Item–Item KNN    | 0.00152      | 0.0152    | 0.0093  |
| User–User KNN    | **0.00296**  | **0.0296**| **0.0189** |
| Hybrid           | 0.00244      | 0.0244    | 0.0142  |
| Dual-pool Hybrid | 0.00224      | 0.0224    | 0.0116  |

**Takeaways:**  
- User–User CF performed best.  
- Hybrid and Dual-pool were limited by sparse metadata (Age, Location; Title, Author, Publisher, Year).  
- Richer features are needed for Hybrid to surpass CF baselines.

---

## 🖥️ Dashboard  

We provide two interactive dashboards to explore recommendations:  

### 1. Streamlit (`streamlit_app.py`)  
- Input: select a **User-ID** to get recommendations  
- Input: enter an **ISBN** to find similar books  
- Output: table + cover images of recommended books  

Run:  
```bash
streamlit run streamlit_app.py
```

### 2. Gradio (`gradio_app.py`)  
- Same features, but launches in a browser tab  
- Works easily in **Colab** with `share=True`  

Run:  
```bash
python gradio_app.py
```

---

## ⚡ Scalability  

- Candidate retrieval (item–item KNN) is sparse & efficient  
- Re-ranking with a lightweight neural net is fast at inference  
- For production scale:  
  - Replace KNN with **FAISS** / **Annoy** for million-item catalogs  
  - Serve model as a **microservice API**  

---

## 📦 Project Structure  

```
.
├── Books.csv
├── Users.csv
├── Ratings.csv
├── processed/              # saved models & features
│   ├── hybrid_model.pt
│   ├── hybrid_config.json
│   ├── U_feat.npy
│   └── I_feat.npy
├── train_eval.ipynb        # training + evaluation notebook
├── streamlit_app.py        # Streamlit dashboard
├── gradio_app.py           # Gradio dashboard
└── README.md
```

---

## 🔑 Key Takeaways  

- **Collaborative filtering** is effective when user–item interactions are dense.  
- **Hybrid models** handle cold-start by leveraging user demographics and book metadata.  
- **Dashboard** demonstrates recommendations interactively to non-technical teams.  
- The system is **modular and scalable**: retrieval + re-ranker → can be deployed in production.  

---

## ✅ Next Steps  

- Add richer content features (book descriptions, embeddings, cover images).  
- Deploy dashboard on **Streamlit Cloud** or **HuggingFace Spaces**.  
- Extend evaluation with online metrics (CTR, conversions).  

---

🙌 **Done by:** *[Your Name]*  
