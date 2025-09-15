# streamlit_app.py
import os, json
import numpy as np
import pandas as pd
import streamlit as st
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

# Optional PyTorch hybrid re-ranker
USE_TORCH = True
try:
    import torch
    import torch.nn as nn
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except Exception:
    USE_TORCH = False
    device = "cpu"

# -------------------------------
# Config
# -------------------------------
DATA_DIR = "/content/drive/MyDrive/processed_ice"                  # where Books.csv, Users.csv, Ratings.csv live
OUT_DIR  = "/content/drive/MyDrive/processed_ice/processed"        # where we’ll save cleaned files
MIN_USER_INTERACTIONS = 5
MIN_ITEM_INTERACTIONS = 5
IMPLICIT_THRESH = 5
K_ITEM_DEFAULT  = 50
K_USER_DEFAULT  = 100
TOPN_DEFAULT    = 10

# -------------------------------
# Load data
# -------------------------------
@st.cache_data(show_spinner=True)
def load_raw():
    books   = pd.read_csv(os.path.join(DATA_DIR, "Books.csv"), dtype=str, encoding="latin-1")
    ratings = pd.read_csv(os.path.join(DATA_DIR, "Ratings.csv"), dtype=str, encoding="latin-1")
    users   = pd.read_csv(os.path.join(DATA_DIR, "Users.csv"), dtype=str, encoding="latin-1")

    books.columns   = [c.strip().replace(" ", "-") for c in books.columns]
    ratings.columns = [c.strip().replace(" ", "-") for c in ratings.columns]
    users.columns   = [c.strip().replace(" ", "-") for c in users.columns]

    def to_int_safe(x):
        try: return int(float(str(x).strip()))
        except: return np.nan

    ratings["User-ID"] = ratings["User-ID"].apply(to_int_safe)
    ratings["Book-Rating"] = ratings["Book-Rating"].apply(to_int_safe)
    ratings = ratings.dropna(subset=["User-ID","ISBN","Book-Rating"])
    ratings["User-ID"] = ratings["User-ID"].astype(int)
    ratings["Book-Rating"] = ratings["Book-Rating"].astype(int)
    users['User-ID'] = users['User-ID'].apply(to_int_safe)

    # keep consistent ids + activity filtering
    df = ratings.merge(books[["ISBN"]], on="ISBN", how="inner")
    df = df.merge(users[["User-ID"]], on="User-ID", how="inner")
    uc = df["User-ID"].value_counts()
    ic = df["ISBN"].value_counts()
    keep_u = set(uc[uc >= MIN_USER_INTERACTIONS].index)
    keep_i = set(ic[ic >= MIN_ITEM_INTERACTIONS].index)
    df = df[df["User-ID"].isin(keep_u) & df["ISBN"].isin(keep_i)].copy()

    return books, users, df

books, users, df = load_raw()

# -------------------------------
# Mappings & implicit UI matrix
# -------------------------------
uid2ix = {u:i for i,u in enumerate(sorted(df["User-ID"].unique()))}
ix2uid = {i:u for u,i in uid2ix.items()}
isbn2ix = {b:i for i,b in enumerate(sorted(df["ISBN"].unique()))}
ix2isbn = {i:b for b,i in isbn2ix.items()}

n_users, n_items = len(uid2ix), len(isbn2ix)

implicit = df.copy()
implicit["y"] = (implicit["Book-Rating"] >= IMPLICIT_THRESH).astype(int)
rows = implicit["User-ID"].map(uid2ix).to_numpy()
cols = implicit["ISBN"].map(isbn2ix).to_numpy()
vals = implicit["y"].to_numpy()
ui_matrix = csr_matrix((vals, (rows, cols)), shape=(n_users, n_items))

# -------------------------------
# Item–Item KNN (retrieval)
# -------------------------------
@st.cache_resource(show_spinner=True)
def build_item_knn(ui_mat, k_neighbors=K_ITEM_DEFAULT):
    return NearestNeighbors(
        n_neighbors=k_neighbors, metric="cosine", algorithm="brute", n_jobs=-1
    ).fit(ui_mat.T)

knn_items = build_item_knn(ui_matrix, K_ITEM_DEFAULT)

def recommend_items_for_user(user_id, N=TOPN_DEFAULT, K=K_ITEM_DEFAULT):
    if user_id not in uid2ix: return []
    uix = uid2ix[user_id]
    user_items = ui_matrix[uix].indices
    if len(user_items) == 0: return []
    dist, idx = knn_items.kneighbors(ui_matrix.T[user_items], n_neighbors=K, return_distance=True)
    sim = 1.0 - dist
    scores = np.zeros(n_items, dtype=np.float32)
    for nbrs, sims in zip(idx, sim):
        for j, s in zip(nbrs[1:], sims[1:]):  # skip self
            scores[j] += s
    seen = set(user_items.tolist())
    if seen: scores[list(seen)] = -1e9
    order = np.argsort(-scores)[:N].tolist()
    return [ix2isbn[i] for i in order]

# -------------------------------
# User–User KNN (for Dual-Pool)
# -------------------------------
@st.cache_resource(show_spinner=True)
def build_user_knn(ui_mat, k_neighbors=K_USER_DEFAULT):
    return NearestNeighbors(
        n_neighbors=k_neighbors, metric="cosine", algorithm="brute", n_jobs=-1
    ).fit(ui_mat)

knn_users = build_user_knn(ui_matrix, K_USER_DEFAULT)

def item_knn_scores_for_user(uix: int, K: int = K_ITEM_DEFAULT) -> np.ndarray:
    scores = np.zeros(n_items, dtype=np.float32)
    user_items = ui_matrix[uix].indices
    if len(user_items) == 0: return scores
    dist, idx = knn_items.kneighbors(ui_matrix.T[user_items], n_neighbors=K, return_distance=True)
    sim = 1.0 - dist
    for nbrs, sims in zip(idx, sim):
        for j, s in zip(nbrs[1:], sims[1:]):
            scores[j] += s
    return scores

def user_knn_scores_for_user(uix: int, K: int = K_USER_DEFAULT) -> np.ndarray:
    dist, idx = knn_users.kneighbors(ui_matrix[uix], n_neighbors=K, return_distance=True)
    sims = 1.0 - dist.ravel()
    nbrs = idx.ravel()
    scores = np.zeros(n_items, dtype=np.float32)
    for s, v in zip(sims[1:], nbrs[1:]):  # skip self
        if s <= 0: continue
        v_items = ui_matrix[v].indices
        scores[v_items] += s
    return scores

def dual_pool_candidates(user_id, pool_item=200, pool_user=200, K_item=K_ITEM_DEFAULT, K_user=K_USER_DEFAULT, w_item=1.0, w_user=1.0):
    if user_id not in uid2ix: return []
    uix = uid2ix[user_id]
    s_item = item_knn_scores_for_user(uix, K=K_item)
    s_user = user_knn_scores_for_user(uix, K=K_user)

    seen = set(ui_matrix[uix].indices.tolist())
    if seen:
        s_item[list(seen)] = -1e9
        s_user[list(seen)] = -1e9

    cand_item = np.argsort(-s_item)[:pool_item]
    cand_user = np.argsort(-s_user)[:pool_user]

    merged = {}
    for i in cand_item:
        if s_item[i] > -1e8:
            merged[i] = merged.get(i, 0.0) + w_item * float(s_item[i])
    for i in cand_user:
        if s_user[i] > -1e8:
            merged[i] = merged.get(i, 0.0) + w_user * float(s_user[i])

    if not merged: return []
    ranked_iix = sorted(merged.keys(), key=lambda j: merged[j], reverse=True)
    return [ix2isbn[i] for i in ranked_iix]

# -------------------------------
# Optional Hybrid re-ranker
# -------------------------------
class HybridRec(nn.Module):
    def __init__(self, n_users, n_items, d_id=64, d_user_feat=0, d_item_feat=0, hidden=128, dropout=0.1):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, d_id)
        self.item_emb = nn.Embedding(n_items, d_id)
        self.use_user_feat = d_user_feat > 0
        self.use_item_feat = d_item_feat > 0
        if self.use_user_feat: self.user_feat_proj = nn.Linear(d_user_feat, d_id)
        if self.use_item_feat: self.item_feat_proj = nn.Linear(d_item_feat, d_id)
        self.mlp = nn.Sequential(
            nn.Linear(d_id*2, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden//2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden//2, 1)
        )
    def forward(self, uix, iix, U_feat=None, I_feat=None):
        if iix.ndim == 1:
            uix = uix.repeat(iix.shape[0])
            u = self.user_emb(uix); i = self.item_emb(iix)
            if self.use_user_feat and U_feat is not None: u = u + self.user_feat_proj(U_feat[uix])
            if self.use_item_feat and I_feat is not None: i = i + self.item_feat_proj(I_feat[iix])
            x = torch.cat([u, i], dim=-1)
            return self.mlp(x).squeeze(-1)
        B, M = iix.shape
        u = self.user_emb(uix); i = self.item_emb(iix)
        if self.use_user_feat and U_feat is not None: u = u + self.user_feat_proj(U_feat[uix])
        if self.use_item_feat and I_feat is not None: i = i + self.item_feat_proj(I_feat[iix])
        u = u.unsqueeze(1).expand(-1, M, -1)
        x = torch.cat([u, i], dim=-1)
        return self.mlp(x).squeeze(-1)

@st.cache_resource(show_spinner=True)
def maybe_load_hybrid():
    ckpt = os.path.join(OUT_DIR, "hybrid_model.pt")
    cfgp = os.path.join(OUT_DIR, "hybrid_config.json")
    up   = os.path.join(OUT_DIR, "U_feat.npy")
    ip   = os.path.join(OUT_DIR, "I_feat.npy")
    if not (USE_TORCH and os.path.exists(ckpt) and os.path.exists(cfgp)):
        return None, None, None
    with open(cfgp, "r") as f:
        cfg = json.load(f)
    model = HybridRec(
        n_users=cfg["n_users"], n_items=cfg["n_items"],
        d_id=cfg["d_id"], d_user_feat=cfg["d_user_feat"], d_item_feat=cfg["d_item_feat"],
        hidden=cfg["hidden"], dropout=cfg["dropout"]
    ).to(device).eval()
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state)
    U_feat = torch.from_numpy(np.load(up)).to(device) if os.path.exists(up) else None
    I_feat = torch.from_numpy(np.load(ip)).to(device) if os.path.exists(ip) else None
    return model, U_feat, I_feat

hybrid_model, U_feat_t, I_feat_t = maybe_load_hybrid()

def rerank_with_hybrid(user_id, base_isbns, topN=TOPN_DEFAULT):
    if hybrid_model is None or not base_isbns:
        return base_isbns[:topN]
    uix = torch.tensor([uid2ix[user_id]], dtype=torch.long, device=device)
    iix = torch.tensor([isbn2ix[i] for i in base_isbns], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = hybrid_model(uix, iix, U_feat=U_feat_t, I_feat=I_feat_t).float().cpu().numpy()
    order = np.argsort(-logits)[:topN].tolist()
    return [base_isbns[i] for i in order]

# -------------------------------
# Metadata lookup
# -------------------------------
book_lookup = books.set_index("ISBN")[["Book-Title","Book-Author","Year-Of-Publication","Image-URL-S"]].to_dict(orient="index")

def to_table(isbns):
    rows = []
    for isbn in isbns:
        meta = book_lookup.get(isbn, {})
        rows.append({
            "ISBN": isbn,
            "Title": meta.get("Book-Title",""),
            "Author": meta.get("Book-Author",""),
            "Year": meta.get("Year-Of-Publication",""),
            "Image": meta.get("Image-URL-S",""),
        })
    return pd.DataFrame(rows)

# -------------------------------
# UI
# -------------------------------
st.set_page_config(page_title="Book Recommender", page_icon="📚", layout="wide")
st.title("📚 Personalized Book Recommender")

tab1, tab2 = st.tabs(["🔮 For You (User-ID)", "🧭 Similar Items (ISBN)"])

with tab1:
    st.subheader("Get personalized recommendations")
    colA, colB = st.columns([2,1])
    with colA:
        all_user_ids = sorted(uid2ix.keys())
        user_id = st.selectbox("User-ID", all_user_ids, index=0)
    with colB:
        topN = st.slider("Top-N", 5, 30, TOPN_DEFAULT)
        K_item = st.slider("Neighbors per item (K_item)", 10, 200, K_ITEM_DEFAULT, step=10)
        use_hybrid = st.checkbox("Re-rank with Hybrid (if available)", value=(hybrid_model is not None))
        use_dualpool = st.checkbox("Use Dual-Pool Retrieval (Item+User CF)", value=False)
        w_user = st.slider("Dual-Pool weight: User CF", 0.0, 3.0, 1.0, 0.1)
        w_item = st.slider("Dual-Pool weight: Item CF", 0.0, 3.0, 1.0, 0.1)

    if st.button("Recommend"):
        if use_dualpool:
            base = dual_pool_candidates(user_id, pool_item=max(3*topN,150), pool_user=max(3*topN,150),
                                        K_item=K_item, K_user=K_USER_DEFAULT,
                                        w_item=w_item, w_user=w_user)
        else:
            base = recommend_items_for_user(user_id, N=max(3*topN,100), K=K_item)
        final = rerank_with_hybrid(user_id, base, topN) if use_hybrid else base[:topN]
        df_show = to_table(final)
        st.dataframe(df_show, use_container_width=True)
        if df_show["Image"].notna().any():
            st.markdown("---")
            for _, r in df_show.iterrows():
                c1, c2 = st.columns([1,6])
                with c1:
                    if isinstance(r["Image"], str) and r["Image"].startswith("http"):
                        st.image(r["Image"], width=90)
                with c2:
                    st.markdown(f"**{r['Title']}**")
                    st.caption(f"{r['Author']} • {r['Year']} • ISBN: {r['ISBN']}")

with tab2:
    st.subheader("Find similar books by ISBN")
    isbn_input = st.text_input("Enter an ISBN from the dataset")
    if st.button("Find Similar"):
        if isbn_input in isbn2ix:
            base_iix = isbn2ix[isbn_input]
            dist, idx = knn_items.kneighbors(ui_matrix.T[base_iix], n_neighbors=TOPN_DEFAULT+1, return_distance=True)
            nbrs = idx.ravel().tolist()[1:]
            isbns = [ix2isbn[i] for i in nbrs][:TOPN_DEFAULT]
            st.dataframe(to_table(isbns), use_container_width=True)
        else:
            st.warning("ISBN not found after filtering. Try another one.")
