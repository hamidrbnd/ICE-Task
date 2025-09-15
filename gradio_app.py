# gradio_app.py
import os, json
import numpy as np
import pandas as pd
import gradio as gr
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

USE_TORCH = True
try:
    import torch, torch.nn as nn
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except Exception:
    USE_TORCH = False
    device = "cpu"

DATA_DIR = "/content/drive/MyDrive/processed_ice"                  # where Books.csv, Users.csv, Ratings.csv live
OUT_DIR  = "/content/drive/MyDrive/processed_ice/processed"        # where we’ll save cleaned files
MIN_USER_INTERACTIONS = 5
MIN_ITEM_INTERACTIONS = 5
IMPLICIT_THRESH = 5
K_NEIGHBORS = 50

# load & prep (same as streamlit)
books = pd.read_csv(os.path.join(DATA_DIR, "Books.csv"), dtype=str, encoding="latin-1")
ratings = pd.read_csv(os.path.join(DATA_DIR, "Ratings.csv"), dtype=str, encoding="latin-1")
users = pd.read_csv(os.path.join(DATA_DIR, "Users.csv"), dtype=str, encoding="latin-1")
books.columns = [c.strip().replace(" ", "-") for c in books.columns]
ratings.columns = [c.strip().replace(" ", "-") for c in ratings.columns]
users.columns = [c.strip().replace(" ", "-") for c in users.columns]

def to_int_safe(x):
    try: return int(float(str(x).strip()))
    except: return np.nan

ratings["User-ID"] = ratings["User-ID"].apply(to_int_safe)
ratings["Book-Rating"] = ratings["Book-Rating"].apply(to_int_safe)
ratings = ratings.dropna(subset=["User-ID","ISBN","Book-Rating"])
ratings["User-ID"] = ratings["User-ID"].astype(int)
ratings["Book-Rating"] = ratings["Book-Rating"].astype(int)

users['User-ID'] = users['User-ID'].apply(to_int_safe)

df = ratings.merge(books[["ISBN"]], on="ISBN", how="inner").merge(users[["User-ID"]], on="User-ID", how="inner")
uc, ic = df["User-ID"].value_counts(), df["ISBN"].value_counts()
keep_u = set(uc[uc >= MIN_USER_INTERACTIONS].index)
keep_i = set(ic[ic >= MIN_ITEM_INTERACTIONS].index)
df = df[df["User-ID"].isin(keep_u) & df["ISBN"].isin(keep_i)].copy()

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

knn_items = NearestNeighbors(n_neighbors=K_NEIGHBORS, metric="cosine", algorithm="brute", n_jobs=-1).fit(ui_matrix.T)

def recommend_items_for_user(user_id, N=10, K=K_NEIGHBORS):
    if user_id not in uid2ix: return []
    uix = uid2ix[user_id]
    user_items = ui_matrix[uix].indices
    if len(user_items) == 0: return []
    dist, idx = knn_items.kneighbors(ui_matrix.T[user_items], n_neighbors=K, return_distance=True)
    sim = 1.0 - dist
    scores = np.zeros(n_items, dtype=np.float32)
    for nbrs, sims in zip(idx, sim):
        for j, s in zip(nbrs[1:], sims[1:]):
            scores[j] += s
    seen = set(user_items.tolist())
    if seen: scores[list(seen)] = -1e9
    order = np.argsort(-scores)[:N].tolist()
    return [ix2isbn[i] for i in order]

book_lookup = books.set_index("ISBN")[["Book-Title","Book-Author","Year-Of-Publication","Image-URL-S"]].to_dict(orient="index")

def to_table(isbns):
    rows = []
    for isbn in isbns:
        meta = book_lookup.get(isbn, {})
        rows.append([isbn, meta.get("Book-Title",""), meta.get("Book-Author",""), meta.get("Year-Of-Publication",""), meta.get("Image-URL-S","")])
    return pd.DataFrame(rows, columns=["ISBN","Title","Author","Year","Image"])

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

def load_hybrid():
    ckpt = os.path.join(OUT_DIR, "hybrid_model.pt")
    cfgp = os.path.join(OUT_DIR, "hybrid_config.json")
    up = os.path.join(OUT_DIR, "U_feat.npy")
    ip = os.path.join(OUT_DIR, "I_feat.npy")
    if not (USE_TORCH and os.path.exists(ckpt) and os.path.exists(cfgp)):
        return None, None, None
    with open(cfgp, "r") as f:
        cfg = json.load(f)
    model = HybridRec(cfg["n_users"], cfg["n_items"], cfg["d_id"], cfg["d_user_feat"], cfg["d_item_feat"], cfg["hidden"], cfg["dropout"]).to(device).eval()
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state)
    U_feat = torch.from_numpy(np.load(up)).to(device) if os.path.exists(up) else None
    I_feat = torch.from_numpy(np.load(ip)).to(device) if os.path.exists(ip) else None
    return model, U_feat, I_feat

hybrid_model, U_feat_t, I_feat_t = load_hybrid()

def rec_for_user(user_id, topN=10, K=50, use_hybrid=True):
    base = recommend_items_for_user(user_id, N=max(3*topN, 100), K=K)
    if use_hybrid and hybrid_model is not None and len(base)>0:
        uix = torch.tensor([uid2ix[user_id]], dtype=torch.long, device=device)
        iix = torch.tensor([isbn2ix[i] for i in base], dtype=torch.long, device=device)
        with torch.no_grad():
            logits = hybrid_model(uix, iix, U_feat=U_feat_t, I_feat=I_feat_t).float().cpu().numpy()
        order = np.argsort(-logits)[:topN].tolist()
        base = [base[i] for i in order]
    return to_table(base[:topN])

def similar_items(isbn, topN=10):
    if isbn not in isbn2ix:
        return pd.DataFrame(columns=["ISBN","Title","Author","Year","Image"])
    base_iix = isbn2ix[isbn]
    dist, idx = knn_items.kneighbors(ui_matrix.T[base_iix], n_neighbors=topN+1, return_distance=True)
    nbrs = idx.ravel().tolist()[1:]
    return to_table([ix2isbn[i] for i in nbrs][:topN])

users_list = sorted(uid2ix.keys())

with gr.Blocks(title="Book Recommender") as demo:
    gr.Markdown("## 📚 Personalized Book Recommender")
    with gr.Tab("🔮 For You (User-ID)"):
        user = gr.Dropdown(choices=users_list, label="User-ID", value=users_list[0] if users_list else None)
        topN = gr.Slider(5, 30, value=10, step=1, label="Top-N")
        K    = gr.Slider(10, 200, value=K_NEIGHBORS, step=10, label="Neighbors per item (K)")
        useH = gr.Checkbox(value=(hybrid_model is not None), label="Re-rank with Hybrid (if available)")
        btn1 = gr.Button("Recommend")
        out1 = gr.Dataframe(headers=["ISBN","Title","Author","Year","Image"], wrap=True)
        btn1.click(fn=rec_for_user, inputs=[user, topN, K, useH], outputs=out1)

    with gr.Tab("🧭 Similar Items (ISBN)"):
        isbn_in = gr.Textbox(label="ISBN")
        btn2 = gr.Button("Find Similar")
        out2 = gr.Dataframe(headers=["ISBN","Title","Author","Year","Image"], wrap=True)
        btn2.click(fn=similar_items, inputs=[isbn_in, topN], outputs=out2)

demo.launch(share=True)

