"""
interface.py — Système de Recommandation de Films (MovieLens 100K)
Couvre les points 1 à 7 du sujet.
Lancer : streamlit run interface.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="MovieLens — Recommandation",
    page_icon="M",
    layout="wide",
    initial_sidebar_state="expanded"
)

C_RED   = '#e50914'
C_BLUE  = '#4466ff'
C_GREEN = '#22cc88'
C_BG    = '#080810'
C_CARD  = '#0d0d1a'
C_BORDER= '#1a1a30'

plt.rcParams.update({
    'figure.facecolor': '#0d0d18',
    'axes.facecolor':   '#13131f',
    'text.color':       '#e0e0f0',
    'axes.labelcolor':  '#e0e0f0',
    'xtick.color':      '#7070a0',
    'ytick.color':      '#7070a0',
    'axes.edgecolor':   '#2a2a45',
    'grid.color':       '#1a1a30',
    'axes.grid':        True,
    'grid.alpha':       0.3,
})

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

html, body, [data-testid="stAppViewContainer"], section.main {
    background: #080810 !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    color: #c0c0d8 !important;
}
[data-testid="stHeader"]     { background: transparent !important; }
[data-testid="stSidebar"]    { background: #0a0a14 !important; border-right: 1px solid #1a1a30; }
[data-testid="stDecoration"] { display: none !important; }
#MainMenu, footer            { visibility: hidden; }
.main .block-container       { padding: 2rem 2.5rem !important; max-width: 1400px; }

[data-testid="stSidebar"] label {
    color: #3a3a5a !important;
    font-size: 0.65rem !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    font-family: 'IBM Plex Mono', monospace !important;
}

.logo {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.1rem;
    font-weight: 600;
    color: #e50914;
    letter-spacing: 4px;
    text-transform: uppercase;
    padding: 0.2rem 0 1.5rem 0;
    border-bottom: 1px solid #1a1a30;
    margin-bottom: 1.5rem;
}
.logo-sub {
    font-size: 0.55rem;
    color: #222240;
    letter-spacing: 2px;
    margin-top: 0.2rem;
}

.ph { margin-bottom: 2rem; padding-bottom: 1.2rem; border-bottom: 1px solid #1a1a30; }
.ph-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.1rem;
    font-weight: 600;
    color: #e8e8f8;
    letter-spacing: 2px;
    text-transform: uppercase;
}
.ph-sub { font-size: 0.7rem; color: #333355; letter-spacing: 1.5px; margin-top: 0.3rem; }

.mrow { display: flex; gap: 0.8rem; margin-bottom: 1.8rem; flex-wrap: wrap; }
.mcard {
    flex: 1; min-width: 120px;
    background: #0d0d1a; border: 1px solid #1a1a30;
    border-radius: 6px; padding: 1rem 1.2rem;
}
.mcard:hover { border-color: #e50914; }
.mval { font-family: 'IBM Plex Mono', monospace; font-size: 1.6rem; color: #f0f0ff; font-weight: 600; }
.mlbl { font-size: 0.58rem; color: #2a2a45; text-transform: uppercase; letter-spacing: 2px; margin-top: 0.2rem; }

.stitle {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    font-weight: 600;
    color: #3a3a5a;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin: 1.5rem 0 0.8rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #0f0f20;
}

.info-box {
    background: rgba(68,102,255,0.06);
    border-left: 2px solid #4466ff;
    border-radius: 4px;
    padding: 0.7rem 1rem;
    margin: 0.8rem 0 1.2rem 0;
    font-size: 0.8rem;
    color: #6070b0;
    font-family: 'IBM Plex Mono', monospace;
}
.ok-box {
    background: rgba(34,204,136,0.06);
    border-left: 2px solid #22cc88;
    border-radius: 4px;
    padding: 0.7rem 1rem;
    margin: 0.8rem 0;
    font-size: 0.8rem;
    color: #40a080;
    font-family: 'IBM Plex Mono', monospace;
}

.rtable { width: 100%; border-collapse: collapse; }
.rtable th {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.55rem; letter-spacing: 2.5px;
    text-transform: uppercase; color: #2a2a45;
    padding: 0.5rem 0.8rem; border-bottom: 1px solid #1a1a30; text-align: left;
}
.rtable td { padding: 0.65rem 0.8rem; border-bottom: 1px solid #0d0d1e; font-size: 0.82rem; color: #b0b0cc; }
.rtable tr:hover td { background: #0d0d1e; }
.rbadge {
    display: inline-flex; align-items: center; justify-content: center;
    width: 20px; height: 20px; border-radius: 3px;
    background: #e50914; color: #fff;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.6rem; font-weight: 600;
}
.rbadge.b { background: #4466ff; }
.rbadge.g { background: #22cc88; color: #000; }
.film-title { color: #e8e8f8; font-weight: 500; }
.gp {
    display: inline-block; padding: 0.1rem 0.45rem;
    background: rgba(80,80,160,0.1); border: 1px solid #1e1e38;
    border-radius: 3px; font-size: 0.6rem; color: #5050a0; margin: 0.1rem;
    font-family: 'IBM Plex Mono', monospace;
}
.sbar-wrap { background: #1a1a30; border-radius: 2px; height: 3px; width: 70px; display: inline-block; vertical-align: middle; }
.sbar      { height: 3px; border-radius: 2px; background: #e50914; }
.sbar.b    { background: #4466ff; }
.sbar.g    { background: #22cc88; }

.ecard {
    flex: 1; min-width: 150px;
    background: #0d0d1a; border: 1px solid #1a1a30;
    border-radius: 6px; padding: 1rem 1.2rem;
}
.ecard.best { border-color: #22cc88; }
.emethod { font-family: 'IBM Plex Mono', monospace; font-size: 0.55rem; letter-spacing: 2px; text-transform: uppercase; color: #3a3a5a; }
.ermse { font-family: 'IBM Plex Mono', monospace; font-size: 1.4rem; color: #f0f0ff; margin: 0.4rem 0 0.2rem 0; }
.eprec { font-size: 0.7rem; color: #5050a0; }
.ebest-tag { font-size: 0.6rem; color: #22cc88; margin-top: 0.3rem; font-family: 'IBM Plex Mono', monospace; }

.c7-box {
    background: #0d0d1a;
    border: 1px solid #1a1a30;
    border-radius: 6px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}
.c7-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    color: #e50914;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-bottom: 1rem;
    border-bottom: 1px solid #1a1a30;
    padding-bottom: 0.6rem;
}
.c7-check { font-size: 0.82rem; color: #b0b0cc; padding: 0.3rem 0; }
.c7-check strong { color: #e8e8f8; font-family: 'IBM Plex Mono', monospace; }

::-webkit-scrollbar { width: 3px; }
::-webkit-scrollbar-thumb { background: #e50914; border-radius: 2px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────
GENRE_COLS = ['Action','Adventure','Animation',"Children's",'Comedy','Crime',
              'Documentary','Drama','Fantasy','Film-Noir','Horror','Musical',
              'Mystery','Romance','Sci-Fi','Thriller','War','Western']

@st.cache_data
def load_data():
    ratings = pd.read_csv('data/u.data', sep='\t',
        names=['user_id','item_id','rating','timestamp'])
    movies = pd.read_csv('data/u.item', sep='|', encoding='latin-1',
        names=['item_id','title','release_date','video_date','imdb_url','unknown']+GENRE_COLS,
        usecols=range(24))
    movies['year'] = pd.to_datetime(movies['release_date'], errors='coerce').dt.year
    movies['genres'] = movies[GENRE_COLS].apply(
        lambda r: [g for g in GENRE_COLS if r[g]==1], axis=1)
    movies['genres_str'] = movies['genres'].apply(lambda g: ', '.join(g) if g else 'Unknown')
    return ratings, movies

@st.cache_data
def build_matrices(_ratings):
    mat    = _ratings.pivot_table(index='user_id', columns='item_id', values='rating')
    filled = mat.fillna(0)
    u_sim  = pd.DataFrame(cosine_similarity(filled),   index=mat.index,    columns=mat.index)
    i_sim  = pd.DataFrame(cosine_similarity(filled.T), index=mat.columns,  columns=mat.columns)
    return mat, filled, u_sim, i_sim

@st.cache_data
def build_genre_mat(_movies):
    return _movies.set_index('item_id')[GENRE_COLS].fillna(0)

try:
    ratings, movies = load_data()
    umx, mfill, usim, isim = build_matrices(ratings)
    gmat = build_genre_mat(movies)
    DATA_OK = True
except Exception as e:
    DATA_OK = False; ERR = str(e)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def ub_reco(uid, n=5, ns=20):
    seen = umx.loc[uid].dropna().index.tolist()
    nb   = usim[uid].drop(uid).sort_values(ascending=False).head(ns)
    sc   = {}
    for item in [c for c in umx.columns if c not in seen]:
        col = umx[item]; v = nb[nb.index.isin(col.dropna().index)]
        if v.empty or v.sum()==0: continue
        sc[item] = np.dot(v.values, col[v.index].values) / v.sum()
    top = pd.Series(sc).sort_values(ascending=False).head(n)
    res = movies[movies.item_id.isin(top.index)][['item_id','title','year','genres_str']].copy()
    res['score'] = res.item_id.map(top)
    return res.sort_values('score', ascending=False).reset_index(drop=True)

def ib_reco(uid, n=5, ns=10):
    ur   = umx.loc[uid].dropna(); seen = ur.index.tolist()
    sc   = {}
    for item in [c for c in umx.columns if c not in seen]:
        if item not in isim.index: continue
        sims = isim[item][seen].sort_values(ascending=False).head(ns)
        if sims.sum()==0: continue
        sc[item] = np.dot(sims.values, ur[sims.index].values) / sims.sum()
    top = pd.Series(sc).sort_values(ascending=False).head(n)
    res = movies[movies.item_id.isin(top.index)][['item_id','title','year','genres_str']].copy()
    res['score'] = res.item_id.map(top)
    return res.sort_values('score', ascending=False).reset_index(drop=True)

def cb_reco(uid, n=5):
    ur    = umx.loc[uid].dropna()
    liked = ur[ur>=4].index.tolist() or ur.index.tolist()
    lin   = [i for i in liked if i in gmat.index]
    if not lin: return pd.DataFrame()
    prof  = gmat.loc[lin].mean().values.reshape(1,-1)
    sims  = pd.Series(cosine_similarity(prof, gmat)[0], index=gmat.index)
    sims  = sims.drop([i for i in ur.index if i in sims.index])
    top   = sims.sort_values(ascending=False).head(n)
    res   = movies[movies.item_id.isin(top.index)][['item_id','title','year','genres_str']].copy()
    res['score'] = res.item_id.map(top)
    return res.sort_values('score', ascending=False).reset_index(drop=True)

def rtable(df, score_col='score', badge='r', max_s=5.0):
    bc = {'r':'','b':'b','g':'g'}[badge]
    sc = {'r':'','b':'b','g':'g'}[badge]
    rows = ""
    for i, row in df.iterrows():
        pct  = min(row[score_col]/max_s, 1.0)*100
        gens = ''.join([f'<span class="gp">{g}</span>' for g in row['genres_str'].split(', ')[:3]])
        yr   = int(row['year']) if pd.notna(row.get('year')) else ''
        rows += f"""<tr>
          <td><span class="rbadge {bc}">{i+1}</span></td>
          <td><span class="film-title">{row['title']}</span>
              <span style="color:#2a2a45;font-size:0.65rem;margin-left:0.4rem">{yr}</span></td>
          <td>{gens}</td>
          <td><span style="color:#e0e0f0;font-family:'IBM Plex Mono',monospace">{row[score_col]:.3f}</span>
              <br><div class="sbar-wrap"><div class="sbar {sc}" style="width:{pct:.0f}%"></div></div></td>
        </tr>"""
    return f"""<table class="rtable"><thead><tr>
        <th>#</th><th>Film</th><th>Genres</th><th>Score</th>
    </tr></thead><tbody>{rows}</tbody></table>"""


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="logo">MOVIELENS
      <div class="logo-sub">RECOMMENDATION SYSTEM</div>
    </div>""", unsafe_allow_html=True)

    page = st.radio("Navigation", [
        "1. Import Dataset",
        "2. Matrice User-Movie",
        "3a. User-Based Filtering",
        "3b. Item-Based Filtering",
        "4. Content-Based",
        "5. Top-5 Utilisateur",
        "6. Evaluation",
        "7. Contraintes Techniques",
    ], label_visibility="collapsed")

    if DATA_OK:
        st.markdown(f"""
        <div style="margin-top:2rem">
        <div style="font-family:'IBM Plex Mono',monospace;font-size:0.55rem;color:#1e1e38;letter-spacing:2px;text-transform:uppercase;margin-bottom:0.6rem">Dataset</div>
        <div style="font-size:0.75rem;color:#3a3a5a;font-family:'IBM Plex Mono',monospace">{len(ratings):,} evaluations</div>
        <div style="font-size:0.75rem;color:#3a3a5a;font-family:'IBM Plex Mono',monospace">{ratings.user_id.nunique()} utilisateurs</div>
        <div style="font-size:0.75rem;color:#3a3a5a;font-family:'IBM Plex Mono',monospace">{ratings.item_id.nunique()} films</div>
        </div>""", unsafe_allow_html=True)

if not DATA_OK:
    st.error(f"Erreur chargement : {ERR}")
    st.info("Verifiez que data/u.data et data/u.item existent.")
    st.stop()


# ─────────────────────────────────────────────
# PAGE 1 — DATASET
# ─────────────────────────────────────────────
if page == "1. Import Dataset":
    st.markdown("""<div class="ph">
        <div class="ph-title">Import du Dataset</div>
        <div class="ph-sub">ETAPE 1 — MovieLens 100K — Chargement local depuis data/</div>
    </div>""", unsafe_allow_html=True)

    avg = ratings['rating'].mean()
    st.markdown(f"""<div class="mrow">
      <div class="mcard"><div class="mval">{len(ratings):,}</div><div class="mlbl">Evaluations</div></div>
      <div class="mcard"><div class="mval">{ratings.user_id.nunique()}</div><div class="mlbl">Utilisateurs</div></div>
      <div class="mcard"><div class="mval">{ratings.item_id.nunique()}</div><div class="mlbl">Films</div></div>
      <div class="mcard"><div class="mval">{avg:.2f}</div><div class="mlbl">Note moyenne / 5</div></div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="stitle">Apercu des films</div>', unsafe_allow_html=True)
    st.dataframe(
        movies[['item_id','title','year','genres_str']].head(10)
            .rename(columns={'item_id':'ID','title':'Titre','year':'Annee','genres_str':'Genres'}),
        use_container_width=True, hide_index=True
    )

    st.markdown('<div class="stitle">Visualisations</div>', unsafe_allow_html=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.patch.set_facecolor('#0d0d18')

    nc = ratings['rating'].value_counts().sort_index()
    axes[0].bar(nc.index, nc.values, color=C_RED, alpha=0.85, width=0.6)
    axes[0].set_title('Distribution des notes', color='#e0e0f0', pad=10)
    axes[0].set_xlabel('Note'); axes[0].set_ylabel('Evaluations')
    axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'{int(x):,}'))

    gc = pd.Series({g: movies[g].sum() for g in GENRE_COLS}).sort_values().tail(10)
    axes[1].barh(gc.index, gc.values, color=C_BLUE, alpha=0.8)
    axes[1].set_title('Top 10 Genres', color='#e0e0f0', pad=10)

    axes[2].hist(ratings.groupby('user_id').size(), bins=30, color=C_GREEN, alpha=0.8)
    axes[2].set_title('Evaluations / utilisateur', color='#e0e0f0', pad=10)

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)


# ─────────────────────────────────────────────
# PAGE 2 — MATRICE
# ─────────────────────────────────────────────
elif page == "2. Matrice User-Movie":
    st.markdown("""<div class="ph">
        <div class="ph-title">Matrice User-Movie</div>
        <div class="ph-sub">ETAPE 2 — pivot_table(user_id x item_id, values=rating)</div>
    </div>""", unsafe_allow_html=True)

    sparsity = 1 - (len(ratings) / (umx.shape[0] * umx.shape[1]))
    st.markdown(f"""<div class="mrow">
      <div class="mcard"><div class="mval">{umx.shape[0]}</div><div class="mlbl">Lignes (users)</div></div>
      <div class="mcard"><div class="mval">{umx.shape[1]}</div><div class="mlbl">Colonnes (films)</div></div>
      <div class="mcard"><div class="mval">{sparsity:.1%}</div><div class="mlbl">Sparsite</div></div>
      <div class="mcard"><div class="mval">{(1-sparsity):.1%}</div><div class="mlbl">Densite</div></div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="info-box">Les cellules NaN indiquent qu\'un utilisateur n\'a pas note ce film. Elles sont remplacees par 0 pour le calcul de similarite cosinus.</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="stitle">Extrait de la matrice (10x10)</div>', unsafe_allow_html=True)
        st.dataframe(umx.iloc[:10,:10].round(1), use_container_width=True)
    with col2:
        size = st.slider("Taille heatmap (N x N)", 10, 40, 20)
        fig, ax = plt.subplots(figsize=(6,5))
        fig.patch.set_facecolor('#0d0d18')
        sns.heatmap(umx.iloc[:size,:size].fillna(0), cmap='Reds',
                    linewidths=0.15, linecolor='#050510',
                    ax=ax, vmin=0, vmax=5)
        ax.set_title(f'Matrice {size}x{size}', color='#e0e0f0')
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)


# ─────────────────────────────────────────────
# PAGE 3a — USER-BASED
# ─────────────────────────────────────────────
elif page == "3a. User-Based Filtering":
    st.markdown("""<div class="ph">
        <div class="ph-title">User-Based Collaborative Filtering</div>
        <div class="ph-sub">ETAPE 3a — Cosine Similarity entre utilisateurs</div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="info-box">Formule : sim(u,v) = cos(r_u, r_v) = (r_u . r_v) / (||r_u|| x ||r_v||)\nPrediction : note(u,i) = somme(sim(u,v) x note(v,i)) / somme(sim(u,v))</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])
    with col1:
        uid  = st.selectbox("Utilisateur", sorted(umx.index.tolist()), index=0)
        nsim = st.slider("Nb voisins", 5, 50, 20)
        nrec = st.slider("Nb recommandations", 3, 10, 5)
        hist = ratings[ratings.user_id==uid]
        st.markdown(f"""<div class="mrow" style="flex-direction:column;gap:0.5rem">
          <div class="mcard"><div class="mval">{len(hist)}</div><div class="mlbl">Films notes</div></div>
          <div class="mcard"><div class="mval">{hist.rating.mean():.2f}</div><div class="mlbl">Note moyenne</div></div>
        </div>""", unsafe_allow_html=True)
    with col2:
        with st.spinner("Calcul..."):
            reco = ub_reco(uid, n=nrec, ns=nsim)
        st.markdown('<div class="stitle">Top-N recommandations</div>', unsafe_allow_html=True)
        if not reco.empty:
            st.markdown(rtable(reco, badge='r'), unsafe_allow_html=True)

    st.markdown('<div class="stitle">Heatmap similarite cosinus (15 premiers utilisateurs)</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(8,4))
    fig.patch.set_facecolor('#0d0d18')
    sns.heatmap(usim.iloc[:15,:15], cmap='Blues', ax=ax,
                linewidths=0.15, linecolor='#050510', vmin=0, vmax=1)
    ax.set_title('Cosine Similarity — Utilisateurs', color='#e0e0f0')
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)


# ─────────────────────────────────────────────
# PAGE 3b — ITEM-BASED
# ─────────────────────────────────────────────
elif page == "3b. Item-Based Filtering":
    st.markdown("""<div class="ph">
        <div class="ph-title">Item-Based Collaborative Filtering</div>
        <div class="ph-sub">ETAPE 3b — Cosine Similarity entre films</div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="info-box">Formule : sim(i,j) = cos(r_i, r_j)\nOn transpose la matrice (films x users) pour calculer la similarite entre items.</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="stitle">Recommandations pour un utilisateur</div>', unsafe_allow_html=True)
        uid2 = st.selectbox("Utilisateur", sorted(umx.index.tolist()), index=0, key='ib_u')
        nr2  = st.slider("Nb recommandations", 3, 10, 5, key='ib_n')
        with st.spinner("Calcul..."):
            reco2 = ib_reco(uid2, n=nr2)
        if not reco2.empty:
            st.markdown(rtable(reco2, badge='b'), unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="stitle">Films similaires a un film donne</div>', unsafe_allow_html=True)
        ftitles = movies[['item_id','title']].set_index('item_id')['title'].to_dict()
        ref_id  = st.selectbox("Film de reference", list(ftitles.keys()),
                               format_func=lambda x: ftitles[x], index=0)
        ns2     = st.slider("Nb films similaires", 3, 10, 5, key='ib_s')
        sr      = isim[ref_id].drop(ref_id).sort_values(ascending=False).head(ns2)
        sres    = movies[movies.item_id.isin(sr.index)][['item_id','title','year','genres_str']].copy()
        sres['score'] = sr.values[:len(sres)]
        sres    = sres.sort_values('score', ascending=False).reset_index(drop=True)
        st.markdown(f'<div class="info-box">Reference : {ftitles[ref_id][:40]}</div>', unsafe_allow_html=True)
        st.markdown(rtable(sres, badge='b', max_s=1.0), unsafe_allow_html=True)

    st.markdown('<div class="stitle">Heatmap similarite cosinus (15 premiers films)</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(8,4))
    fig.patch.set_facecolor('#0d0d18')
    sns.heatmap(isim.iloc[:15,:15], cmap='Purples', ax=ax,
                linewidths=0.15, linecolor='#050510', vmin=0, vmax=1)
    ax.set_title('Cosine Similarity — Films', color='#e0e0f0')
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)


# ─────────────────────────────────────────────
# PAGE 4 — CONTENT-BASED
# ─────────────────────────────────────────────
elif page == "4. Content-Based":
    st.markdown("""<div class="ph">
        <div class="ph-title">Content-Based Filtering</div>
        <div class="ph-sub">ETAPE 4 — Profil utilisateur base sur les genres</div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="info-box">Profil utilisateur = moyenne des vecteurs de genres des films notes >= 4.\nScore(film) = cosine_similarity(profil_user, vecteur_genre_film)</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])
    with col1:
        uid3 = st.selectbox("Utilisateur", sorted(umx.index.tolist()), index=0, key='cb_u')
        nr3  = st.slider("Nb recommandations", 3, 10, 5, key='cb_n')

        lk   = umx.loc[uid3].dropna(); lk = lk[lk>=4].index
        prof = gmat[gmat.index.isin(lk)].mean()
        top_g = prof[prof>0].sort_values(ascending=False)

        st.markdown('<div class="stitle">Profil de genres</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(4,4))
        fig.patch.set_facecolor('#0d0d18')
        p = top_g.sort_values()
        ax.barh(p.index, p.values, color=C_GREEN, alpha=0.8)
        ax.set_xlim(0, 1.1)
        ax.set_xlabel('Score moyen', color='#7070a0')
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

    with col2:
        with st.spinner("Calcul..."):
            reco3 = cb_reco(uid3, n=nr3)
        st.markdown('<div class="stitle">Top-N recommandations</div>', unsafe_allow_html=True)
        if not reco3.empty:
            st.markdown(rtable(reco3, badge='g', max_s=1.0), unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PAGE 5 — TOP-5
# ─────────────────────────────────────────────
elif page == "5. Top-5 Utilisateur":
    st.markdown("""<div class="ph">
        <div class="ph-title">Top-5 pour un Utilisateur Reel</div>
        <div class="ph-sub">ETAPE 5 — Exemple concret avec les 3 methodes</div>
    </div>""", unsafe_allow_html=True)

    uid4 = st.selectbox("Utilisateur (1 - 943)", sorted(umx.index.tolist()), index=0)

    h4 = ratings[ratings.user_id==uid4].merge(
        movies[['item_id','title','genres_str']], on='item_id'
    ).sort_values('rating', ascending=False)

    st.markdown(f"""<div class="mrow">
      <div class="mcard"><div class="mval">{len(h4)}</div><div class="mlbl">Films notes</div></div>
      <div class="mcard"><div class="mval">{h4.rating.mean():.2f}</div><div class="mlbl">Note moyenne</div></div>
      <div class="mcard"><div class="mval">{int(h4.rating.max())}</div><div class="mlbl">Note max</div></div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="stitle">Films deja vus (top 5 mieux notes)</div>', unsafe_allow_html=True)
    st.dataframe(
        h4[['title','rating','genres_str']].head(5).reset_index(drop=True)
          .rename(columns={'title':'Titre','rating':'Note','genres_str':'Genres'}),
        use_container_width=True, hide_index=True
    )

    with st.spinner("Calcul des 3 methodes..."):
        r4ub = ub_reco(uid4)
        r4ib = ib_reco(uid4)
        r4cb = cb_reco(uid4)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="stitle">User-Based</div>', unsafe_allow_html=True)
        if not r4ub.empty: st.markdown(rtable(r4ub, badge='r'), unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="stitle">Item-Based</div>', unsafe_allow_html=True)
        if not r4ib.empty: st.markdown(rtable(r4ib, badge='b'), unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="stitle">Content-Based</div>', unsafe_allow_html=True)
        if not r4cb.empty: st.markdown(rtable(r4cb, badge='g', max_s=1.0), unsafe_allow_html=True)

    st.markdown('<div class="stitle">Comparaison visuelle</div>', unsafe_allow_html=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.patch.set_facecolor('#0d0d18')
    for ax, (df, col, label, color) in zip(axes, [
        (r4ub, 'score', 'User-Based',   C_RED),
        (r4ib, 'score', 'Item-Based',   C_BLUE),
        (r4cb, 'score', 'Content-Based',C_GREEN)]):
        if not df.empty:
            bars = ax.barh(df['title'].str[:28], df[col], color=color, alpha=0.85)
            for b in bars:
                ax.text(b.get_width()+0.02, b.get_y()+b.get_height()/2,
                        f'{b.get_width():.2f}', va='center', fontsize=8, color='#e0e0f0')
        ax.set_title(label, color='#e0e0f0')
        ax.invert_yaxis()
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)


# ─────────────────────────────────────────────
# PAGE 6 — EVALUATION
# ─────────────────────────────────────────────
elif page == "6. Evaluation":
    st.markdown("""<div class="ph">
        <div class="ph-title">Evaluation des Recommandations</div>
        <div class="ph-sub">ETAPE 6 — RMSE + Precision + Comparaison — Train/Test 80/20</div>
    </div>""", unsafe_allow_html=True)

    n_sample = st.slider("Taille echantillon de test", 500, 3000, 1500, step=250)

    @st.cache_data
    def run_eval(n_sample):
        tr, te = train_test_split(ratings, test_size=0.2, random_state=42)
        tm = tr.pivot_table(index='user_id', columns='item_id', values='rating').fillna(0)
        us = pd.DataFrame(cosine_similarity(tm),   index=tm.index,    columns=tm.index)
        is_ = pd.DataFrame(cosine_similarity(tm.T), index=tm.columns, columns=tm.columns)

        def p_ub(uid, iid, n=20):
            if uid not in us.index or iid not in tm.columns: return None
            nb = us[uid].drop(uid).sort_values(ascending=False).head(n)
            col = tm[iid]; v = nb[nb.index.isin(col[col>0].index)]
            if v.empty or v.sum()==0: return None
            return np.dot(v.values, col[v.index].values) / v.sum()

        def p_ib(uid, iid, n=10):
            if uid not in tm.index or iid not in is_.index: return None
            rated = tm.loc[uid]; rated = rated[rated>0]
            sims = is_[iid][rated.index].sort_values(ascending=False).head(n)
            if sims.sum()==0: return None
            return np.dot(sims.values, rated[sims.index].values) / sims.sum()

        def p_cb(uid, iid):
            if uid not in tm.index or iid not in gmat.index: return None
            rated = tm.loc[uid]
            liked = rated[rated>=4].index
            if len(liked)==0: liked = rated[rated>0].index
            lin = [i for i in liked if i in gmat.index]
            if not lin: return None
            prof = gmat.loc[lin].mean().values.reshape(1,-1)
            return 1 + cosine_similarity(prof, gmat.loc[[iid]])[0][0] * 4

        sample = te.sample(n=min(n_sample, len(te)), random_state=42)
        ub_p, ib_p, cb_p, act = [], [], [], []
        for _, row in sample.iterrows():
            uid, iid, r = int(row.user_id), int(row.item_id), row.rating
            p1,p2,p3 = p_ub(uid,iid), p_ib(uid,iid), p_cb(uid,iid)
            if p1 and p2 and p3:
                ub_p.append(p1); ib_p.append(p2); cb_p.append(p3); act.append(r)

        act = np.array(act)
        rmse = lambda p: np.sqrt(mean_squared_error(act, p))
        prec = lambda p,t=0.5: sum(abs(a-x)<=t for a,x in zip(act,p))/len(act)
        bl   = [act.mean()]*len(act)
        return {
            'n': len(act), 'n_tr': len(tr), 'n_te': len(te),
            'rmse': {'Baseline':rmse(bl), 'User-Based':rmse(ub_p),
                     'Item-Based':rmse(ib_p), 'Content-Based':rmse(cb_p)},
            'prec': {'Baseline':prec(bl), 'User-Based':prec(ub_p),
                     'Item-Based':prec(ib_p), 'Content-Based':prec(cb_p)},
        }

    with st.spinner("Evaluation en cours (30-60s)..."):
        ev = run_eval(n_sample)

    st.markdown(f"""<div class="mrow">
      <div class="mcard"><div class="mval">{ev['n_tr']:,}</div><div class="mlbl">Train (80%)</div></div>
      <div class="mcard"><div class="mval">{ev['n_te']:,}</div><div class="mlbl">Test (20%)</div></div>
      <div class="mcard"><div class="mval">{ev['n']}</div><div class="mlbl">Paires evaluees</div></div>
    </div>""", unsafe_allow_html=True)

    best = min([(k,v) for k,v in ev['rmse'].items() if k!='Baseline'], key=lambda x:x[1])[0]
    cmap = {'Baseline':'#2a2a45','User-Based':C_RED,'Item-Based':C_BLUE,'Content-Based':C_GREEN}

    cards = ""
    for name in ['Baseline','User-Based','Item-Based','Content-Based']:
        bc  = 'best' if name==best else ''
        col = cmap[name]
        tag = '<div class="ebest-tag">meilleur RMSE</div>' if name==best else ''
        cards += f"""<div class="ecard {bc}">
          <div class="emethod" style="color:{col}">{name}</div>
          <div class="ermse">{ev['rmse'][name]:.4f}</div>
          <div class="eprec">Precision : {ev['prec'][name]:.1%}</div>
          {tag}
        </div>"""
    st.markdown(f'<div style="display:flex;gap:0.8rem;flex-wrap:wrap;margin-bottom:1.5rem">{cards}</div>', unsafe_allow_html=True)

    labels = list(ev['rmse'].keys())
    rmses  = list(ev['rmse'].values())
    precs  = list(ev['prec'].values())
    cols_l = [cmap[l] for l in labels]

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    fig.patch.set_facecolor('#0d0d18')

    for ax, (vals, title, pct) in zip(axes, [
        (rmses, 'RMSE  (bas = meilleur)', False),
        (precs, 'Precision +/-0.5  (haut = meilleur)', True)]):
        bars = ax.bar(labels, vals, color=cols_l, alpha=0.85, width=0.45)
        ax.set_title(title, color='#e0e0f0', pad=10)
        ax.set_ylim(0, max(vals)*1.25)
        if pct: ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        for b, v in zip(bars, vals):
            lbl = f'{v:.1%}' if pct else f'{v:.3f}'
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+max(vals)*0.02,
                    lbl, ha='center', fontsize=9, color='#e0e0f0',
                    fontfamily='monospace')

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

    st.markdown(f'<div class="ok-box">Meilleure approche : {best} — RMSE = {ev["rmse"][best]:.4f} | Precision = {ev["prec"][best]:.1%}</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PAGE 7 — CONTRAINTES TECHNIQUES
# ─────────────────────────────────────────────
elif page == "7. Contraintes Techniques":
    st.markdown("""<div class="ph">
        <div class="ph-title">Contraintes Techniques</div>
        <div class="ph-sub">ETAPE 7 — Verification des exigences du sujet</div>
    </div>""", unsafe_allow_html=True)

    # ── Contrainte 1 : Similarite cosinus ──
    st.markdown("""<div class="c7-box">
        <div class="c7-title">Contrainte 1 — Similarite cosinus obligatoire</div>
        <div class="c7-check"><strong>User-Based :</strong> cosine_similarity(matrix_filled) — entre lignes (utilisateurs)</div>
        <div class="c7-check"><strong>Item-Based :</strong> cosine_similarity(matrix_filled.T) — entre colonnes (films)</div>
        <div class="c7-check"><strong>Content-Based :</strong> cosine_similarity(profil_user, genre_matrix) — entre profil et films</div>
    </div>""", unsafe_allow_html=True)

    # Demo cosine similarity
    st.markdown('<div class="stitle">Demonstration — Cosine Similarity entre 2 utilisateurs</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        u_a = st.selectbox("Utilisateur A", sorted(umx.index.tolist()), index=0, key='c7_ua')
    with col2:
        u_b = st.selectbox("Utilisateur B", sorted(umx.index.tolist()), index=1, key='c7_ub')

    vec_a = mfill.loc[u_a].values
    vec_b = mfill.loc[u_b].values
    dot   = np.dot(vec_a, vec_b)
    norm  = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
    cos   = dot / norm if norm != 0 else 0.0
    cos_sk = usim.loc[u_a, u_b]

    st.markdown(f"""<div class="mrow">
      <div class="mcard"><div class="mval">{dot:.1f}</div><div class="mlbl">Produit scalaire</div></div>
      <div class="mcard"><div class="mval">{norm:.2f}</div><div class="mlbl">Produit des normes</div></div>
      <div class="mcard"><div class="mval">{cos:.4f}</div><div class="mlbl">Similarite cosinus</div></div>
      <div class="mcard"><div class="mval">{cos_sk:.4f}</div><div class="mlbl">Verification sklearn</div></div>
    </div>""", unsafe_allow_html=True)

    films_communs = umx.loc[[u_a, u_b]].dropna(axis=1, how='any')
    st.markdown(f'<div class="info-box">Films notes en commun entre user {u_a} et user {u_b} : {films_communs.shape[1]} films</div>', unsafe_allow_html=True)

    # ── Contrainte 2 : Exemple concret utilisateur reel ──
    st.markdown("""<div class="c7-box" style="margin-top:1.5rem">
        <div class="c7-title">Contrainte 2 — Exemple concret sur un utilisateur reel</div>
    </div>""", unsafe_allow_html=True)

    uid_demo = st.selectbox("Choisir l'utilisateur de demonstration", sorted(umx.index.tolist()), index=0, key='c7_demo')

    h_demo = ratings[ratings.user_id==uid_demo].merge(
        movies[['item_id','title','genres_str']], on='item_id'
    ).sort_values('rating', ascending=False)

    st.markdown(f'<div class="info-box">Utilisateur {uid_demo} — {len(h_demo)} films notes — Moyenne : {h_demo.rating.mean():.2f} / 5</div>', unsafe_allow_html=True)

    st.markdown('<div class="stitle">Historique (top 5)</div>', unsafe_allow_html=True)
    st.dataframe(
        h_demo[['title','rating','genres_str']].head(5).reset_index(drop=True)
               .rename(columns={'title':'Titre','rating':'Note','genres_str':'Genres'}),
        use_container_width=True, hide_index=True
    )

    with st.spinner("Calcul User-Based + Item-Based..."):
        demo_ub = ub_reco(uid_demo, n=5)
        demo_ib = ib_reco(uid_demo, n=5)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="stitle">Recommandations User-Based</div>', unsafe_allow_html=True)
        if not demo_ub.empty:
            st.markdown(rtable(demo_ub, badge='r'), unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="stitle">Recommandations Item-Based</div>', unsafe_allow_html=True)
        if not demo_ib.empty:
            st.markdown(rtable(demo_ib, badge='b'), unsafe_allow_html=True)

    # ── Contrainte 3 : Comparaison des 2 approches ──
    st.markdown("""<div class="c7-box" style="margin-top:1.5rem">
        <div class="c7-title">Contrainte 3 — Comparaison des deux approches</div>
    </div>""", unsafe_allow_html=True)

    # Films en commun
    if not demo_ub.empty and not demo_ib.empty:
        common = set(demo_ub['title']) & set(demo_ib['title'])
        only_ub = set(demo_ub['title']) - set(demo_ib['title'])
        only_ib = set(demo_ib['title']) - set(demo_ub['title'])

        st.markdown(f"""<div class="mrow">
          <div class="mcard"><div class="mval">{len(common)}</div><div class="mlbl">Films en commun</div></div>
          <div class="mcard"><div class="mval">{len(only_ub)}</div><div class="mlbl">Exclusifs User-Based</div></div>
          <div class="mcard"><div class="mval">{len(only_ib)}</div><div class="mlbl">Exclusifs Item-Based</div></div>
        </div>""", unsafe_allow_html=True)

    # Tableau comparatif
    st.markdown('<div class="stitle">Comparaison methodologique</div>', unsafe_allow_html=True)
    comp = pd.DataFrame({
        'Critere': [
            'Base de calcul',
            'Similarite calculee entre',
            'Matrice utilisee',
            'Cold start nouvel user',
            'Cold start nouveau film',
            'Complexite calcul',
            'Interpretabilite',
        ],
        'User-Based': [
            'Notes des utilisateurs',
            'Paires d\'utilisateurs',
            'User x Film (remplie par 0)',
            'Impossible (pas de vecteur)',
            'Possible si d\'autres users ont note',
            'O(U^2 x I) — couteux si U grand',
            'Intuitive (voisins similaires)',
        ],
        'Item-Based': [
            'Notes des utilisateurs',
            'Paires de films',
            'Film x User (transposee)',
            'Possible si l\'user a note au moins 1 film',
            'Impossible (pas de vecteur)',
            'O(I^2 x U) — stable si I stable',
            'Intuitive (films similaires)',
        ]
    })
    st.dataframe(comp, use_container_width=True, hide_index=True)

    # Graphe comparatif des scores
    if not demo_ub.empty and not demo_ib.empty:
        st.markdown('<div class="stitle">Comparaison des scores Top-5</div>', unsafe_allow_html=True)
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        fig.patch.set_facecolor('#0d0d18')

        for ax, (df, label, color) in zip(axes, [
            (demo_ub, f'User-Based — User {uid_demo}', C_RED),
            (demo_ib, f'Item-Based — User {uid_demo}', C_BLUE)]):
            bars = ax.barh(df['title'].str[:30], df['score'], color=color, alpha=0.85)
            for b in bars:
                ax.text(b.get_width()+0.05, b.get_y()+b.get_height()/2,
                        f'{b.get_width():.3f}', va='center', fontsize=8, color='#e0e0f0')
            ax.set_title(label, color='#e0e0f0')
            ax.set_xlim(0, 5.5)
            ax.invert_yaxis()

        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

    st.markdown("""<div class="ok-box">
    Les 3 contraintes techniques sont respectees :
    1. Similarite cosinus utilisee dans les 3 approches (User-Based, Item-Based, Content-Based)
    2. Exemple concret sur un utilisateur reel avec historique affiche
    3. Comparaison User-Based vs Item-Based avec tableau methodologique et graphe
    </div>""", unsafe_allow_html=True)
