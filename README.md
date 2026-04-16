# 🎬 MovieLens Recommender System

Système de recommandation de films basé sur le dataset **MovieLens 100K**, avec une interface interactive **Streamlit**.

## 📊 Dataset

| Statistique | Valeur |
|---|---|
| Utilisateurs | 943 |
| Films | 1 682 |
| Évaluations | 100 000 |
| Notes | 1 à 5 étoiles |
| Sparsité | ~93.7% |

Les données sont issues du projet [MovieLens](https://grouplens.org/datasets/movielens/100k/) (GroupLens Research).

## 🧠 Modèles implémentés

| Modèle | Description |
|---|---|
| **Popularité** | Baseline — recommande les films les mieux notés |
| **User-Based CF** | Filtrage collaboratif — trouve des utilisateurs similaires |
| **Item-Based CF** | Filtrage collaboratif — trouve des films similaires basé sur les notes |
| **Content-Based** | Recommandation par similarité de genres |

## 🚀 Installation & Lancement

### 1. Cloner le dépôt
```bash
git clone https://github.com/<votre-username>/movielens-recommender.git
cd movielens-recommender
```

### 2. Créer un environnement virtuel
```bash
python -m venv .venv
source .venv/bin/activate       # Linux / macOS
.venv\Scripts\activate          # Windows
```

### 3. Installer les dépendances
```bash
pip install -r requirements.txt
```

### 4. Lancer l'application
```bash
streamlit run app.py
```

L'application s'ouvre automatiquement sur `http://localhost:8501`

## 📁 Structure du projet

```
movielens-recommender/
├── app.py                  # Interface Streamlit
├── requirements.txt
├── README.md
├── .gitignore
├── data/                   # Dataset MovieLens 100K
│   ├── u.data              # 100 000 évaluations
│   ├── u.item              # 1 682 films
│   ├── u.user              # 943 utilisateurs
│   ├── u.genre             # 19 genres
│   ├── u.occupation        # Liste des professions
│   ├── u1.base / u1.test   # Splits train/test (x5 + ua, ub)
│   └── ...
└── src/
    ├── __init__.py
    ├── data_loader.py      # Chargement & préparation des données
    └── recommenders.py     # Modèles de recommandation
```

## 🖥️ Pages de l'application

- **🏠 Accueil & Stats** — Statistiques générales, distributions des notes, genres, démographie
- **👤 Reco Utilisateur** — Recommandations personnalisées (User-CF ou Item-CF)
- **🎥 Films Similaires** — Trouver des films similaires à un film donné
- **🎭 Par Genre** — Recommandation par sélection de genres
- **📊 Évaluation** — Calcul du RMSE, comparaison des modèles

## 📐 Métriques

**RMSE** (Root Mean Square Error) — mesure l'écart entre la note prédite et la note réelle :

```
RMSE = √( Σ(r̂ - r)² / n )
```

## 🛠️ Technologies

- **Python 3.10+**
- **Streamlit** — Interface web
- **Pandas / NumPy** — Manipulation des données
- **Scikit-learn** — Similarité cosinus
- **Plotly** — Visualisations interactives

## 📄 Licence

Dataset MovieLens : usage académique et non-commercial.  
Code : MIT License.
