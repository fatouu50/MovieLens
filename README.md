# MovieLens Recommender System

Système de recommandation de films basé sur le dataset **MovieLens 100K**, avec une interface interactive Streamlit.

---

## Dataset

| Statistique | Valeur |
|---|---|
| Utilisateurs | 943 |
| Films | 1 682 |
| Évaluations | 100 000 |
| Notes | 1 à 5 étoiles |
| Sparsité | ~93.7% |

Les données sont issues du projet [MovieLens](https://grouplens.org/datasets/movielens/100k/) (GroupLens Research).

---

## Modèles implémentés

| Modèle | Description |
|---|---|
| **Baseline (Popularité)** | Recommande les films les mieux notés ayant reçu un minimum de votes — aucune personnalisation, sert de référence de comparaison |
| **User-Based CF** | Filtrage collaboratif utilisateur-utilisateur — trouve des utilisateurs aux goûts similaires via similarité cosinus, puis prédit les notes par moyenne pondérée |
| **Item-Based CF** | Filtrage collaboratif item-item — compare les films selon leurs patterns de notation communs pour recommander des films proches de ceux déjà appréciés |
| **Content-Based** | Recommandation par similarité de genres — représente chaque film par un vecteur binaire de genres et calcule la similarité cosinus ; intègre un algorithme **MMR** (Maximal Marginal Relevance) pour diversifier les résultats |

---

## Comment ça marche

Le système combine plusieurs approches complémentaires :

- **Baseline** : classe les films par note moyenne, filtré par un seuil minimum de votes pour éviter les biais sur peu d'évaluations.
- **User-Based CF** : construit une matrice utilisateur × film, calcule la similarité cosinus entre utilisateurs, et prédit les notes manquantes via les *k* voisins les plus proches.
- **Item-Based CF** : transpose la même logique sur les films — deux films sont similaires s'ils sont notés de façon cohérente par les mêmes utilisateurs.
- **Content-Based** : encode les genres avec un `MultiLabelBinarizer`, calcule la similarité cosinus entre films, puis applique le **MMR** (Maximal Marginal Relevance) pour équilibrer pertinence et diversité dans les résultats.

La performance du modèle User-Based CF est évaluée via le **RMSE** sur le jeu de test.

---

## Sécurité

### Deux niveaux de protection

**Niveau 1 — Transport réseau (TLS/SSL)**
Le cadenas visible dans le navigateur. Il chiffre les données entre l'utilisateur et le serveur.
- Les requêtes vers TMDB utilisent `https://` → chiffrement TLS actif automatiquement
- En production sur Streamlit Cloud, TLS est fourni automatiquement par la plateforme
- Rien à configurer — il est déjà actif.

**Niveau 2 — Sécurité applicative (`src/security.py`)**
Protège les données à l'intérieur du code.

| Sécurité | Technique | Qui s'en occupe |
|---|---|---|
| Chiffrement réseau | TLS/SSL (https://) | Automatique — Streamlit Cloud |
| Clé API cachée | Variables d'environnement (.env) | `src/security.py` |
| Validation des entrées | Sanitisation + whitelist | `src/security.py` |
| Contrôle des requêtes | Rate limiting + timeout + whitelist domaines | `src/security.py` |
| Vérification au démarrage | `check_env()` | `src/security.py` |

### Ce que fait `src/security.py`

- **Chargement sécurisé de la clé TMDB** — cherche la clé dans Colab Secrets, puis variable d'environnement, puis fichier `.env` local (jamais dans le code)
- **Validation des entrées utilisateur** — `validate_film_id()`, `validate_genre()`, `sanitize_search_query()` protègent contre les injections
- **Requêtes HTTP sécurisées** — `safe_tmdb_request()` impose un timeout de 5s, un rate limiting (max 40 req/10s), une whitelist sur `api.themoviedb.org` et 2 tentatives automatiques en cas d'erreur
- **Vérification au démarrage** — `check_env()` vérifie la clé TMDB, le dossier `data/`, les fichiers requis et que `.env` est bien dans `.gitignore`

### Configuration de la clé TMDB

```bash
cp .env.example .env
# Puis éditer .env :
TMDB_API_KEY=ta_vraie_cle_ici
```

Le fichier `.env` est dans `.gitignore` — il ne sera **jamais** envoyé sur GitHub.

---

## Installation & Lancement

### 1. Cloner le dépôt
```bash
git clone https://github.com/fatouu50/MovieLens.git
cd MovieLens
```

### 2. Créer un environnement virtuel
```bash
python3 -m venv .venv
source .venv/bin/activate       # Linux / macOS
.venv\Scripts\activate          # Windows
```

### 3. Configurer la clé API
```bash
cp .env.example .env
# Ajouter votre clé TMDB dans .env
```

### 4. Installer les dépendances
```bash
pip install -r requirements.txt
```

### 5. Lancer l'application
```bash
streamlit run app.py
```

L'application s'ouvre automatiquement sur `http://localhost:8501`

---

## Structure du projet

```
MovieLens/
├── app.py                  # Interface Streamlit
├── setup_data.py           # Script de préparation des données
├── requirements.txt
├── README.md
├── .env.example            # Template de configuration (sans vraie clé)
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
    ├── recommenders.py     # Modèles de recommandation
    └── security.py         # Sécurité applicative centralisée
```

---

## Pages de l'application

- **Accueil & Stats** — Statistiques générales, distributions des notes, genres, démographie
- **Recommandations personnalisées** — Recommandations User-Based CF ou Item-Based CF pour un utilisateur donné
- **Films Similaires** — Trouver des films similaires à un film donné (Item-Based CF ou Content-Based)
- **Par Genre** — Recommandation par sélection de genres (Content-Based)
- **Films Populaires** — Classement baseline par popularité

---

## Travail en équipe — Branches

Ce projet est développé en équipe. Chaque membre travaille sur sa propre branche avant de fusionner sur `main`.

### Créer et basculer sur sa branche

**Fatouma**
```bash
git checkout -b fatouma
```

**Mako**
```bash
git checkout -b mako
```

**Madina**
```bash
git checkout -b madina
```

**Kadiga**
```bash
git checkout -b kadiga
```

### Workflow quotidien
```bash
# 1. Se mettre à jour depuis main
git pull origin main

# 2. Travailler sur ses fichiers, puis sauvegarder
git add .
git commit -m "Description de ce que tu as fait"

# 3. Pousser sa branche sur GitHub
git push origin nom-de-ta-branche
```

### Fusionner son travail dans main

Une fois le travail validé, ouvrir une **Pull Request** sur GitHub depuis ta branche vers `main`.

---

## Technologies

| Outil | Rôle |
|---|---|
| Python 3.10+ | Langage principal |
| Streamlit | Interface web interactive |
| Pandas / NumPy | Manipulation des données |
| Scikit-learn | Similarité cosinus, encodage des genres |
| Plotly | Visualisations interactives |

---

## Licence

Dataset MovieLens : usage académique et non-commercial.  
Code : MIT License.
