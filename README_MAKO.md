# Branche `mako` — Securisation centralisee

## Pourquoi cette branche ?

Avant cette branche, la cle API TMDB etait ecrite directement dans le code.
Nimporte qui lisant le code pouvait la voir et lutiliser.
Cette branche centralise toute la securite dans un seul fichier : `src/security.py`.

---

## Ce qui a change

### 1. Chargement securise de la cle TMDB
**Avant (dangereux) :**
```python
TMDB_API_KEY = "abc123monvraicle"  # visible par tout le monde
```

**Apres (securise) :**
```python
from src.security import get_tmdb_api_key
TMDB_API_KEY = get_tmdb_api_key()  # charge depuis .env, jamais dans le code
```

La fonction cherche la cle dans cet ordre :
1. Colab Secrets (si on tourne sur Google Colab)
2. Variable d environnement `TMDB_API_KEY`
3. Fichier `.env` local (jamais commite sur GitHub)

---

### 2. Validation des entrees utilisateur
Toutes les donnees venant de l utilisateur sont verifiees avant utilisation.

- `validate_film_id(id)` — verifie que l ID est un entier entre 1 et 1682
- `validate_genre(genre)` — verifie que le genre est dans la liste autorisee
- `sanitize_search_query(query)` — supprime les balises HTML et caracteres dangereux
- `sanitize_html(text)` — protege contre l injection HTML dans les templates

Sans ca, un utilisateur pourrait entrer `<script>alert('hack')</script>` dans la recherche.

---

### 3. Requetes HTTP securisees vers TMDB
La fonction `safe_tmdb_request()` remplace les appels directs a `requests.get()`.

Elle ajoute automatiquement :
- **Whitelist de domaines** : seul `api.themoviedb.org` est autorise
- **Timeout obligatoire** : 5 secondes max, evite que l app se bloque
- **Rate limiting** : pause automatique entre les appels (max 40 req/10s)
- **Retry automatique** : 2 tentatives si erreur reseau temporaire
- **Gestion des erreurs** : 401 (cle invalide), 429 (trop de requetes), 500 (serveur en panne)

---

### 4. Verification au demarrage
`check_env()` verifie que tout est en ordre quand l app se lance :
- La cle TMDB est bien configuree
- Le dossier `data/` existe
- Les fichiers `u.data`, `u.item`, `u.user` sont presents
- Le fichier `.env` est bien dans `.gitignore`

---

### 5. Protection des secrets avec .env
Le fichier `.env.example` montre la structure attendue.
Il faut le copier en `.env` et y mettre sa vraie cle :

```bash
cp .env.example .env
# Puis editer .env :
TMDB_API_KEY=ta_vraie_cle_ici
```

Le fichier `.env` est dans `.gitignore` donc il ne sera JAMAIS envoye sur GitHub.

---

## Fichiers ajoutes dans cette branche

| Fichier | Role |
|---|---|
| `src/security.py` | Toutes les fonctions de securite |
| `.env.example` | Template de configuration (sans vraie cle) |
| `.gitignore` | Liste des fichiers exclus de Git |
| `README_MAKO.md` | Ce fichier |

---

## Lancer le projet

```bash
cp .env.example .env
# Ajouter votre cle TMDB dans .env
streamlit run app.py
```
