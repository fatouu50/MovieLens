"""
setup_data.py — Télécharge et installe automatiquement le dataset MovieLens 100K.

Usage :
    python setup_data.py
"""

import os
import urllib.request
import zipfile
import shutil

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
ZIP_PATH = "ml-100k.zip"

REQUIRED_FILES = [
    'u.data', 'u.item', 'u.user',
    'u.genre', 'u.occupation',
    'u1.base', 'u1.test'
]


def download_dataset():
    print("📥 Téléchargement de MovieLens 100K...")
    urllib.request.urlretrieve(URL, ZIP_PATH, reporthook=progress_bar)
    print("\n✅ Téléchargement terminé.")


def progress_bar(block_num, block_size, total_size):
    downloaded = block_num * block_size
    pct = min(downloaded / total_size * 100, 100)
    bar = int(pct / 2)
    print(f"\r[{'█' * bar}{'░' * (50 - bar)}] {pct:.1f}%", end='', flush=True)


def extract_and_install():
    print("📂 Extraction de l'archive...")
    os.makedirs(DATA_DIR, exist_ok=True)

    with zipfile.ZipFile(ZIP_PATH, 'r') as z:
        z.extractall(".")

    for filename in REQUIRED_FILES:
        src = os.path.join("ml-100k", filename)
        dst = os.path.join(DATA_DIR, filename)
        if os.path.exists(src):
            shutil.copy(src, dst)
            print(f"  ✅ {filename}")
        else:
            print(f"  ⚠️  {filename} introuvable dans l'archive")

    # Nettoyage
    shutil.rmtree("ml-100k", ignore_errors=True)
    os.remove(ZIP_PATH)
    print("\n🧹 Nettoyage terminé.")


def check_existing():
    missing = []
    for f in REQUIRED_FILES:
        if not os.path.exists(os.path.join(DATA_DIR, f)):
            missing.append(f)
    return missing


if __name__ == "__main__":
    print("=" * 55)
    print("  Setup — MovieLens 100K Recommender System")
    print("=" * 55)

    missing = check_existing()

    if not missing:
        print("✅ Tous les fichiers de données sont déjà présents !")
        print(f"   Dossier : {DATA_DIR}")
    else:
        print(f"⚠️  Fichiers manquants : {', '.join(missing)}")
        print()
        try:
            download_dataset()
            extract_and_install()
            print("\n✅ Dataset prêt ! Lance l'app avec : streamlit run app.py")
        except Exception as e:
            print(f"\n❌ Erreur : {e}")
            print("\nTélécharge manuellement depuis :")
            print("  https://grouplens.org/datasets/movielens/100k/")
            print(f"  Puis place ces fichiers dans : {DATA_DIR}/")
            for f in REQUIRED_FILES:
                print(f"    - {f}")
