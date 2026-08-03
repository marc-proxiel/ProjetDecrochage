# =============================================================================
#  src/decrochage/config.py  —  CONFIGURATION CENTRALE du projet
# -----------------------------------------------------------------------------
#  But : rassembler au même endroit tous les RÉGLAGES (chemins de données,
#  graine aléatoire, nom de la cible…) au lieu de les écrire « en dur » un peu
#  partout dans le code.
#
#  Astuce clé : on utilise `pydantic-settings`. Chaque réglage peut être
#  SURCHARGÉ sans toucher au code, via une variable d'environnement ou un
#  fichier `.env`. Exemple : poser `DECROCHAGE_RANDOM_SEED=7` change la graine.
#  C'est la bonne pratique « 12-factor app » : la config vit hors du code.
# =============================================================================
from pathlib import Path  # objet « chemin de fichier » portable Windows/macOS/Linux

# BaseSettings : une classe spéciale qui lit automatiquement les variables
# d'environnement ; SettingsConfigDict : pour configurer son comportement.
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Tous les réglages du projet, typés et surchargeables par l'environnement."""

    # --- Comportement de lecture de la configuration -------------------------
    model_config = SettingsConfigDict(
        env_prefix="DECROCHAGE_",  # ne lit que les variables qui commencent par DECROCHAGE_
        env_file=".env",  # lit aussi un fichier `.env` à la racine s'il existe
        extra="ignore",  # ignore les variables inconnues au lieu de planter
    )

    # --- Les réglages eux-mêmes (avec leur valeur par défaut) ----------------
    # Type annoté = garde-fou : si on passe "abc" à random_seed, pydantic refuse.
    data_dir: Path = Path("data/raw")  # dossier des données BRUTES (capteurs, incidents)
    silver_dir: Path = Path("data/silver")  # dossier du dataset « silver » (analyser, transformé)
    gold_dir: Path = Path("data/gold")  # dossier du dataset « gold » (prêt à entraîner)
    model_dir: Path = Path("artifacts/models")  # dossier où l'on sauvegarde le modèle entraîné
    random_seed: int = 42  # graine aléatoire = résultats REPRODUCTIBLES
    target_col: str = "abandon"  # nom de la colonne à prédire (0 = OK, 1 = abandon)
    incident_window_hours: int = 24  # fenêtre : on étiquette « panne » les 24 h AVANT un incident
    # --- Réglages de l'API (modules 25-26) ---
    # Clé attendue dans X-API-Key ; surcharger en prod via DECROCHAGE_API_KEY.
    api_key: str = "dev-key"
    decision_threshold: float = (
        0.5  # seuil proba → décision « alerte »/« ok » (DECROCHAGE_DECISION_THRESHOLD)
    )


# On crée UNE instance partagée, importable partout via `from decrochage.config import settings`.
# (Au moment de cette ligne, pydantic lit l'environnement / le .env et remplit les valeurs.)
settings = Settings()
