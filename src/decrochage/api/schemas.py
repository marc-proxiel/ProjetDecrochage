# =============================================================================
#  src/decrochage/api/schemas.py  —  le CONTRAT des donnees de l'API (Pydantic)
# -----------------------------------------------------------------------------
#  Contrat d'entree/sortie de /predict-tabular : un etudiant a mi-semestre,
#  scoré pour le risque de decrochage (`abandon`). Les champs correspondent
#  aux colonnes disponibles a mi-S1 dans le dataset (voir chapitre 6.5 du
#  notebook EtudeDecrochageMVA.ipynb) — pas de `moyenne_partiels_s1` ni de
#  `nb_ue_validees_s1` (fuite temporelle), pas de `sexe`/`boursier` (retirees
#  du modele), pas de `date_inscription` (non utilisee par le modele).
#
#  NOTE : ce fichier remplace une version anterieure du starter (maintenance
#  predictive industrielle, relevés capteurs) qui ne correspondait plus au cas
#  d'usage actuel du projet.
# =============================================================================
from __future__ import annotations

from pydantic import BaseModel, Field


# -----------------------------------------------------------------------------
#  SCHEMA 1 : la REQUETE envoyee a /predict-tabular (un etudiant, mi-semestre).
# -----------------------------------------------------------------------------
class StudentFeatures(BaseModel):
    """Releve mi-semestre d'un etudiant, tel que dispo pour un scoring precoce."""

    filiere: str = Field(
        ...,
        description=(
            "Filiere de L1, doit correspondre a une entree du catalogue des formations "
            "(Biologie, Droit, Gestion, Informatique, Lettres, Mathematiques, Psychologie, "
            "STAPS). Une filiere inconnue renvoie 422."
        ),
        examples=["Informatique"],
    )
    age: int = Field(..., ge=16, le=60, description="Age de l'etudiant.", examples=[19])
    bac_type: str = Field(
        ...,
        description="Type de bac : general, techno ou pro.",
        examples=["general"],
    )
    mention_bac: str | None = Field(
        None,
        description="Mention au bac : Passable, Assez Bien, Bien ou Tres Bien. Omis si inconnu.",
        examples=["Bien"],
    )
    etablissement_origine: str | None = Field(
        None,
        description="Type d'etablissement d'origine (lycee_public, lycee_prive, cfa, autre).",
        examples=["lycee_public"],
    )

    distance_domicile_km: float | None = Field(
        None, ge=0, description="Distance domicile-campus en kilometres.", examples=[12.5]
    )
    heures_travail_remunere_sem: float | None = Field(
        None, ge=0, description="Heures de travail remunere par semaine.", examples=[5.0]
    )
    taux_presence_pct: float = Field(
        ..., ge=0, le=100, description="Taux de presence en cours, en pourcentage.", examples=[82.0]
    )
    connexions_lms_30j: float | None = Field(
        None,
        ge=0,
        description="Nombre de connexions au LMS sur les 30 derniers jours.",
        examples=[30.0],
    )
    heures_lms_total: float = Field(
        ..., ge=0, description="Temps cumule passe sur le LMS, en heures.", examples=[42.0]
    )
    ressources_consultees: int = Field(
        ..., ge=0, description="Nombre de ressources pedagogiques consultees.", examples=[60]
    )
    retards_rendus: int = Field(
        ..., ge=0, description="Nombre de devoirs rendus en retard.", examples=[1]
    )
    nb_devoirs_rendus: int = Field(
        ..., ge=0, description="Nombre de devoirs rendus a ce jour.", examples=[9]
    )
    messages_forum: int = Field(
        ..., ge=0, description="Nombre de messages postes sur le forum du cours.", examples=[3]
    )

    motivation: float | None = Field(
        None,
        ge=1,
        le=5,
        description="Auto-evaluation de la motivation (echelle 1-5).",
        examples=[3.0],
    )
    satisfaction: float | None = Field(
        None,
        ge=1,
        le=5,
        description="Auto-evaluation de la satisfaction (echelle 1-5).",
        examples=[4.0],
    )
    sentiment_appartenance: float | None = Field(
        None,
        ge=1,
        le=5,
        description="Auto-evaluation du sentiment d'appartenance (echelle 1-5).",
        examples=[3.0],
    )


# -----------------------------------------------------------------------------
#  SCHEMA 2 : la REPONSE renvoyee par /predict-tabular.
# -----------------------------------------------------------------------------
class PredictionResponse(BaseModel):
    """Resultat combine des deux modeles independants (classifieur + regresseur)."""

    proba_abandon: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Probabilite de decrochage (classe 1) predite par la regression logistique.",
        examples=[0.056],
    )
    decision: str = Field(
        ...,
        description='"a_risque" si proba_abandon >= threshold, sinon "ok".',
        examples=["ok"],
    )
    moyenne_predite: float | None = Field(
        None,
        description=(
            "Estimation de moyenne_finale (/20) par le regresseur Gradient Boosting, "
            "independant du classifieur (aucune fuite entre les deux). "
            "Null si aucun regresseur n'est charge."
        ),
        examples=[16.85],
    )
    model_version: str = Field(..., description="Version du package ayant produit la prediction.")
    threshold: float = Field(..., description="Seuil de decision applique a proba_abandon.")
