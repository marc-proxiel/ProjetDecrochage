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

    filiere: str = Field(..., examples=["Informatique"])
    age: int = Field(..., ge=16, le=60, examples=[19])
    bac_type: str = Field(..., examples=["general"])
    mention_bac: str | None = Field(None, examples=["Bien"])
    etablissement_origine: str | None = Field(None, examples=["lycee_public"])

    distance_domicile_km: float | None = Field(None, ge=0, examples=[12.5])
    heures_travail_remunere_sem: float | None = Field(None, ge=0, examples=[5.0])
    taux_presence_pct: float = Field(..., ge=0, le=100, examples=[82.0])
    connexions_lms_30j: float | None = Field(None, ge=0, examples=[30.0])
    heures_lms_total: float = Field(..., ge=0, examples=[42.0])
    ressources_consultees: int = Field(..., ge=0, examples=[60])
    retards_rendus: int = Field(..., ge=0, examples=[1])
    nb_devoirs_rendus: int = Field(..., ge=0, examples=[9])
    messages_forum: int = Field(..., ge=0, examples=[3])

    motivation: float | None = Field(None, ge=1, le=5, examples=[3.0])
    satisfaction: float | None = Field(None, ge=1, le=5, examples=[4.0])
    sentiment_appartenance: float | None = Field(None, ge=1, le=5, examples=[3.0])


# -----------------------------------------------------------------------------
#  SCHEMA 2 : la REPONSE renvoyee par /predict-tabular.
# -----------------------------------------------------------------------------
class PredictionResponse(BaseModel):
    """Resultat de la prediction renvoye au client."""

    proba_abandon: float = Field(..., ge=0.0, le=1.0)
    decision: str
    moyenne_predite: float | None = Field(None, description="Estimation de moyenne_finale /20")
    model_version: str
    threshold: float
