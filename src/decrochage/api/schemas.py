# =============================================================================
#  src/decrochage/api/schemas.py  —  le CONTRAT des données de l'API (Pydantic)
# -----------------------------------------------------------------------------
#  Place dans le projet : Sprint 3, module API (n°25).
#
#  RÔLE DU FICHIER
#  Ce fichier définit la FORME EXACTE des données que l'API accepte en entrée
#  et renvoie en sortie. On parle de « schémas » ou de « contrat d'I/O »
#  (Input/Output = entrées/sorties).
#
#  POURQUOI C'EST IMPORTANT ?
#  Une API reçoit du JSON venu de l'extérieur : on ne peut JAMAIS faire confiance
#  à ce qui arrive. Un client pourrait envoyer une température "abc", une pression
#  négative, ou oublier un champ. Plutôt que d'écrire nous-mêmes des dizaines de
#  « if » de vérification, on délègue ce travail à Pydantic.
#
#  PYDANTIC, C'EST QUOI ?
#  Une bibliothèque qui, à partir d'une simple classe Python, sait :
#    1) VÉRIFIER que les données reçues respectent les règles (validation) ;
#    2) CONVERTIR les types (ex. la chaîne "51.2" -> le nombre 51.2) ;
#    3) Générer automatiquement la documentation interactive de l'API (Swagger).
#  Si une donnée ne respecte pas le contrat, FastAPI renvoie tout seul une
#  erreur HTTP 422 (Unprocessable Entity = « j'ai compris ta requête mais les
#  données sont invalides ») avec le détail de ce qui ne va pas. On n'a rien à
#  coder pour ça : c'est le gros intérêt de Pydantic.
# =============================================================================

# `from __future__ import annotations` : petite ligne « magique » à mettre en
# tête de fichier. Elle permet d'écrire des annotations de type modernes
# (ex. `list[SensorReading]` au lieu de `List[SensorReading]`) même sur des
# versions de Python plus anciennes, en traitant les annotations comme du texte.
from __future__ import annotations

# `datetime` : le type Python standard pour représenter une DATE + HEURE.
# Pydantic sait transformer une chaîne ISO (ex. "2024-01-01T10:00:00") en un
# vrai objet datetime, et inversement. On s'en sert pour l'horodatage des relevés.
from datetime import datetime

# On importe les deux briques de base de Pydantic :
#   - BaseModel : la classe MÈRE dont héritent tous nos schémas. Hériter d'elle,
#     c'est obtenir « gratuitement » la validation, la conversion et l'export JSON.
#   - Field     : sert à attacher des RÈGLES et des métadonnées à un champ précis
#     (valeur obligatoire, bornes min/max, exemples pour la doc, etc.).
from pydantic import BaseModel, Field


# -----------------------------------------------------------------------------
#  SCHÉMA 1 : un relevé de capteur, à un instant donné.
# -----------------------------------------------------------------------------
class SensorReading(BaseModel):
    """Un point de mesure : une date + une température + une pression."""

    # `timestamp` : l'horodatage du relevé. Le type `datetime` suffit à dire à
    # Pydantic « ce champ doit être une date/heure valide » ; sinon -> erreur 422.
    timestamp: datetime

    # `temperature` : un nombre à virgule (`float`).
    # Décortiquons le `Field(...)` :
    #   - `...`  (les trois points = « Ellipsis ») signifie « champ OBLIGATOIRE » :
    #            si le client l'oublie, Pydantic refuse la requête (422).
    #   - `ge=-20` : « greater than or equal » = la valeur doit être >= -20.
    #   - `le=200` : « less than or equal »    = la valeur doit être <= 200.
    #     Ces deux bornes encodent une RÈGLE MÉTIER : une température de capteur
    #     industriel hors de [-20 °C, 200 °C] est forcément une erreur de mesure.
    #   - `examples=[51.2]` : une valeur d'exemple. Elle ne change RIEN à la
    #     validation ; elle sert juste à pré-remplir la documentation Swagger
    #     pour aider les utilisateurs de l'API à comprendre le format attendu.
    temperature: float = Field(..., ge=-20, le=200, examples=[51.2])

    # `pressure_bar` : la pression, en bars, là encore un `float` obligatoire.
    #   - `gt=0` : « greater than » (STRICTEMENT supérieur à 0). Différence avec
    #     `ge` : ici 0 est INTERDIT (une pression nulle ou négative n'a pas de
    #     sens physique pour ce procédé).
    #   - `le=400` : pression maximale plausible (au-delà -> capteur défaillant).
    pressure_bar: float = Field(..., gt=0, le=400, examples=[195.7])


# -----------------------------------------------------------------------------
#  SCHÉMA 2 : la REQUÊTE envoyée à /predict-tabular (ce que le client poste).
# -----------------------------------------------------------------------------
class TabularPredictionRequest(BaseModel):
    """Demande de prédiction : l'identifiant machine + son historique de relevés."""

    # `machine_id` : l'identifiant de la machine concernée, sous forme de texte.
    #   - `...` : obligatoire.
    #   - `examples=["MACH-01"]` : format canonique attendu (pour la doc).
    #     NB : l'API tolère aussi des variantes (M-7, MACH_07…) qu'elle
    #     normalisera elle-même côté serveur ; voir main.py.
    machine_id: str = Field(..., examples=["MACH-01"])

    # `readings` : une LISTE de relevés `SensorReading`. Pydantic validera CHAQUE
    # élément de la liste avec les règles définies plus haut (température, pression).
    #   - `...`        : la liste est obligatoire.
    #   - `min_length=7` : il faut AU MOINS 7 relevés. POURQUOI 7 ?
    #     Le modèle s'appuie sur des « features temporelles » : des décalages
    #     (lags) jusqu'à 6 pas dans le passé et des moyennes glissantes
    #     (rolling) sur 3 à 6 pas. Pour calculer ces indicateurs sur la DERNIÈRE
    #     mesure sans « tricher » (sans fuite d'information future), il faut
    #     pouvoir regarder assez loin en arrière. Avec moins de 7 relevés, ces
    #     colonnes vaudraient « NaN » (valeur manquante) et la prédiction serait
    #     impossible. Cette borne de 7 évite donc une requête vouée à l'échec :
    #     c'est une règle métier déduite directement de la façon dont on calcule
    #     les features. (Si l'historique reste insuffisant après calcul, main.py
    #     renvoie en plus une 422 « Historique insuffisant ».)
    readings: list[SensorReading] = Field(..., min_length=7)


# -----------------------------------------------------------------------------
#  SCHÉMA 3 : la RÉPONSE renvoyée par /predict-tabular (ce que l'API retourne).
# -----------------------------------------------------------------------------
class PredictionResponse(BaseModel):
    """Résultat de la prédiction renvoyé au client."""

    # `machine_id` : on RENVOIE l'identifiant tel que le client l'a fourni,
    # pour qu'il puisse relier facilement la réponse à sa demande d'origine.
    machine_id: str

    # `proba_panne` : la probabilité de panne estimée par le modèle.
    #   - C'est une PROBABILITÉ, donc forcément comprise entre 0 et 1.
    #   - `ge=0.0` et `le=1.0` documentent (et garantissent) cet intervalle.
    #     Définir ces bornes en SORTIE sert de garde-fou : si jamais notre code
    #     produisait une valeur aberrante (ex. 1.3), Pydantic le signalerait.
    proba_panne: float = Field(..., ge=0.0, le=1.0)

    # `decision` : la décision lisible déduite de la probabilité (ex. "alerte"
    # si la proba dépasse le seuil, sinon "ok"). C'est juste du texte.
    decision: str

    # `model_version` : la version du modèle ayant produit la prédiction.
    # Très utile pour la TRAÇABILITÉ : savoir quel modèle a décidé quoi (audit,
    # débogage, comparaison avant/après un réentraînement).
    model_version: str

    # `threshold` : le SEUIL de décision utilisé (au-dessus = "alerte").
    # Le renvoyer rend la réponse transparente : le client voit la règle exacte
    # qui a transformé la probabilité en décision.
    threshold: float
