# syntax=docker/dockerfile:1
# =============================================================================
#  FICHIER : Dockerfile
#  RÔLE    : "recette" qui décrit COMMENT construire l'image Docker de l'API
#            Decrochage. Une image = un modèle figé de l'application (code +
#            dépendances + système) à partir duquel on lance des conteneurs.
#  MODULE  : 27 (conteneurisation de l'API / packaging).
#
#  IDÉE GÉNÉRALE POUR DÉBUTANT :
#    - Docker lit ce fichier de HAUT en BAS et exécute chaque instruction.
#    - Chaque instruction crée une "couche" (layer) ; Docker met ces couches
#      en cache pour reconstruire plus vite la fois suivante.
#    - On utilise ici un "build multi-stage" (plusieurs étapes FROM) :
#         * une étape "build"   = sert à INSTALLER/COMPILER (lourde, jetable) ;
#         * une étape "runtime" = image FINALE, mince, qui FAIT TOURNER l'API.
#      Résultat : l'outil d'installation (uv) et les caches ne polluent pas
#      l'image livrée -> image plus petite, plus sûre, plus rapide à déployer.
# =============================================================================

# 'syntax' indique à Docker quelle "grammaire" de Dockerfile utiliser.
# La version 'dockerfile:1' débloque des fonctions modernes comme
# '--mount=type=cache' (voir plus bas). C'est un commentaire SPÉCIAL :
# il DOIT rester la toute première ligne pour être pris en compte.
# Bloc de documentation du projet (commentaires libres, ignorés par Docker).
# Decrochage 4.0 — image API (module 27). Variante A : le modèle pré-entraîné
# (artifacts/models/) est embarqué -> /ready 200 + /predict-tabular OK d'emblée.
# Design relu en binôme (retour_codex §19.2/§35.2) ; entrypoint = uvicorn decrochage.api.main:app.

# ---- build : dépendances figées (uv.lock) puis projet installé en wheel (non-editable) ----
# FROM = image de DÉPART de cette étape. 'python:3.13-slim' = une image officielle
# Python 3.13 en version "slim" (allégée : moins de paquets système -> plus légère).
# 'AS build' donne un NOM à cette étape pour pouvoir y faire référence plus tard.
FROM python:3.13-slim AS build

# On copie l'exécutable 'uv' depuis une autre image (celle d'Astral, les auteurs d'uv).
# 'uv' est un gestionnaire de paquets Python ULTRA rapide (remplace pip/venv ici).
# '--from=...:0.11.19' épingle une VERSION précise d'uv -> builds reproductibles
# (on aura toujours exactement le même uv, pas une version "surprise" du jour).
COPY --from=ghcr.io/astral-sh/uv:0.11.19 /uv /usr/local/bin/uv

# ENV = variables d'environnement (réglages lus par les outils pendant le build).
#   UV_COMPILE_BYTECODE=1 : uv pré-compile le .py en .pyc -> démarrage plus rapide.
#   UV_LINK_MODE=copy      : uv COPIE les fichiers au lieu de faire des liens durs,
#                            ce qui évite des soucis quand on traverse des volumes/caches.
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

# WORKDIR = "dossier de travail" courant DANS l'image. Les commandes suivantes
# s'exécuteront depuis /app, et les chemins relatifs partiront de là.
WORKDIR /app

# On copie D'ABORD uniquement la "liste des dépendances" :
#   - pyproject.toml = description du projet + dépendances voulues ;
#   - uv.lock        = versions EXACTES verrouillées de chaque dépendance.
# Pourquoi seulement ces 2 fichiers d'abord ? Pour le CACHE : tant qu'ils ne
# changent pas, Docker réutilise la couche d'installation sans tout refaire,
# même si on a modifié le code source juste après. C'est une optimisation clé.
COPY pyproject.toml uv.lock ./

# Première installation : SEULEMENT les dépendances (pas encore notre code).
#   --mount=type=cache,target=/root/.cache/uv : monte un CACHE persistant pour uv
#       le temps de cette commande. Les paquets déjà téléchargés sont réutilisés
#       d'un build à l'autre -> beaucoup plus rapide. (Activé grâce au 'syntax' du haut.)
#   uv sync     : installe l'environnement décrit par le projet.
#   --frozen    : interdit de modifier uv.lock ; on installe PILE les versions figées
#                 (si le lock ne correspond pas, ça échoue au lieu de "bidouiller").
#                 -> garantit que tout le monde a EXACTEMENT les mêmes versions.
#   --no-install-project : on n'installe pas encore Decrochage lui-même (juste ses deps).
#   --no-dev    : on saute les dépendances de développement (tests, linters...) :
#                 inutiles en production -> image plus petite et plus sûre.
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-install-project --no-dev

# MAINTENANT seulement on copie le code source de l'application.
# Comme c'est APRÈS l'installation des deps, modifier le code n'invalide pas
# la (longue) couche d'installation précédente : on garde un build rapide.
COPY src ./src

# Deuxième installation : on installe le PROJET Decrochage lui-même.
#   --no-editable : installe une COPIE figée (un "wheel"), PAS un lien vers le code.
#                   En "editable", le paquet pointerait vers ./src ; ici on veut une
#                   version autonome et stable, adaptée à une image de production.
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev --no-editable

# ---- runtime : image mince, utilisateur non-root ----
# NOUVELLE étape FROM = on REPART d'une image Python propre pour l'image FINALE.
# Tout ce qui était dans l'étape 'build' (uv, caches, fichiers temporaires) est
# ABANDONNÉ ici : on ne récupérera que le strict nécessaire -> image minimale.
FROM python:3.13-slim AS runtime

# Sécurité : on crée un utilisateur NON privilégié nommé 'appuser'.
#   useradd -m       : crée aussi son dossier personnel (/home/appuser).
#   -u 10001         : lui fixe un identifiant (UID) élevé et stable.
# Pourquoi ? Par défaut un conteneur tourne en 'root' (tout-puissant). Si l'appli
# est compromise, un attaquant hériterait des droits root. Tourner en utilisateur
# limité réduit fortement les dégâts possibles. (On bascule dessus plus bas.)
RUN useradd -m -u 10001 appuser

# Dossier de travail de l'image finale (mêmes raisons que plus haut).
WORKDIR /app

# Variables d'environnement de l'image finale (le '\' relie les lignes : une seule ENV) :
#   PATH="/app/.venv/bin:$PATH" : on place le venv EN TÊTE du PATH. Ainsi, taper
#       'python' ou 'uvicorn' utilise les binaires de NOTRE environnement virtuel
#       (celui qu'on a installé), sans avoir à "activer" le venv manuellement.
#   PYTHONUNBUFFERED=1 : Python n'attend pas pour afficher les logs (pas de tampon)
#       -> les messages apparaissent en TEMPS RÉEL dans 'docker logs' (debug plus facile).
#   DECROCHAGE_MODEL_DIR=/app/artifacts/models : dit à l'appli OÙ trouver le modèle
#       (chemin du modèle embarqué dans l'image — cf. Variante A ci-dessous).
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    DECROCHAGE_MODEL_DIR=/app/artifacts/models

# On récupère depuis l'étape 'build' UNIQUEMENT l'environnement virtuel déjà installé.
# '--from=build' = "va chercher dans l'étape nommée build". C'est tout l'intérêt du
# multi-stage : on importe le résultat (le .venv prêt) sans importer les outils de build.
COPY --from=build /app/.venv /app/.venv

# Variante A : modèle livré dans l'image. (Variante B : retirer la ligne ci-dessous,
# voir README §Packaging — il faut alors data/ + entraînement, pas un simple `decrochage train` runtime.)
# Ici on COPIE le modèle pré-entraîné DANS l'image. Conséquence : dès le démarrage,
# l'API peut prédire immédiatement (pas besoin d'entraîner ni de télécharger un modèle).
COPY artifacts/models ./artifacts/models

# À partir d'ICI, on bascule sur l'utilisateur non privilégié créé plus haut.
# Toutes les instructions suivantes ET le processus de l'appli tourneront en 'appuser'.
USER appuser

# EXPOSE documente le PORT sur lequel l'application écoute (ici 8000, le port d'uvicorn).
# C'est surtout informatif/déclaratif : ça aide les outils et lecteurs à savoir quel
# port ouvrir. La vraie publication "vers l'extérieur" se fait via 'ports:' (compose).
EXPOSE 8000

# HEALTHCHECK = "examen de santé" périodique du conteneur : Docker exécute une
# commande à intervalles réguliers pour savoir si l'appli répond (healthy) ou non.
#   --interval=30s : on teste toutes les 30 secondes ;
#   --timeout=3s   : chaque test a 3 s pour répondre, sinon il est considéré en échec ;
#   --retries=3    : il faut 3 échecs consécutifs pour déclarer le conteneur "unhealthy".
# La commande de test : un petit script Python qui appelle l'URL interne /health.
#   - urllib.request.urlopen(...) : fait une requête HTTP vers http://localhost:8000/health ;
#   - sys.exit(0 if status==200 else 1) : sort avec le code 0 (=OK/sain) si le serveur
#       répond 200, sinon code 1 (=échec). Docker lit ce code pour juger la santé.
# Utilité : compose/orchestrateur peut attendre que l'API soit réellement prête.
HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
  CMD python -c "import urllib.request,sys;sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

# CMD = commande lancée AU DÉMARRAGE du conteneur (le "point d'entrée" par défaut).
#   uvicorn                    : serveur ASGI qui exécute notre application web Python ;
#   decrochage.api.main:app     : chemin "module:variable" -> dans le fichier
#                                decrochage/api/main.py, on lance l'objet 'app' (l'API) ;
#   --host 0.0.0.0             : écoute sur TOUTES les interfaces réseau du conteneur,
#                                indispensable pour être joignable depuis l'extérieur
#                                (127.0.0.1 ne serait accessible QUE de l'intérieur) ;
#   --port 8000               : port d'écoute (cohérent avec EXPOSE 8000 ci-dessus).
# Forme "JSON/exec" (liste de chaînes) recommandée : pas de shell intermédiaire,
# donc l'appli reçoit correctement les signaux d'arrêt (stop propre du conteneur).
CMD ["uvicorn", "decrochage.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
