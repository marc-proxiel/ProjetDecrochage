# =============================================================================
#  src/decrochage/api/security.py  —  GARDE-FOUS de l'API (protection de base)
# -----------------------------------------------------------------------------
#  Place dans le projet : Sprint 3, module SÉCURITÉ (n°26).
#
#  RÔLE DU FICHIER
#  Une API exposée sur le réseau doit se protéger contre les usages abusifs ou
#  malveillants. Ce fichier fournit deux protections complémentaires :
#
#    1) `limit_body_size` : un MIDDLEWARE qui refuse les requêtes dont le corps
#       (le « payload ») est trop gros -> code HTTP 413 (Payload Too Large).
#       But : éviter qu'un client n'envoie un fichier énorme qui saturerait la
#       mémoire/le réseau du serveur (déni de service).
#
#    2) `rate_limit` : un LIMITEUR DE DÉBIT (anti-« flood ») qui plafonne le
#       nombre de requêtes par adresse IP sur une fenêtre de temps glissante
#       -> code HTTP 429 (Too Many Requests).
#       But : empêcher qu'un seul client ne bombarde l'API et la rende
#       indisponible pour les autres.
#
#  POURQUOI PROTÉGER L'API ?
#  Sans ces garde-fous, l'API est vulnérable : un attaquant (ou un client
#  buggé) peut la rendre indisponible (DoS), gaspiller des ressources, ou faire
#  exploser la facture cloud. Ces protections sont SIMPLES mais constituent une
#  première barrière essentielle, en complément de l'authentification par clé
#  API (gérée dans main.py).
#
#  RAPPEL — un MIDDLEWARE, c'est quoi ?
#  Un « intercepteur » placé AUTOUR du traitement de chaque requête. Il voit la
#  requête AVANT qu'elle n'atteigne la route, peut la laisser passer (en
#  appelant `call_next`) ou la bloquer, puis peut agir sur la réponse au retour.
# =============================================================================

# Annotations de type modernes (voir explication dans les autres fichiers).
from __future__ import annotations

# `time` : module standard pour mesurer le temps. On utilisera `time.time()`,
# qui renvoie le nombre de secondes écoulées depuis une date de référence
# (« epoch »). Pratique pour comparer des instants entre eux (fenêtre glissante).
import time

# Deux structures de données venues du module `collections` :
#   - `defaultdict` : un dictionnaire qui CRÉE automatiquement une valeur par
#     défaut quand on accède à une clé encore inexistante. Évite d'écrire
#     « if ip not in dico: dico[ip] = ... » à la main.
#   - `deque` : une « file à double extrémité » (prononcer « deck »). On peut y
#     ajouter/retirer efficacement des éléments AUX DEUX BOUTS. Idéale pour une
#     fenêtre glissante : on ajoute les requêtes récentes à droite et on retire
#     les trop vieilles à gauche, le tout en temps quasi constant.
from collections import defaultdict, deque

# Briques FastAPI/Starlette nécessaires :
#   - `HTTPException` : exception spéciale ; la lever interrompt le traitement et
#     fait renvoyer par FastAPI une réponse d'erreur HTTP propre (code + détail).
#   - `Request`       : l'objet représentant la requête HTTP entrante (en-têtes,
#     adresse IP du client, corps, etc.).
#   - `status`        : un catalogue de constantes nommées pour les codes HTTP
#     (ex. `status.HTTP_429_TOO_MANY_REQUESTS` au lieu du nombre brut 429).
#     Utiliser ces noms rend le code plus lisible et évite les fautes de frappe.
from fastapi import HTTPException, Request, status

# `JSONResponse` : permet de construire et RENVOYER directement une réponse au
# format JSON, avec un code HTTP choisi. On l'utilise dans le middleware quand
# on veut court-circuiter la requête (renvoyer 413 sans appeler la route).
from fastapi.responses import JSONResponse

# -----------------------------------------------------------------------------
#  Réglages et état partagé du module
# -----------------------------------------------------------------------------

# Taille MAXIMALE autorisée pour le corps d'une requête : 64 Kio (kibioctets).
# Calcul : 64 * 1024 = 65 536 octets. On écrit « 64 * 1024 » plutôt que « 65536 »
# car c'est plus parlant (« 64 Ko ») et auto-documenté. Toute requête déclarant
# un corps plus grand que cela sera rejetée par le middleware ci-dessous.
MAX_BODY_BYTES = 64 * 1024

# `_hits` : la mémoire du limiteur de débit. C'est un dictionnaire qui associe
# à CHAQUE adresse IP la liste (une `deque`) des instants de ses requêtes
# récentes. Grâce à `defaultdict(deque)`, la première fois qu'on rencontre une
# IP, une `deque` vide est créée automatiquement pour elle.
#   - Le préfixe « _ » (underscore) signale, par convention Python, que c'est un
#     détail interne au module, pas destiné à être utilisé de l'extérieur.
#   - L'annotation `dict[str, deque]` documente : clés = IP (texte),
#     valeurs = files d'horodatages.
#   ATTENTION (limite à connaître) : cet état vit en MÉMOIRE et par PROCESSUS.
#   Il se vide à chaque redémarrage et n'est pas partagé entre plusieurs
#   instances du serveur. Pour de la production multi-serveurs, on utiliserait
#   plutôt un stockage commun (ex. Redis). Ici, c'est volontairement simple.
_hits: dict[str, deque] = defaultdict(deque)


# -----------------------------------------------------------------------------
#  PROTECTION 1 : limiter la taille du corps de la requête (-> 413)
# -----------------------------------------------------------------------------
# `async def` : fonction ASYNCHRONE. Les middlewares HTTP de FastAPI doivent être
# asynchrones car le serveur traite de nombreuses requêtes « en parallèle » sans
# se bloquer. Le mot-clé `await` (plus bas) sert à attendre une opération async.
async def limit_body_size(request: Request, call_next):
    # Signature imposée par FastAPI pour un middleware « http » :
    #   - `request`   : la requête entrante ;
    #   - `call_next` : une fonction qui, si on l'appelle, transmet la requête à
    #     la suite de la chaîne (les autres middlewares puis la route) et renvoie
    #     la réponse. NE PAS l'appeler = bloquer la requête ici même.

    # En HTTP, l'en-tête « Content-Length » annonce la taille (en octets) du
    # corps que le client compte envoyer. On la récupère ; elle peut être absente
    # (`None`) si le client ne l'a pas fournie.
    content_length = request.headers.get("content-length")

    # Si l'en-tête est présent ET que la taille annoncée dépasse notre maximum :
    #   - `content_length` est du TEXTE -> on le convertit en entier avec `int()`.
    #   - Le test `if content_length and ...` évite de faire `int(None)` (erreur).
    if content_length and int(content_length) > MAX_BODY_BYTES:
        # On REFUSE immédiatement, sans appeler `call_next` : la requête n'ira
        # jamais jusqu'à la route. On renvoie une réponse JSON avec le code 413
        # (« Payload Too Large ») et un message d'explication en français.
        return JSONResponse(status_code=413, content={"detail": "Payload trop volumineux"})

    # Sinon (taille acceptable ou inconnue) : on laisse la requête poursuivre son
    # chemin normalement. `await call_next(request)` exécute la suite et nous rend
    # la réponse finale, que l'on retourne telle quelle.
    return await call_next(request)


# -----------------------------------------------------------------------------
#  PROTECTION 2 : limiter le nombre de requêtes par IP (fenêtre glissante -> 429)
# -----------------------------------------------------------------------------
# Cette fonction n'est PAS un middleware : elle est conçue pour être branchée
# comme « dépendance » FastAPI (via Depends) sur les routes à protéger. À chaque
# appel d'une route protégée, FastAPI exécutera `rate_limit` AVANT la route ;
# si la limite est dépassée, l'exception levée bloque l'accès.
#   - `-> None` : la fonction ne renvoie rien d'utile. Soit elle laisse passer
#     (silencieusement), soit elle lève une exception qui interrompt tout.
def rate_limit(request: Request, limit: int = 60, window: float = 60.0) -> None:
    # Paramètres avec valeurs par défaut :
    #   - `limit`  = 60   : nombre maximum de requêtes autorisées par IP...
    #   - `window` = 60.0 : ...sur une fenêtre de 60 secondes.
    #   => Politique : « au plus 60 requêtes par minute et par adresse IP ».

    # On identifie le client par son adresse IP. `request.client.host` la fournit.
    ip = request.client.host

    # On note l'instant présent (en secondes depuis l'epoch), pour le comparer
    # aux horodatages des requêtes précédentes.
    now = time.time()

    # On récupère la file d'horodatages associée à cette IP. Si l'IP est vue pour
    # la première fois, `defaultdict` crée ici une `deque` vide automatiquement.
    q = _hits[ip]

    # NETTOYAGE — on fait « glisser » la fenêtre : on retire de la file tous les
    # horodatages TROP VIEUX, c'est-à-dire antérieurs à (maintenant - window).
    #   - `q[0]` est l'élément le plus ANCIEN (à gauche de la file).
    #   - Tant qu'il existe et qu'il est plus vieux que la fenêtre, on le retire
    #     par la gauche avec `popleft()` (très efficace sur une deque).
    #   Après cette boucle, `q` ne contient plus que les requêtes des `window`
    #   dernières secondes : c'est ça, la « fenêtre glissante ».
    while q and q[0] < now - window:
        q.popleft()

    # DÉCISION — si, après nettoyage, il reste déjà `limit` requêtes (ou plus)
    # dans la fenêtre, c'est que le quota est atteint : on REFUSE la requête.
    if len(q) >= limit:
        # On lève une `HTTPException` avec le code 429 (« Too Many Requests »).
        # FastAPI l'intercepte et renvoie au client une erreur propre. Le client
        # comprend qu'il doit ralentir et réessayer plus tard.
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Trop de requêtes"
        )

    # SI ON ARRIVE ICI : la requête est acceptée. On enregistre son horodatage à
    # DROITE de la file (`append`), pour qu'elle compte dans les prochains calculs
    # de la fenêtre. Puis la fonction se termine sans rien renvoyer (None) et
    # FastAPI poursuit vers la route demandée.
    q.append(now)
