# =============================================================================
# Makefile — Raccourcis de commandes pour le projet "Decrochage"
# -----------------------------------------------------------------------------
# ROLE : regroupe les commandes frequentes sous des noms courts ("cibles").
# Au lieu de retaper une longue ligne, on lance par exemple :  make test
# L'outil "make" lit ce fichier et execute la recette associee a la cible.
#
# ATTENTION (syntaxe make) : chaque ligne de recette (les commandes a executer)
# DOIT commencer par une TABULATION, pas par des espaces. C'est une regle stricte
# de make. Les commentaires (#) sont places sur leur propre ligne, jamais sur
# une ligne de recette indentee.
#
# Cibles disponibles ci-dessous :
#   make install       -> installe les dependances (prod + groupe dev)
#   make test          -> lance les tests
#   make lint          -> verifie la qualite du code (ruff)
#   make format-check  -> verifie le formatage (black), sans modifier
#   make check         -> tout enchaine : tests + lint + format + CLI
# =============================================================================

# Cible "install" : installe les dependances du projet ainsi que le groupe dev.
install:
	uv sync --extra dev

# Cible "test" : execute la suite de tests avec pytest (-q = sortie concise).
test:
	uv run pytest -q

# Cible "lint" : analyse la qualite du code avec ruff (detecte les problemes).
lint:
	uv run ruff check .

# Cible "format-check" : verifie le formatage avec black (--check = ne modifie rien,
# echoue seulement si du code n'est pas bien formate).
format-check:
	uv run black --check .

# Cible "check" : verification complete. Enchaine les 4 commandes ci-dessous,
# dans l'ordre. Si l'une echoue, make s'arrete (utile avant un commit/push).
check:
	uv run pytest -q
	uv run ruff check .
	uv run black --check .
	uv run decrochage --help

# Lancer l'API en local avec rechargement auto (module 25)
serve:
	uv run uvicorn decrochage.api.main:app --reload

# --- Docker (J3 : modules 27-28) ---
docker-build:
	docker build -t decrochage:local .

up:
	docker compose up -d --build

down:
	docker compose down

logs:
	docker compose logs -f

ps:
	docker compose ps

smoke:
	curl -fsS http://localhost:8000/health && echo " OK"
