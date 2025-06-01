# cognitive-swarm-agents/Dockerfile

# Étape 1: Choisir une image Python de base
# Utiliser une image slim pour réduire la taille, Python 3.11 comme spécifié
FROM python:3.11-slim AS builder

# Définir les variables d'environnement pour Python
ENV PYTHONUNBUFFERED 1
ENV PYTHONDONTWRITEBYTECODE 1

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Copier d'abord le fichier des dépendances et les installer
# Cela permet de mettre en cache cette couche si les dépendances ne changent pas
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copier le reste du code de l'application
COPY ./config /app/config
COPY ./src /app/src
COPY ./scripts /app/scripts
# Optionnel: Copier le répertoire data/ si des données par défaut sont nécessaires dans l'image.
# Sinon, monter data/ comme un volume au runtime.
# COPY ./data /app/data 

# S'assurer que le répertoire de données (et ses sous-répertoires si nécessaire) existent
# si des scripts s'attendent à écrire dedans à l'intérieur de l'image par défaut
# (par exemple, si DATA_DIR dans settings.py pointe vers un chemin relatif à /app/data)
# RUN mkdir -p /app/data/corpus/rl_robotics_arxiv/pdfs && \
#     mkdir -p /app/data/corpus/rl_robotics_arxiv/metadata && \
#     mkdir -p /app/data/evaluation

# Définir le PYTHONPATH pour que les imports depuis src/ et config/ fonctionnent correctement
ENV PYTHONPATH="/app"

# Commande par défaut pour afficher l'aide si le conteneur est lancé sans arguments.
# On pourrait aussi lancer un script spécifique, par exemple, l'aide de run_cognitive_swarm.
CMD ["python", "-m", "scripts.run_cognitive_swarm", "--help"]

# Pour exécuter un script spécifique, par exemple :
# docker run cognitive-swarm-app python -m scripts.run_ingestion --max_results 1
# docker run -e OPENAI_API_KEY=$OPENAI_API_KEY -e MONGO_URI=$MONGO_URI cognitive-swarm-app python -m scripts.run_cognitive_swarm --query "My query"