# makers/Dockerfile

# Étape 1: Choisir une image Python de base
# Utiliser une image slim pour réduire la taille, Python 3.11 comme spécifié
FROM python:3.11-slim AS builder

# Définir les variables d'environnement pour Python
ENV PYTHONUNBUFFERED 1
ENV PYTHONDONTWRITEBYTECODE 1

# Créer un utilisateur et un groupe non root pour l'application
RUN groupadd -r appgroup && useradd --no-log-init -r -g appgroup appuser

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Copier d'abord le fichier des dépendances et les installer
# Cela permet de mettre en cache cette couche si les dépendances ne changent pas
COPY --chown=appuser:appgroup requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copier le reste du code de l'application avec les bonnes permissions
COPY --chown=appuser:appgroup ./config /app/config
COPY --chown=appuser:appgroup ./src /app/src
COPY --chown=appuser:appgroup ./scripts /app/scripts

# Optionnel: Copier le répertoire data/ si des données par défaut sont nécessaires dans l'image.
# Si vous copiez des données, assurez-vous qu'elles appartiennent également à appuser:appgroup
# COPY --chown=appuser:appgroup ./data /app/data 

# S'assurer que la structure de base du répertoire de données existe
# Ces répertoires appartiendront à root initialement, mais appuser devrait pouvoir écrire dedans
# si /app/data (ou /app) est correctement possédé ou si les permissions sont plus larges.
# Alternativement, créer ces répertoires après avoir changé d'utilisateur.
# Pour une meilleure pratique, créons-les et assurons-nous que appuser peut les utiliser.
# RUN mkdir -p /app/data/corpus /app/data/evaluation && \
#     chown -R appuser:appgroup /app/data

# Définir le PYTHONPATH pour que les imports depuis src/ et config/ fonctionnent correctement
ENV PYTHONPATH="/app"

# Changer d'utilisateur pour exécuter l'application
USER appuser

# Créer les sous-répertoires de données en tant que appuser pour garantir les permissions d'écriture
# DATA_DIR dans settings.py est /app/data
RUN mkdir -p /app/data/corpus/default_corpus/pdfs && \
    mkdir -p /app/data/corpus/default_corpus/metadata && \
    mkdir -p /app/data/evaluation

# Commande par défaut pour afficher l'aide si le conteneur est lancé sans arguments.
# On pourrait aussi lancer un script spécifique, par exemple, l'aide de run_makers.
CMD ["python", "-m", "scripts.run_makers", "--help"]

# Pour exécuter un script spécifique, par exemple :
# docker run makers-app python -m scripts.run_ingestion --max_results 1
# docker run -e OPENAI_API_KEY=$OPENAI_API_KEY -e MONGODB_URI=$MONGODB_URI makers-app python -m scripts.run_makers --query "My query"