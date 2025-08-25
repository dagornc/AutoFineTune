# Étape 1: Base Image
FROM python:3.12.3-slim

# Définir le répertoire de travail
WORKDIR /app

# Variables d'environnement pour Python
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Copier le fichier des dépendances
COPY requirements.txt .

# Installer les dépendances
# --no-cache-dir pour réduire la taille de l'image
RUN pip install --no-cache-dir -r requirements.txt

# Copier tout le code source de l'application
COPY . .

# Point d'entrée pour exécuter l'application
# Le conteneur lancera le fine-tuning dès son démarrage.
CMD ["python3", "src/main.py"]
