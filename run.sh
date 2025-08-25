#!/bin/bash

# ==============================================================================
# SCRIPT D'EXÉCUTION DU PROJET AUTOFINETUNE VIA DOCKER
# ==============================================================================
# Ce script vérifie les prérequis et lance le conteneur Docker pour
# démarrer le processus de fine-tuning.
# ==============================================================================

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_info "Vérification des prérequis..."

if ! command -v docker &> /dev/null; then
    log_error "Docker n'est pas installé. Veuillez l'installer avant de continuer."
    exit 1
fi

if ! docker info &> /dev/null; then
    log_error "Le démon Docker ne semble pas fonctionner. Veuillez le démarrer."
    exit 1
fi

log_info "Docker est prêt."

IMAGE_NAME="autofinetune"
if ! docker image inspect "$IMAGE_NAME" &> /dev/null; then
    log_error "L'image Docker '${IMAGE_NAME}' n'a pas été trouvée."
    read -p "$(echo -e "${YELLOW}Voulez-vous la construire maintenant (Y/n) ? ${NC}")" REPLY
    REPLY=${REPLY:-Y}
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_info "Opération annulée."
        exit 1
    else
        log_info "Construction de l'image..."
        if ! docker build -t ${IMAGE_NAME} .; then
            log_error "La construction de l'image a échoué. Arrêt du script."
            exit 1
        fi
        log_info "Image construite avec succès."
    fi
fi

log_info "Lancement du conteneur de fine-tuning 'autofinetune'..."
log_info "Les logs seront mappés dans './logs' et le modèle de sortie dans './output_model'."
log_info "Appuyez sur CTRL+C pour arrêter le processus."

docker run --rm -it \
  --name autofinetune \
  -v "$(pwd)/config":/app/config \
  -v "$(pwd)/logs":/app/logs \
  -v "$(pwd)/output_model":/app/output_model \
  -v "$(pwd)/data":/app/data \
  ${IMAGE_NAME}

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    log_info "Exécution terminée avec succès."
else
    log_error "Le conteneur s'est terminé avec un code d'erreur : $EXIT_CODE"
fi
