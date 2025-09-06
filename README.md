# AutoFineTune

![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![PyTorch Version](https://img.shields.io/badge/pytorch-2.0%2B-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg?logo=docker)](https://hub.docker.com/)

## 1. Description Fonctionnelle

AutoFineTune est un outil en ligne de commande puissant et flexible, conçu pour simplifier et accélérer le processus de fine-tuning de modèles de Machine Learning. À partir d'un modèle pré-entraîné (provenant, par exemple, de la plateforme Hugging Face) et d'un jeu de données fourni par l'utilisateur, AutoFineTune automatise les étapes clés du ré-entraînement pour adapter le modèle à une tâche spécifique.

### Fonctionnalités Clés

*   **Chargement de Modèles Simplifié** : Chargez n'importe quel modèle compatible depuis Hugging Face Hub ou un répertoire local.
*   **Support de Datasets Multiples** : Compatible avec les datasets de Hugging Face ou vos propres fichiers (CSV, JSON Lines).
*   **Configuration Centralisée** : Pilotez l'ensemble du processus via un unique fichier de configuration (`config.yaml`).
*   **Entraînement Optimisé** : Utilise des techniques modernes comme le mixed-precision training et le gradient accumulation pour une performance maximale.
*   **Suivi des Expériences** : Intégration optionnelle avec des outils comme Weights & Biases ou TensorBoard pour visualiser les métriques d'entraînement.
*   **Environnement Reproductible** : Entièrement conteneurisé avec Docker pour garantir la reproductibilité des résultats.

## 2. Architecture Technique

L'architecture d'AutoFineTune est conçue pour être modulaire et scalable. Elle repose sur des composants découplés qui gèrent chacun une partie spécifique du pipeline de fine-tuning.

```
+--------------------------+
|   Interface CLI (argparse)   |
+--------------------------+
            |
+--------------------------+
|  Configuration (config.yaml) |
+--------------------------+
            |
+--------------------------+
|  Moteur de Fine-Tuning (PyTorch) |
+--------------------------+
            |
+-----------+-----------+
|                       |
+-----------------+ +-----------------+
|  Gestionnaire de  | |  Gestionnaire de  |
|      Modèles      | |     Données       |
| (Hugging Face Hub)| |   (Datasets Lib)  |
+-----------------+ +-----------------+
            |
+--------------------------+
|     Sauvegarde du Modèle     |
+--------------------------+
```

*   **Interface en Ligne de Commande (CLI)** : Construite avec le module `argparse` de Python, elle sert de point d'entrée pour l'utilisateur.
*   **Gestionnaire de Configuration** : Un fichier `config.yaml` centralise tous les hyperparamètres (learning rate, batch size, epochs, etc.) et les chemins d'accès, permettant une gestion simple des expériences.
*   **Moteur de Fine-Tuning** : Le cœur de l'application, écrit en PyTorch. Il utilise la librairie `transformers` de Hugging Face pour la gestion des modèles et `accelerate` pour la distribution de l'entraînement sur plusieurs GPUs.
*   **Gestionnaire de Modèles** : Responsable du chargement du modèle de base et de son tokenizer.
*   **Gestionnaire de Données** : Charge, pré-traite et prépare le dataset pour l'entraînement.

## 3. Algorithme de Fine-Tuning

L'algorithme implémenté suit une approche de **transfer learning**.

1.  **Chargement du Modèle de Base** : Un modèle pré-entraîné sur une grande quantité de données généralistes (par exemple, `bert-base-uncased` pour le NLP) est chargé en mémoire.
2.  **Préparation des Données** : Le jeu de données spécifique à la tâche est tokenisé (converti en une représentation numérique que le modèle peut comprendre) et divisé en lots (batches).
3.  **Adaptation de la Tête du Modèle** : La dernière couche du modèle (la "tête") est remplacée par une nouvelle couche correspondant à la tâche de fine-tuning (par exemple, une couche de classification avec le bon nombre de classes). Les poids de cette nouvelle couche sont initialisés aléatoirement.
4.  **Entraînement Différentiel** : Le modèle est entraîné sur le nouveau jeu de données. Typiquement, la majorité des couches du modèle de base sont "gelées" (leurs poids ne sont pas mis à jour) pendant les premières époques, et seule la nouvelle tête est entraînée. Ensuite, l'ensemble du modèle peut être dégelé et entraîné avec un très faible taux d'apprentissage (learning rate) pour ajuster finement tous les poids à la nouvelle tâche sans "oublier" les connaissances acquises lors du pré-entraînement.
5.  **Sauvegarde** : Le modèle fine-tuné, désormais spécialisé, est sauvegardé pour une utilisation ultérieure en inférence.

## 4. Démarrage Rapide

### Prérequis

*   Python 3.10+
*   Docker & Docker Compose
*   (Optionnel) Un GPU NVIDIA avec les drivers CUDA installés.

### Installation Locale

1.  **Clonez le repository :**
    ```bash
    git clone https://github.com/dagornc/AutoFineTune.git
    cd AutoFineTune
    ```

2.  **Créez un environnement virtuel et installez les dépendances :**
    ```bash
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

### Installation avec Docker (Recommandé)

1.  **Clonez le repository :**
    ```bash
    git clone https://github.com/dagornc/AutoFineTune.git
    cd AutoFineTune
    ```

2.  **Construisez l'image Docker :**
    ```bash
    docker build -t autofinetune .
    ```

## 5. Utilisation

Le fine-tuning est lancé via la commande `run.py`.

### Exemple de Commande

```bash
python run.py --config config.yaml
```

Ou avec Docker :

```bash
docker run --rm -it \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/models:/app/models \
    --gpus all \
    autofinetune \
    --config config.yaml
```

### Fichier de Configuration

Personnalisez le fichier `config.yaml` pour définir votre modèle, votre dataset et vos hyperparamètres.

```yaml
# config.yaml

# Modèle à fine-tuner
model:
  name: "distilbert-base-uncased"
  # chemin local : "path/to/your/model"

# Données pour le fine-tuning
dataset:
  name: "imdb" # Nom sur Hugging Face Hub
  # chemin local : "data/my_dataset.csv"
  text_column: "text"
  label_column: "label"

# Hyperparamètres d'entraînement
training:
  epochs: 3
  batch_size: 16
  learning_rate: 2e-5
  output_dir: "models/fine_tuned_model"
```

## 6. Contribution

Les contributions sont les bienvenues. N'hésitez pas à ouvrir une issue pour discuter des changements ou à soumettre une pull request.

## 7. Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.
