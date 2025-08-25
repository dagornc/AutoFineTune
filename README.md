# AutoFineTune 🚀

AutoFineTune est un système backend complet, gratuit et open-source pour simplifier et automatiser le processus de fine-tuning des grands modèles de langage (LLM) sur **CPU**.

## ✨ Fonctionnalités

-   **Fine-tuning sur CPU :** Optimisé pour fonctionner sur des serveurs sans GPU grâce à la technique QLoRA 🤖.
-   **Hautement Configurable :** Pilotez l'ensemble du processus via un unique fichier `config/config.yaml`.
-   **Sources Flexibles :** Utilisez des modèles et des datasets depuis le Hub Hugging Face ou en local.
-   **Conteneurisé :** Entièrement déployable avec Docker pour une portabilité maximale 📦.
-   **Gratuit & Open-Source :** Construit uniquement avec des technologies libres.

---

## ⚙️ Prérequis

-   [Docker](https://www.docker.com/) installé et en cours d'exécution.
-   Au moins **16 Go de RAM** sont recommandés pour le fine-tuning de modèles de petite taille.

---

## 🚀 Démarrage Rapide

1.  **Personnalisez la configuration :**
    Ouvrez et modifiez le fichier `config/config.yaml` pour choisir votre modèle, votre dataset et vos hyperparamètres. (Voir les détails ci-dessous).

2.  **Construisez l'image Docker :**
    ```bash
    docker build -t autofinetune .
    ```

3.  **Lancez le processus de fine-tuning :**
    ```bash
    ./run.sh
    ```
    Le script `run.sh` gère le lancement du conteneur Docker avec les bons volumes mappés. Les logs apparaîtront dans le terminal et seront sauvegardés dans le dossier `/logs`.

---

## 🤖 Utiliser le Modèle Fine-tuné

Une fois l'entraînement terminé, un script `inference.py` est généré à la racine du projet. Vous pouvez l'utiliser pour interagir avec votre nouveau modèle.

```bash
# Assurez-vous d'avoir installé les dépendances localement si vous n'utilisez pas Docker
# pip install -r requirements.txt

python inference.py
