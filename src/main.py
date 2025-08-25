import yaml
from pathlib import Path
from src.utils import logger
from src.data_loader import load_data, format_prompt
from src.model_loader import load_model_and_tokenizer
from src.trainer import train_model
from src.inference import generate_inference_script

def main():
    """Point d'entrée principal pour le processus de fine-tuning."""
    logger.info("========================================")
    logger.info("Démarrage du processus AutoFineTune")
    logger.info("========================================")

    config_path = Path("config/config.yaml")
    if not config_path.exists():
        logger.error("Fichier de configuration 'config/config.yaml' non trouvé.")
        return
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    logger.info("Configuration chargée avec succès.")

    try:
        raw_dataset = load_data(config)
        tokenized_dataset = raw_dataset.map(format_prompt)
        
        logger.info(f"Dataset formaté. Exemple de prompt:\n{tokenized_dataset[0]['text']}")

        if config['model']['source'] == 'ollama':
            logger.error("Le support d'Ollama n'est pas implémenté dans cette version via la bibliothèque `transformers`.")
            logger.error("Veuillez choisir 'huggingface' comme source de modèle.")
            return

        model, tokenizer = load_model_and_tokenizer(config)

        train_model(model, tokenizer, tokenized_dataset, config)

        output_model_name = config["training"]["output_model_name"]
        base_model_name = config["model"]["name"]
        output_dir = f"output_model/{output_model_name}"
        generate_inference_script(output_dir, base_model_name)

        logger.info("========================================")
        logger.info("Processus AutoFineTune terminé avec succès !")
        logger.info(f"Modèle sauvegardé dans: {output_dir}")
        logger.info(f"Script d'inférence généré: 'inference.py'")
        logger.info("========================================")

    except Exception as e:
        logger.error(f"Une erreur critique est survenue: {e}", exc_info=True)
        logger.info("Processus AutoFineTune interrompu en raison d'une erreur.")

if __name__ == "__main__":
    main()
