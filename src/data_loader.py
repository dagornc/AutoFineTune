import json
from pathlib import Path
from typing import Dict, Any

from datasets import load_dataset, Dataset
from src.utils import logger

def validate_dataset_row(row: Dict[str, Any], index: int) -> bool:
    """Valide la structure d'une seule ligne du dataset."""
    if not isinstance(row, dict):
        logger.error(f"Validation Erreur: Ligne {index} n'est pas un objet JSON valide.")
        return False
    
    if "instruction" not in row or "output" not in row:
        logger.error(f"Validation Erreur: Ligne {index} manque les clés requises 'instruction' ou 'output'.")
        return False
        
    if not isinstance(row["instruction"], str) or not isinstance(row["output"], str):
        logger.error(f"Validation Erreur: 'instruction' et 'output' à la ligne {index} doivent être des chaînes de caractères.")
        return False
        
    return True

def load_data(config: Dict[str, Any]) -> Dataset:
    """Charge le dataset depuis Hugging Face ou un fichier local."""
    source = config["dataset"]["source"]
    name = config["dataset"]["name"]
    logger.info(f"Chargement du dataset depuis la source : '{source}' avec le nom : '{name}'")

    if source == "huggingface":
        try:
            dataset = load_dataset(name, split="train")
            logger.info("Dataset chargé avec succès depuis Hugging Face.")
            return dataset
        except Exception as e:
            logger.error(f"Impossible de charger le dataset depuis Hugging Face : {e}")
            raise
    
    elif source == "local":
        local_path = Path(name)
        if not local_path.exists():
            logger.error(f"Le fichier de dataset local n'a pas été trouvé : {local_path}")
            raise FileNotFoundError(f"Le fichier {local_path} n'existe pas.")

        data = []
        try:
            with open(local_path, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    row = json.loads(line)
                    if not validate_dataset_row(row, i + 1):
                        raise ValueError("Le format du dataset local est invalide.")
                    data.append(row)
            
            logger.info(f"Dataset local chargé et validé avec succès. {len(data)} lignes trouvées.")
            return Dataset.from_list(data)
        except json.JSONDecodeError as e:
            logger.error(f"Erreur de décodage JSON dans le fichier local : {e}")
            raise
        except Exception as e:
            logger.error(f"Erreur lors de la lecture du fichier de dataset local : {e}")
            raise

    else:
        raise ValueError(f"Source de dataset non reconnue : {source}")

def format_prompt(example: Dict[str, Any]) -> Dict[str, str]:
    """Formate une entrée de dataset en un prompt d'entraînement complet."""
    if "input" in example and example["input"]:
        prompt = f"""Ci-dessous se trouve une instruction qui décrit une tâche, associée à une entrée qui fournit un contexte supplémentaire. Écrivez une réponse qui complète de manière appropriée la demande.

### Instruction:
{example['instruction']}

### Input:
{example['input']}

### Response:
{example['output']}"""
    else:
        prompt = f"""Ci-dessous se trouve une instruction qui décrit une tâche. Écrivez une réponse qui complète de manière appropriée la demande.

### Instruction:
{example['instruction']}

### Response:
{example['output']}"""
    return {"text": prompt}
