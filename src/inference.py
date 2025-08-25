from src.utils import logger

def generate_inference_script(model_dir: str, base_model_name: str):
    """Génère le script python d'inférence autonome."""
    script_content = f'''
import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# ==============================================================================
# SCRIPT D'INFÉRENCE POUR LE MODÈLE FINE-TUNÉ
# ==============================================================================
# Pour utiliser ce script, assurez-vous que les dépendances sont installées
# (voir requirements.txt) et exécutez : python inference.py

# --- Configuration ---
# Chemin vers le modèle de base original (celui utilisé pour le fine-tuning)
BASE_MODEL_NAME = "{base_model_name}" 
# Chemin vers les poids du modèle fine-tuné (le répertoire de sortie de l'entraînement)
PEFT_MODEL_PATH = "{model_dir}"

def main():
    """Charge le modèle fine-tuné et interagit avec lui."""
    print("Chargement du modèle... (cela peut prendre un certain temps sur CPU)")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    device_map = {{"": "cpu"}}
    
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        quantization_config=bnb_config,
        device_map=device_map,
        trust_remote_code=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Chargement des poids LoRA depuis : {{PEFT_MODEL_PATH}}")
    model = PeftModel.from_pretrained(base_model, PEFT_MODEL_PATH)
    model = model.eval()

    print("\\n--- Modèle prêt pour l'inférence ---")
    print("Entrez votre instruction. Tapez 'exit' ou 'quit' pour quitter.")

    while True:
        try:
            instruction = input("\\nInstruction > ")
        except EOFError:
            break
        if instruction.lower() in ["exit", "quit"]:
            break

        prompt = f"""Ci-dessous se trouve une instruction qui décrit une tâche. Écrivez une réponse qui complète de manière appropriée la demande.

### Instruction:
{{instruction}}

### Response:
"""
        
        inputs = tokenizer(prompt, return_tensors="pt").to("cpu")
        
        print("\\nGénération de la réponse...")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                top_k=50,
                top_p=0.95,
                eos_token_id=tokenizer.eos_token_id
            )
        
        response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        try:
            clean_response = response_text.split("### Response:")[1].strip()
        except IndexError:
            clean_response = response_text
        
        print(f"\\nRéponse du Modèle:\\n{{clean_response}}")

if __name__ == "__main__":
    main()
'''
    # Écrit le contenu dans le fichier inference.py à la racine du projet
    with open("inference.py", "w", encoding="utf-8") as f:
        f.write(script_content)
    logger.info("Script 'inference.py' généré avec succès à la racine du projet.")
