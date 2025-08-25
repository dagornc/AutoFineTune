import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from src.utils import logger
from typing import Dict, Any, Tuple

def load_model_and_tokenizer(config: Dict[str, Any]) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Charge le modèle et le tokenizer, en appliquant la quantification QLoRA."""
    model_name = config["model"]["name"]
    quant_config = config["model"]["quantization"]
    qlora_config = config["qlora"]
    
    logger.info(f"Chargement du modèle : {model_name}")

    bnb_config = None
    if quant_config == "4bit":
        logger.info("Configuration de la quantification 4-bit (QLoRA).")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    
    device_map = {"": "cpu"}
    logger.warning("Aucun GPU détecté via la configuration. Le modèle sera chargé sur le CPU. L'entraînement sera très lent.")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map=device_map,
        trust_remote_code=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    logger.info("Préparation du modèle pour l'entraînement k-bit (QLoRA).")
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=qlora_config["r"],
        lora_alpha=qlora_config["lora_alpha"],
        lora_dropout=qlora_config["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"]
    )

    model = get_peft_model(model, lora_config)
    logger.info("Configuration du modèle QLoRA terminée.")
    model.config.use_cache = False
    
    model.print_trainable_parameters()
    
    return model, tokenizer
