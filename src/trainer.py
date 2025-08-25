import transformers
from typing import Dict, Any
from transformers import Trainer, TrainingArguments
from src.utils import logger

def train_model(model, tokenizer, formatted_dataset, config: Dict[str, Any]):
    """Configure et lance le processus de fine-tuning."""
    
    train_conf = config["training"]
    output_dir = f"output_model/{train_conf['output_model_name']}"

    logger.info("Configuration des arguments d'entraînement.")
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=train_conf["per_device_train_batch_size"],
        num_train_epochs=train_conf["epochs"],
        learning_rate=train_conf["learning_rate"],
        optim=train_conf["optimizer"],
        lr_scheduler_type=train_conf["lr_scheduler_type"],
        logging_strategy="steps",
        logging_steps=25,
        save_strategy="epoch",
        fp16=False, 
        bf16=False,
        no_cuda=True,
        report_to="none"
    )

    logger.info("Initialisation du Trainer de Transformers.")
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=formatted_dataset,
        tokenizer=tokenizer,
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    
    logger.info("Lancement du fine-tuning... Cette étape peut être très longue sur CPU.")
    trainer.train()
    
    logger.info("Entraînement terminé.")
    
    logger.info(f"Sauvegarde du modèle fine-tuné dans : {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("Modèle sauvegardé avec succès.")
