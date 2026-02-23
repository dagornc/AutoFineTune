# AutoFineTune

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?logo=pytorch)
![Transformers](https://img.shields.io/badge/Transformers-HuggingFace-yellow.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

Automatisation du **fine-tuning de modèles de langage** à partir d'un dataset et d'un modèle en entrée. Permet d'adapter rapidement des modèles pré-entraînés (LLMs) à des tâches ou domaines spécifiques sans expertise approfondie.

## 📋 Table des matières
- [Présentation](#présentation)
- [Fonctionnalités](#fonctionnalités)
- [Prérequis](#prérequis)
- [Installation](#installation)
- [Usage](#usage)
- [Format du dataset](#format-du-dataset)
- [Modèles supportés](#modèles-supportés)
- [Exemples](#exemples)

## 🎯 Présentation
**AutoFineTune** simplifie le processus de fine-tuning en automatisant les étapes complexes :
- Chargement du modèle de base (HuggingFace Hub ou local)
- Préparation du dataset (tokenisation, formatage, split train/val)
- Configuration de l'entraînement (hyperparametres, scheduler, optimiseur)
- Boucle d'entraînement avec suivi des métriques
- Sauvegarde du modèle fine-tuné

## ✨ Fonctionnalités
- **Fine-tuning automatique** : Spécifiez simplement un modèle + un dataset.
- **Support multi-format** : CSV, JSON, JSONL, TXT.
- **Techniques d'optimisation** : LoRA, QLoRA (fine-tuning efficace sur GPU modeste).
- **Suivi expérimental** : Logs TensorBoard ou Weights & Biases.
- **Sauvegarde de checkpoints** : Récupération en cas d'interruption.

## 📋 Prérequis
- Python 3.8+
- PyTorch 2.0+
- GPU recommandé (CUDA 11.7+)
- 8 GB VRAM minimum (16 GB recommandé)

## 📦 Installation
```bash
git clone https://github.com/dagornc/AutoFineTune.git
cd AutoFineTune
pip install -r requirements.txt
```

## 💻 Usage
### Fine-tuning basique
```bash
python finetune.py \
  --model nom-du-modele-hf \
  --dataset chemin/vers/dataset.json \
  --output ./modele-finetuned \
  --epochs 3
```

### Options CLI
| Argument | Description | Défaut |
|----------|-------------|--------|
| `--model` | Modèle HuggingFace ou chemin local | Requis |
| `--dataset` | Chemin vers le dataset | Requis |
| `--output` | Répertoire de sortie | `./output` |
| `--epochs` | Nombre d'époques | `3` |
| `--batch-size` | Taille du batch | `4` |
| `--lr` | Learning rate | `2e-4` |
| `--lora` | Activer LoRA | `False` |

## 📊 Format du dataset
### Format JSONL (recommandé)
```json
{"instruction": "Qu'est-ce que le fine-tuning ?", "output": "Le fine-tuning est..."}
{"instruction": "Traduis en espagnol : Bonjour", "output": "Hola"}
```

### Format CSV
```csv
instruction,output
"Question","Réponse"
```

## 🤖 Modèles supportés
- Mistral (7B, 8x7B)
- LLaMA 2 & 3 (7B, 13B, 70B)
- Falcon (7B, 40B)
- Phi-2, Phi-3
- GPT-2 et variantes
- Tout modèle Causal LM compatible HuggingFace

## 📝 Exemples
### Fine-tuning avec LoRA (GPU limité)
```bash
python finetune.py \
  --model mistralai/Mistral-7B-v0.1 \
  --dataset dataset_custom.jsonl \
  --lora \
  --batch-size 2 \
  --epochs 5
```

## 📄 Licence
Distribué sous la licence **MIT**.
