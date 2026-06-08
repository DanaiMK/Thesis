#!/bin/bash
#SBATCH --job-name=danai_krikri_ft
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --output=logs/danai_ft_%j.out
#SBATCH --error=logs/danai_ft_%j.err
#SBATCH --account=EUHPC_D34_189

# Μετάβαση στον φάκελο της εργασίας σου στον υπερυπολογιστή
cd $HOME/Thesis

# Φόρτωση του περιβάλλοντος Python (virtual environment)
source ~/.bashrc
conda activate unsloth_env

# Offline mode (ΑΠΑΡΑΙΤΗΤΟ: τα compute nodes δεν έχουν ίντερνετ)
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Εκτέλεση του δικού μας αρχείου Python
python krikri_finetune.py