#!/bin/bash

DATA_ROOT_DIR='/root/autodl-tmp/newcfdemo/CFdemo_HypergraphPathwayEncoder/datasets_csv/reports_clean' # where are the TCGA features stored?
BASE_DIR="/root/autodl-tmp/newcfdemo/CFdemo_HypergraphPathwayEncoder"            # where is the repo cloned?

# STUDY="blca"
# STUDY="brca" 
STUDY="hnsc" 
# STUDY="stad"
# STUDY="coadread"

TYPE_OF_PATH="combine" # what type of pathways? 
MODEL="snn" # what type of model do you want to train?

#改omics_dir
CUDA_VISIBLE_DEVICES=0 python main.py \
    --label_file datasets_csv/clinical_data/tcga_${STUDY}_clinical.csv \
    --study tcga_${STUDY} \
    --split_dir splits \
    --data_root_dir $DATA_ROOT_DIR \
    --task survival \
    --which_splits 5foldcv \
    --omics_dir datasets_csv/raw_rna_data/${TYPE_OF_PATH}/${STUDY}\
    --results_dir "case_results_dbiase001" \
    --batch_size 1 \
    --lr 0.0005 \
    --opt radam \
    --reg 0.0001 \
    --alpha_surv 0.5 \
    --weighted_sample \
    --max_epochs 20 \
    --label_col survival_months_dss \
    --k 5 \
    --bag_loss nll_surv \
    --type_of_path combine \
    --modality $MODEL \
    --enable_multitask \
    --multitask_weight 0.12
    # --multitask_weight 0.12