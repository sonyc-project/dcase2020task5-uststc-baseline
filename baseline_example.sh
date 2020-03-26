#!/usr/bin/env bash

# Activate environment
source activate sonyc-ust-stc

# Extract embeddings
openl3 audio $SONYC_UST_PATH/data/audio --output-dir $SONYC_UST_PATH/embeddings --input-repr mel256 --content-type env --audio-embedding-size 512 --audio-hop-size 1.0 --audio-batch-size 16

# Enter source code directory
pushd src

# Train fine-level model and produce predictions
python classify.py $SONYC_UST_PATH/data/annotations.csv $SONYC_UST_PATH/data/dcase-ust-taxonomy.yaml $SONYC_UST_PATH/embeddings $SONYC_UST_PATH/output baseline_fine --label_mode fine

# Evaluate model based on AUPRC metric
python evaluate_predictions.py $SONYC_UST_PATH/output/baseline_fine/*/output.csv $SONYC_UST_PATH/data/annotations.csv $SONYC_UST_PATH/data/dcase-ust-taxonomy.yaml

# Train coarse-level model and produce predictions
python classify.py $SONYC_UST_PATH/data/annotations.csv $SONYC_UST_PATH/data/dcase-ust-taxonomy.yaml $SONYC_UST_PATH/embeddings $SONYC_UST_PATH/output baseline_coarse --label_mode coarse

# Evaluate model based on AUPRC metrics
python evaluate_predictions.py $SONYC_UST_PATH/output/baseline_coarse/*/output.csv $SONYC_UST_PATH/data/annotations.csv $SONYC_UST_PATH/data/dcase-ust-taxonomy.yaml

# Return to the base directory
popd
