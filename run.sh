#!/usr/bin/env bash

echo "Start training model using SupCon loss..."
python ./a_supcon_model_training.py
wait

echo "Training finished, encoding sentences into embeddings..."
python ./b_sent_encoding.py
wait

echo "Sentence encoding finished, calculating distances and judging OODs..."
python ./c_cmaha_detection.py

echo "Training and testing finished."