# This is the code file to train an out-of-domain detection model using pure supervised contrastive learning
# We use the training framework from sbert, and realized the supcon loss and evaluator ourselves

import os
os.environ['TRANSFORMERS_CACHE'] = './transformers_cache'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

from sentence_transformers import SentenceTransformer, InputExample, SentencesDataset, models
from torch.utils.data import DataLoader

import math
import numpy as np

# the supcon loss is realized in supcon_nlp_loss_normalized
from supcon_ood_lib import supcon_nlp_loss_normalized, roc_evaluator_cmaha

# load training data
train_sentences = np.load('./data/clinc_dataset/ind_train_15000_sent_clean.npy')
train_labels = np.load('./data/clinc_dataset/ind_train_label_number.npy')
dev_sentences = np.load('./data/clinc_dataset/ind_ood_val_3100_sent_clean.npy')

# if you find your VRAM not enough, please change to a lower value
train_batch_size = 512
num_epochs = 20
INPUT_DATA_NUM = train_sentences.shape[0]
OUTPUT_PATH = './trained_models/supcon_ood_mpnet_normalized_t01_20epoch_withdev'

# we use mpnet as the base network.
model_name = 'sentence-transformers/all-mpnet-base-v2'
word_embedding_model = models.Transformer(model_name, max_seq_length=64)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

# data formatting
train_data = []
for i in range(INPUT_DATA_NUM):
    train_data.append(InputExample(texts=[train_sentences[i],train_sentences[i]], label=train_labels[i]))
train_dataset = SentencesDataset(train_data, model)
train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)

# the evaluation also need the training data points as reference points. evaluation is also based on centroid-Mahalanobis distance.
dev_evaluator = roc_evaluator_cmaha.ROCEvaluator(ind_train_sentences=train_sentences, ind_ood_dev_sentences=dev_sentences)


train_loss = supcon_nlp_loss_normalized.SupervisedContrastiveNLPLoss(model)
warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=num_epochs,
    warmup_steps=warmup_steps,
    evaluator=dev_evaluator,
    output_path=OUTPUT_PATH,
    show_progress_bar=True,
    save_best_model=True
)
