# This code file freezes the trained model, and encodes both training and test data to sentence embedding
# for the final, cetroid-Mahalanobis distance-based OOD detection.


import os
os.environ['TRANSFORMERS_CACHE'] = './transformers_cache'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

from sentence_transformers import SentenceTransformer
import numpy as np


# data to be encoded for final detection (testing)
# the training data are also encoded for the cluster centroid calculation and for the final reference.
Texts_train = np.load('./data/clinc_dataset/ind_train_15000_sent_clean.npy')
Texts_test_ind = np.load('./data/clinc_dataset/ind_test_4500_sent_clean.npy')
Texts_test_ood = np.load('./data/clinc_dataset/ood_test_1000_sent_clean.npy')

Texts = np.concatenate((Texts_train, Texts_test_ind, Texts_test_ood), axis=0)


model = SentenceTransformer('./trained_models/supcon_ood_mpnet_normalized_t01_20epoch_withdev')
sentence_embeddings = model.encode(Texts)

np.save('./data/sent_encoding/clinc_train_test_sentvec_supcon.npy',sentence_embeddings)


