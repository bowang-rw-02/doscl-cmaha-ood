# This code file override the ROCEvaluator by sentence bert for OOD detection effect evaluation during training


from sentence_transformers.evaluation import SentenceEvaluator
import numpy as np
import logging
import os
import csv
from typing import List

from scipy.spatial import distance
from sklearn.metrics import roc_curve, auc,precision_recall_curve


logger = logging.getLogger(__name__)

class ROCEvaluator(SentenceEvaluator):
    def __init__(self, ind_train_sentences: List[str], ind_ood_dev_sentences: List[str], show_progress_bar: bool = False, batch_size: int = 128, name: str = '', write_csv: bool = True):

        self.ind_train_sent = ind_train_sentences
        self.ind_ood_dev_sent = ind_ood_dev_sentences
        self.show_progress_bar = show_progress_bar
        self.batch_size = batch_size
        self.name = name

        self.csv_file = "auroc_evaluation_" + name + "_results.csv"
        self.csv_headers = ["epoch", "steps", "AUROC"]
        self.write_csv = write_csv

    def __call__(self, model, output_path, epoch  = -1, steps = -1):
        if epoch != -1:
            if steps == -1:
                out_txt = " after epoch {}:".format(epoch)
            else:
                out_txt = " in epoch {} after {} steps:".format(epoch, steps)
        else:
            out_txt = ":"

        ind_train_embeddings = model.encode(self.ind_train_sent, show_progress_bar=self.show_progress_bar, batch_size=self.batch_size, convert_to_numpy=True)
        ind_ood_dev_embeddings = model.encode(self.ind_ood_dev_sent, show_progress_bar=self.show_progress_bar, batch_size=self.batch_size, convert_to_numpy=True)


        ### AUROC calculation for dev
        NUM_OF_IND_CATES = 150
        USED_IND_DATA_per_cate = 100

        USED_NETWORK_DIM = ind_train_embeddings.shape[1]

        ind_samples = ind_train_embeddings
        eval_samples = ind_ood_dev_embeddings

        label_ind = np.ones(shape=3000)
        label_ood = np.zeros(shape=100)
        Y = np.concatenate((label_ind, label_ood), axis=0)


        c_avg_present = []
        c_cov = np.zeros([USED_NETWORK_DIM, USED_NETWORK_DIM])

        for i in range(NUM_OF_IND_CATES):
            cate_vec = ind_samples[0 + i * USED_IND_DATA_per_cate: USED_IND_DATA_per_cate + i * USED_IND_DATA_per_cate]
            cate_cov = np.cov(cate_vec.T)

            c_avg_present.append(cate_vec.mean(axis=0))
            c_cov = c_cov + cate_cov

        c_avg_present = np.array(c_avg_present)
        c_cov_avged = c_cov / 150

        iV = np.linalg.pinv(c_cov_avged)

        EVAL_DATA_SIZE = ind_ood_dev_embeddings.shape[0]
        eval_distance = np.zeros(shape=EVAL_DATA_SIZE)

        print('One epoch training finished, starting evaluation...')

        for i in range(EVAL_DATA_SIZE):
            min_distance = 1000
            for j in range(NUM_OF_IND_CATES):
                distance_now = distance.mahalanobis(eval_samples[i], c_avg_present[j], iV)
                if distance_now < min_distance:
                    min_distance = distance_now
                if i%1500 == 0 and j%5000==0:
                    print('Finished ', i, ' points evaluation...')
            eval_distance[i] = min_distance

        pred_y_P = -eval_distance
        fpr, tpr, thresholds = roc_curve(Y, pred_y_P)

        auroc = auc(fpr, tpr)
        auroc *= 100
        ###


        logger.info("AUROC evaluation (higher = better) on "+self.name+" dataset"+out_txt)
        logger.info("AUROC (*100):\t{:4f}".format(auroc))

        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            output_file_exists = os.path.isfile(csv_path)
            with open(csv_path, newline='', mode="a" if output_file_exists else 'w', encoding="utf-8") as f:
                writer = csv.writer(f)
                if not output_file_exists:
                    writer.writerow(self.csv_headers)

                writer.writerow([epoch, steps, auroc])

        return auroc