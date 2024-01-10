# This is the code file for centroid-Mahalanobis distance-based out-of-domain detection


import numpy as np
from scipy.spatial import distance
from sklearn.metrics import roc_curve, auc,precision_recall_curve

NETWORK_DIM = 768
NUM_OF_IND_CATES = 150
USED_IND_DATA_per_cate = 100
all_sent_vec_data = np.load('./data/sent_encoding/clinc_train_test_sentvec_supcon.npy')

# clinc dataset, data structure:
# training data: all IND, 150 categories * 100 example per category
# test data: 4500 IND texts, 1000 OOD texts
ind_train_samples = all_sent_vec_data[:15000]
test_samples = all_sent_vec_data[15000:]

label_ind = np.zeros(shape=4500)
label_ood = np.ones(shape=1000)
Y = np.concatenate((label_ind,label_ood),axis=0)

all_samples = np.concatenate((ind_train_samples,test_samples), axis=0)


c_avg_present = []
c_cov = np.zeros([NETWORK_DIM, NETWORK_DIM])

for i in range(150):
    cate_vec = ind_train_samples[0+i*USED_IND_DATA_per_cate : USED_IND_DATA_per_cate+i*USED_IND_DATA_per_cate]
    cate_cov = np.cov(cate_vec.T)

    # centroids of each category
    c_avg_present.append(cate_vec.mean(axis=0))
    # sum of covs of each category
    c_cov = c_cov+cate_cov

c_avg_present = np.array(c_avg_present)
c_cov_avged = c_cov/NUM_OF_IND_CATES


iV = np.linalg.pinv(c_cov_avged)


test_distance = np.zeros(shape=5500)
print('Calculating distances between cluster centroid and test points...')

for i in range(5500):
    min_distance = 1000
    for j in range(NUM_OF_IND_CATES):
        distance_now = distance.mahalanobis(test_samples[i],c_avg_present[j], iV)
        if distance_now<min_distance:
            min_distance = distance_now
        if i%100 == 0 and j%5000 == 0:
            print('Finished ', i, 'points calculation.')
    test_distance[i] = min_distance

pred_y_P = test_distance


precision, recall, thresholds = precision_recall_curve(Y, pred_y_P)
PR_auc = auc(recall, precision)
print('For this model, the auPR (OOD) of it is: ', PR_auc, '.')


fpr, tpr, thresholds = roc_curve(Y, pred_y_P)
fpr95 = 1
fpr90 = 1
for ffpr, ttpr in zip(fpr, tpr):
    if abs(ttpr - 0.95) < 0.01:
        fpr95 = ffpr
        break
for ffpr, ttpr in zip(fpr, tpr):
    if abs(ttpr - 0.90) < 0.01:
        fpr90 = ffpr
        break
print('fpr95: ', fpr95, ' . fpr90: ', fpr90)


roc_auc = auc(fpr, tpr)
print('and the auroc is: ', roc_auc, '.')