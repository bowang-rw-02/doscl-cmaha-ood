# This is the realization of Supervised Contrastive Loss for out-of-domain detection,
# we normalize the sentence embeddings before calculation, so the dot product can be regraded as the similarity directly

# This code partly referred the SCL loss realization by Khosla, Prannay, et al. "Supervised contrastive learning.",
# Advances in neural information processing systems 33 (2020): 18661-18673.
# Original code address: https://github.com/HobbitLong/SupContrast/blob/master/losses.py

import torch
from torch import nn, Tensor
from typing import Iterable, Dict
from sentence_transformers import SentenceTransformer


class SupervisedContrastiveNLPLoss(nn.Module):
    def __init__(self, model: SentenceTransformer, temperature=0.1, contrast_mode='all',
                 base_temperature=0.1):
        super(SupervisedContrastiveNLPLoss, self).__init__()
        self.model = model
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        device = (torch.device('cuda')
                  if labels.is_cuda
                  else torch.device('cpu'))

        reps = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
        embeddings_a = torch.nn.functional.normalize(reps[0], p=2, dim=1)
        embeddings_b = torch.nn.functional.normalize(reps[1], p=2, dim=1)

        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        batch_size = embeddings_a.shape[0]
        contrast_count = 2
        contrast_feature = torch.cat((embeddings_a,embeddings_b),dim=0)

        # The learning mode of contrastive learning
        if self.contrast_mode == 'one':
            # only use the first view as the anchor example
            anchor_feature = embeddings_a
            anchor_count = 1
        elif self.contrast_mode == 'all':
            # use both views as anchors
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)


        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)

        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob, which equals exp(zi x za /T)
        exp_logits = torch.exp(logits) * logits_mask

        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)


        # final SCL loss computation
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss








