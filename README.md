# doscl-cmaha-ood
The codes for the paper: 'Optimizing Upstream Representations for Out-of-Domain Detection with Supervised Contrastive Learning', presented at ACM CIKM' 23.

By running the demo code, it trains a model for out-of-domain (OOD) text detection using supervised contrastive learning on the CLINC [1] dataset. For instant use and easy reproduction, the original data has been converted into the numpy (.npy) format and is stored in `/data/clinc_dataset`. The description and copyright information for the CLINC dataset can be found [here](https://github.com/clinc/oos-eval).

This code also partially references the usage examples of sentence-bert [2] and supervised contrastive learning for image classification [3]. The original repositories for [sbert](https://github.com/UKPLab/sentence-transformers) and [supcon](https://github.com/HobbitLong/SupContrast) can be found at their respective links.

### Recommended Environment & Running Method:
These codes were originally running and tested on Python 3.7.6.

First, please install the necessary libraries by running:
```
pip install -r requirements.txt
```

Then, execute the bash file by:
```
chmod u+x run.sh
sh run.sh
```

### Note: 
If you encounter issues with VRAM capacity, please adjust the `train_batch_size` in `a_supcon_model_training.py` to a lower value. As a reference, the default batch size of 512 requires approximately 35GB of VRAM.

### Reference:
If you find our study useful, please consider citing us:
```
@inproceedings{wang2023optimizing,
  title={Optimizing Upstream Representations for Out-of-Domain Detection with Supervised Contrastive Learning},
  author={Wang, Bo and Mine, Tsunenori},
  booktitle={Proceedings of the 32nd ACM International Conference on Information and Knowledge Management},
  pages={2585--2595},
  year={2023}
}
```
<span style="font-size: 10px;">
[1] Larson, S., Mahendran, A., Peper, J. J., Clarke, C., Lee, A., Hill, P., ... & Mars, J. . An Evaluation Dataset for Intent Classification and Out-of-Scope Prediction. EMNLP 2019 (pp. 1311-1316). <br>
[2] Reimers, N., & Gurevych, I. . Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. EMNLP 2019 (pp. 3982-3992). <br>
[3] Khosla, P., Teterwak, P., Wang, C., Sarna, A., Tian, Y., Isola, P., ... & Krishnan, D. . Supervised contrastive learning. NIPS 2020, 33, 18661-18673. <br>
</span>
