# PredictionFocusedTopicModel
Implementation of Ren et. al. Prediction Focused Topic Models (https://arxiv.org/pdf/1910.05495.pdf)

## Usage

```
python main.py
```
Will train a model with the default parameters:
```
1. --K 5: Number of Topics
2. --model pfslda: Which model (sLDA or pf-sLDA) to train
3. --p 0.15: Value for switch prior in pf-sLDA, only used when model is pf-sLDA
4. --alpha True: If topic prior alpha is considered fixed (to vector of ones)
5. --path None: Load saved model before continuing training
6. --lr 0.025: Initial learning rate to ADAM
7. --lambd 0: Supervised task regularizer weight
8. --num_epochs 500: Number of epochs to train for
9. --check 10: Number of epochs between logging stats and possible saving during training
10. --batch_size 100: batch size during training
11. --y_thresh None: If specified, the yscore (RMSE or AUC) threshold required before saving model
12. --c_thresh None: If specified, the topic coherence threshold required before saving model
```

This training will produce similar results to those given in the paper. Training on a single GPU took around 30 minutes.

We also provide a .ipynb for more interactive running (for those without GPU access, Google Colab is a good option).
