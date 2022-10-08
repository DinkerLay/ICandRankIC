import numpy as np
from scipy.stats import spearmanr

# %%
# eval matric
def corr(ys_true, ys_pred):
    ys_true, ys_pred = ys_true.data.numpy().ravel(), ys_pred.data.numpy().ravel()
    return np.corrcoef(ys_true, ys_pred)

def rankic(ys_true, ys_pred):
    ys_true, ys_pred = ys_true.data.numpy().ravel(), ys_pred.data.numpy().ravel()
    return spearmanr(ys_true, ys_pred)[0]

def ndcg(ys_true, ys_pred, n=100):
    # quantile
    ys_true, ys_pred = ys_true.data.numpy().ravel(), ys_pred.data.numpy().ravel()
    ys_true[ys_true.argsort()] = np.arange(1, ys_true.shape[0] + 1) / ys_true.shape[0]
    ys_pred[ys_pred.argsort()] = np.arange(1, ys_pred.shape[0] + 1) / ys_pred.shape[0]

    # assert (ys_true > 0).all()

    def dcg(ys_true, ys_pred):
        argsort = ys_pred.argsort()[::-1]
        ys_true_sorted = ys_true[argsort]
        ret = 0
        for i, l in enumerate(ys_true_sorted[:n], 1):
            ret += (2 ** l - 1) / np.log2(1 + i)
        return ret

    ideal_dcg = dcg(ys_true, ys_true)
    pred_dcg = dcg(ys_true, ys_pred)
    return (pred_dcg / ideal_dcg)

# def add_pred()