import numpy as np

from sklearn.metrics import (f1_score, recall_score, precision_score, log_loss, accuracy_score, roc_auc_score,
                             average_precision_score, roc_curve)


def get_f_ROCAUC_AP(scorer_name, reduction='none'):
    scorer = {
        "roc_auc": roc_auc_score,
        "ap": average_precision_score
    }[scorer_name]

    def f(y_true, y_pred):
        metrics_per_cls = []
        assert y_true.shape[1] == y_pred.shape[1]
        for i in range(y_true.shape[1]):
            if np.sum(y_true[:, i]) == 0:
                metrics_per_cls.append(-1)
            else:
                metrics_per_cls.append(scorer(y_true[:, i], y_pred[:, i]))

        if reduction == "none":
            return metrics_per_cls
        elif reduction == "average":
            metrics_per_cls = [i for i in metrics_per_cls if i != -1]
            return sum(metrics_per_cls) / len(metrics_per_cls)

    return f


def get_base_metric_function(scorer_name, reduction='none', with_logits=False):
    scorer = {
        "f1": f1_score,
        "recall": recall_score,
        "precision": precision_score
    }[scorer_name]

    def f(y_true, y_pred):
        metrics_per_cls = []
        assert y_true.shape[1] == y_pred.shape[1]
        for i in range(y_true.shape[1]):

            y_pred_c = y_pred[:, i].copy()
            if np.sum(y_true[:, i]) == 0:
                metrics_per_cls.append(-1)
            else:
                if with_logits:
                    fpr, tpr, thresholds = roc_curve(y_true[:, i].copy(), y_pred_c)
                    # find best thr
                    distances = (tpr - fpr) / np.sqrt(2)
                    # binarize
                    y_pred_c = (y_pred_c > thresholds[np.argmax(distances)]).astype(int)

                metrics_per_cls.append(scorer(y_true[:, i], y_pred_c))

        if reduction == "none":
            return metrics_per_cls
        elif reduction == "average":
            metrics_per_cls = [i for i in metrics_per_cls if i != -1]
            return sum(metrics_per_cls) / len(metrics_per_cls)

    return f


def get_f_accuracy(use_sigmoid=False, use_argmax=False, thr=0.5):
    mapper = lambda x: x
    if use_sigmoid:
        mapper = lambda x: 1 / (1 + np.exp(-x))

    def accuracy(y_true, y_pred):
        y_pred = mapper(y_pred)

        if use_argmax:
            y_pred = np.argmax(y_pred, axis=1)
        else:
            y_pred = (y_pred > thr).astype(int)

        return accuracy_score(y_true, y_pred)

    return accuracy


def get_f_log_loss():
    def f(y_true, y_pred):
        labels = np.arange(y_pred.shape[1])
        return log_loss(y_true, y_pred, labels=labels)

    return f


all_metrics = {
    "f1": lambda x: get_base_metric_function("f1", **x),
    "recall": lambda x: get_base_metric_function("recall", **x),
    "precision": lambda x: get_base_metric_function("precision", **x),
    "log_loss": get_f_log_loss,
    "accuracy": get_f_accuracy,
    "roc_auc": lambda x: get_f_ROCAUC_AP("roc_auc", **x),
    "average_precision": lambda x: get_f_ROCAUC_AP("ap", **x)
}