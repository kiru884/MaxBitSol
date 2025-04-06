from .losses import get_f_bce_loss
from .metrics import (
    get_base_metric_function, get_f_accuracy, get_f_log_loss, get_f_ROCAUC_AP
)


SCORES = {
    "loss.bce_loss": get_f_bce_loss,

    "metric.f1": lambda x: get_base_metric_function("f1", **x),
    "metric.recall": lambda x: get_base_metric_function("recall", **x),
    "metric.precision": lambda x: get_base_metric_function("precision", **x),
    "metric.log_loss": get_f_log_loss,
    "metric.accuracy": get_f_accuracy,
    "metric.roc_auc": lambda x: get_f_ROCAUC_AP("roc_auc", **x),
    "metric.average_precision": lambda x: get_f_ROCAUC_AP("ap", **x)
}