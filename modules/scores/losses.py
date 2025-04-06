import torch
import torch.nn as nn


def get_f_bce_loss(pos_weight=None, weight_is_on_gpu=False, convert_from_numpy=False, logits=False):
    l = nn.BCEWithLogitsLoss if logits else nn.BCELoss

    if pos_weight is not None:
        w_device = torch.device("cuda") if weight_is_on_gpu else torch.device("cpu")
        pos_weight = torch.tensor(pos_weight) if type(pos_weight) != torch.tensor else pos_weight
        pos_weight = pos_weight.to(w_device)
        loss = l(pos_weight=pos_weight)
    else:
        loss = l()

    def f(y_true, y_pred_prob):
        if convert_from_numpy:
            y_true = torch.from_numpy(y_true)
            y_pred_prob = torch.from_numpy(y_pred_prob)
            return loss(y_pred_prob, y_true).item()

        l = loss(y_pred_prob, y_true)

        return l

    return f


all_losses = {
    "BCE": get_f_bce_loss,
}
