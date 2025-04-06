import torch

from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau


OPTIMIZERS = {
    "opt.AdamW": torch.optim.AdamW,
    "opt.Adam": torch.optim.Adam,
    "opt.RMSprop": torch.optim.RMSprop,
    "opt.SGD": torch.optim.SGD,
    "sch.CosineAnnealingLR": CosineAnnealingLR,
    "sch.ReduceLROnPlateau": ReduceLROnPlateau
}
