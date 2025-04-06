import random
import numpy as np
import os


def SEED_EVERYTHING(seed=777):
    # python
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # numpy
    np.random.seed(seed)