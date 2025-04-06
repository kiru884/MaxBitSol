import yaml
import os
import logging
#logging.basicConfig(level=logging.DEBUG)

from modules.trainer import TRAINERS
from modules.utils.seed import SEED_EVERYTHING
from pprint import pprint


if __name__ == "__main__":
    C = [
        # "configs/train_cfg_b1.yaml",
        # "configs/train_cfg_b2.yaml",
        # "configs/train_cfg_b3.yaml",
        "configs/train_cfg_b4.yaml"
    ]

    for _c in C:
        # open config, change exp name and train augmentation, save new config
        with open(_c, "r") as f:
            config = yaml.safe_load(f)

        # show config
        print("*" * 50 + "CONFIG" + "*" * 50)
        pprint(config)
        print("*" * 100)

        # set random seed
        random_state_seed = config["random_seed"] if "random_seed" in config else 777
        SEED_EVERYTHING(random_state_seed)

        # init trainer
        trainer_type = TRAINERS[config["trainer_type"]] if "trainer_type" in config else TRAINERS["BaseTrainer"]

        # train model
        t = trainer_type(config, run=None)
        t.train()
