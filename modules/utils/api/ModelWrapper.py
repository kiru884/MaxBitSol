import torch
import torch.nn as nn

from modules.model import MODELS


class ModelWrapper():
    def __init__(self, config):
        super().__init__()

        # init model and load best weights
        self.device = config["device"]
        self.model = MODELS[config["name"]](config, self.device)
        self.model.load_state_dict(torch.load(config["best_model_path"]))
        self.model.eval()

        # softmax layer
        self.softmax = torch.nn.Softmax(dim=1)

        # mapping order
        self.target_mapping = config["target_mapping_order"]


    def __call__(self, inp):
        # inference
        with torch.no_grad():
            predict = self.softmax(self.model(inp))
        predict = list(predict[0].cpu().numpy())
        predict = list(map(float, predict))

        # mapping
        predict = dict(zip(self.target_mapping, predict))
        return predict





