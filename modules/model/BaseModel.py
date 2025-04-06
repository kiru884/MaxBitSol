from platform import architecture

import torch
import torch.nn as nn


class BaseModel(nn.Module):
    def __init__(self, config, device):
        super().__init__()

        architecture_type = config["architecture_type"]
        if architecture_type == "hard":
            self.model = nn.Sequential(
                nn.Linear(in_features=config["in_features"], out_features=2 * config["complexity"]),
                nn.BatchNorm1d(num_features=2 * config["complexity"]),
                nn.ReLU(),
                nn.Linear(in_features=2 * config["complexity"], out_features=config["complexity"]),
                nn.BatchNorm1d(num_features=config["complexity"]),
                nn.ReLU(),
                nn.Linear(in_features=config["complexity"], out_features=config["num_classes"])
            )
        elif architecture_type == "simple":
            self.model = nn.Sequential(
                nn.Linear(in_features=config["in_features"], out_features=config["complexity"]),
                nn.BatchNorm1d(num_features=config["complexity"]),
                nn.ReLU(),
                nn.Linear(in_features=config["complexity"], out_features=config["num_classes"]),
            )
        else:
            raise NotImplementedError
        self.govno = nn.Linear(in_features=config["in_features"], out_features=config["complexity"])

        if config["path"] != "None":
            state_dict = torch.load(config["path"])
            self.model.load_state_dict(state_dict)

        self.model.to(torch.device(device))
        self.govno.to(torch.device(device))
        print("Model loaded...")


    def forward(self, x):
        # print(x)
        # print("X max value: ", torch.max(x))
        # print("X min value: ",torch.min(x))
        #
        # print("GOVNO:")
        # print(self.model)
        # print(self.model(x))
        # print(torch.max(self.model(x)))
        # print(torch.min(self.model(x)))


        return self.model(x)


class BaseModelEmbed(nn.Module):
    def __init__(self, config, device):
        super().__init__()

        emb_dim, emb_num = config["emb_dim"], config["emb_num"]
        # здесь вместо nn.Embedding использован nn.Linear изза того что на вход подается не индекс а OHE фичи.
        # Использование nn.Linear в таком случае идентично использованию nn.Embedding
        self.embed = nn.Linear(in_features=emb_num, out_features=emb_dim, bias=False)

        self.model_first_block = nn.Sequential(
            nn.Linear(in_features=config["in_features"], out_features=2 * config["complexity"]),
            nn.BatchNorm1d(num_features=2 * config["complexity"]),
            nn.ReLU()
        )
        self.model_last_block = nn.Sequential(
            nn.Linear(in_features=(2 * config["complexity"]) + emb_dim, out_features=config["complexity"]),
            nn.BatchNorm1d(num_features=config["complexity"]),
            nn.ReLU(),
            nn.Linear(in_features=config["complexity"], out_features=config["num_classes"])
        )

        if config["path"] != "None":
            state_dict = torch.load(config["path"])
            self.model.load_state_dict(state_dict)

        self.model_first_block.to(torch.device(device))
        self.model_last_block.to(torch.device(device))
        self.embed.to(torch.device(device))
        print("Model loaded..")


    def forward(self, x):
        inp, emb_inp = x
        inp_x = self.model_first_block(inp)
        emb_x = self.embed(emb_inp)
        return self.model_last_block(torch.concatenate([inp_x, emb_x], dim=1))



