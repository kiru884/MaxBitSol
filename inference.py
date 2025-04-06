import yaml
import os
import torch
import pandas as pd


from modules.model import MODELS
from tqdm import tqdm
from modules.utils.seed import SEED_EVERYTHING
from pprint import pprint


if __name__ == "__main__":
    # save test table with predicts path
    save_test_table_path = "resources/data/processed/test_inference.csv"
    # model params
    model_config = {
        "device": "cuda:0",
        "name": "BaseModelEmb",
        "in_features": 28,
        "complexity": 256,
        "num_classes": 3,
        "emb_dim": 16,
        "emb_num": 132,
        "path": "None"
    }
    best_model_path = "resources/model_21.pt"
    # data
    processed_csv_path = "resources/data/processed/test.csv"
    target_name = "health"
    second_inp_prefix = "spc_latin/"

    # init model
    # model
    device = model_config["device"]
    model = MODELS[model_config["name"]](model_config, device)
    state_dict = torch.load(best_model_path)
    model.load_state_dict(state_dict)
    model.eval()
    # softmax layer
    softmax = torch.nn.Softmax(dim=1)

    # init dataset
    processed_test_df = pd.read_csv(processed_csv_path)
    features = list(processed_test_df.columns)
    features.remove(target_name)
    features_si = [i for i in features if i.startswith(second_inp_prefix)]
    features = list(set(features).difference(set(features_si)))

    # INFERENCE
    predicts = []
    for ind, r in tqdm(processed_test_df.iterrows(), "Inference by test...", total=len(processed_test_df)):
        # first input: numerical and simple category features
        inp = torch.from_numpy(r[features].to_numpy().astype(float)).float()
        inp = torch.unsqueeze(inp, 0)
        inp = [inp]
        # second input
        inp_embed = torch.from_numpy(r[features_si].to_numpy().astype(float)).float()
        inp_embed = torch.unsqueeze(inp_embed, 0)
        inp += [inp_embed]
        # to device
        inp = [i.to(torch.device(device)) for i in inp]

        with torch.no_grad():
            predict = softmax(model(inp))
        predict = list(predict[0].cpu().numpy())
        predict = list(map(float, predict))
        predicts.append(predict)

    processed_test_df["PREDICT"] = predicts

# save test table with predicts
processed_test_df.to_csv(save_test_table_path, index=False)

