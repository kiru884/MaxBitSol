import numpy as np
import pandas as pd
import pickle
import torch

from tqdm import tqdm


class Preprocessor():
    def __init__(self, config):
        self.features_columns = config["features_columns"]
        self.encoder_path = config["encoder_path"]
        self.tree_dbh_max_clip_value = config["tree_dbh_max_clip_value"]
        self.embedding_features_prefix = config["embedding_features_prefix"]

        # open encoder
        with open(self.encoder_path, "rb") as f:
            encoders = pickle.load(f)
        self.order = encoders["order"]
        self.encoders = encoders["encoders"]

        # check features
        assert set(self.order) == set(self.features_columns)


    def __call__(self, table):
        prep_table = table[self.features_columns]

        # process tree_dbh
        prep_table["tree_dbh"] = np.clip(prep_table["tree_dbh"], a_min=0, a_max=self.tree_dbh_max_clip_value)
        prep_table["tree_dbh"] = np.log10(prep_table["tree_dbh"] + 1)

        # transform categorical features
        for f in tqdm(self.order, desc="Features processing..."):
            if self.encoders[f] == -1:
                continue

            # transform
            transformed = self.encoders[f].transform(np.array(prep_table[f].tolist()).reshape(-1, 1)).toarray()
            cats = list(map(str, self.encoders[f].categories_[0]))
            cats = cats[1:] if len(cats) <= 2 else cats
            cats = [f"{f}/{c}" for c in cats]

            # concat
            transformed = pd.DataFrame(transformed, columns=cats)
            new_columns = list(prep_table.columns) + list(transformed.columns)
            prep_table = pd.concat([prep_table, transformed], axis=1, ignore_index=True)
            prep_table.columns = new_columns
            prep_table = prep_table.drop(columns=[f])

        # split preprocessed data on simple data and embedding data for model input
        no_emb_features = list(prep_table.columns)
        emb_features = [i for i in no_emb_features if i.startswith(self.embedding_features_prefix)]
        no_emb_features = list(set(no_emb_features).difference(set(emb_features)))

        # simple input, no embedding data
        inp = torch.from_numpy(prep_table[no_emb_features].to_numpy().astype(float)).float()
        #inp = torch.unsqueeze(inp, 0)
        inp = [inp]
        # embedding data input
        inp_embed = torch.from_numpy(prep_table[emb_features].to_numpy().astype(float)).float()
        #inp_embed = torch.unsqueeze(inp_embed, 0)
        inp += [inp_embed]

        return inp
