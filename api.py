import json
import pandas as pd
import uvicorn

from modules.utils.api.Preprocessor import Preprocessor
from modules.utils.api.ModelWrapper import ModelWrapper
from modules.utils.api.InputModel import InputModel
from fastapi import FastAPI, HTTPException
from typing_extensions import Annotated, deprecated


# CONFIG
CONFIG = {
    "model": {
        "device": "cpu",
        "name": "BaseModelEmb",
        "in_features": 28,
        "complexity": 256,
        "num_classes": 3,
        "emb_dim": 16,
        "emb_num": 132,
        "path": "None",
        "best_model_path": "resources/model_21.pt",
        "target_mapping_order": [
            "Poor",
            "Fair",
            "Good"
        ]
    },
    "preprocessor": {
        "tree_dbh_max_clip_value": 32.5,
        "encoder_path": "resources/encoder",
        "embedding_features_prefix": "spc_latin/",
        "features_columns": [
            "tree_dbh",
            "curb_loc",
            "user_type",
            "borough",
            "sidewalk",
            "guards",
            "spc_latin",
            "steward",
            "root_stone",
            "root_grate",
            "root_other",
            "trunk_wire",
            "trnk_light",
            "trnk_other",
            "brch_light",
            "brch_shoe",
            "brch_other"
        ]
    }
}


# Initializations
app = FastAPI()
# Model wrapper
model = ModelWrapper(CONFIG["model"])
# Preprocessor
preprocessor = Preprocessor(CONFIG["preprocessor"])


@app.post("/get_predict")
async def predict(data: InputModel):
    try:
        # data to data frame
        input_data = json.loads(data.model_dump_json())
        input_data = pd.DataFrame([input_data])
        # preprocess data
        processed_data = preprocessor(input_data)
        # inference
        predict = model(processed_data)

        return predict

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# Запуск приложения
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)