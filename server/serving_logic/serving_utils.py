import json
import os
from typing import Union

import torch
from torch.jit import RecursiveScriptModule
from ts.context import Context

from constants import MODEL_CONFIG_FILE_NAME
from entities import ModelConfig


def load_model_config(context: Context) -> ModelConfig:
    model_config_path = os.path.join(
        context.system_properties["model_dir"], MODEL_CONFIG_FILE_NAME
    )
    model_config_raw = load_json_file(path=model_config_path)
    if model_config_raw is None:
        raise RuntimeError("Malformed (empty) config found.")
    return ModelConfig.from_dict(model_config_raw)


def load_model(model_config: ModelConfig, context: Context) -> RecursiveScriptModule:
    weights_path = os.path.join(
        context.system_properties["model_dir"],
        model_config.weights_file_name,
    )
    device = extract_device_from_context(context=context)
    model = torch.jit.load(weights_path, map_location=device)
    model = model.to(device)
    model.eval()
    return model


def extract_device_from_context(context: Context) -> torch.device:
    return torch.device(
        "cuda:" + str(context.system_properties.get("gpu_id"))
        if torch.cuda.is_available()
        else "cpu"
    )


def load_json_file(path: str) -> Union[dict, list, None]:
    with open(path) as f:
        return json.load(f)
