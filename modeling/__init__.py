from modeling.base_model import BaseModel
from modeling.hf_model import HFModel

import pkgutil
import importlib
from pathlib import Path
import sys

from modeling.prompt_builder import build_prompt


MODEL_REGISTER = {}

def model(cls):
    module = sys.modules[cls.__module__]
    filename = Path(module.__file__).stem # type: ignore
    MODEL_REGISTER[filename] = cls
    return cls

for _, module_name, _ in pkgutil.iter_modules(__path__):
    importlib.import_module(f"{__name__}.{module_name}")


def load_model(model_name: str, **kwargs) -> BaseModel:
    model_class = MODEL_REGISTER.get(model_name, None)
    if model_class is None:
        raise ValueError(f"Model {model_name} not found in MODEL_REGISTER.")
    return model_class(model_name, **kwargs)
