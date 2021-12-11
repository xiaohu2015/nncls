from ..utils import Registry


MODELS = Registry("MODEL")
WEIGHTS = Registry("WEIGHT")


def get_model(model_name):
    return MODELS.get(model_name)
  

def get_weight(model_name, value_name):
    return WEIGHTS.get(f"{model_name}_weights")[value_name]
