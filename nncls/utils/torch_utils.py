from typing import no_type_check
import torch

try:
    from torch.hub import load_state_dict_from_url  # noqa: 401
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url  # noqa: 401

    
@no_type_check
def _log_api_usage_once(module: str, name: str) -> None:
    if torch.jit.is_scripting() or torch.jit.is_tracing():
        return
    torch._C._log_api_usage_once(f"nncls.{module}.{name}")
