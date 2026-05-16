import math
import torch
import torch.nn as nn
from .config import LoraConfig
from .utils import is_package_available
if is_package_available("bitsandbytes"):
    from bitsandbytes.nn import Linear4bit, Linear8bitLt
else:
    from .utils import Linear8bitLt, Linear4bit
from typing import Optional, Tuple


class LoraLinear(nn.Module):
    def __init__(
        self,
        base_layer: nn.Module,
        config: LoraConfig,
        weight: Tuple[Optional[torch.Tensor], Optional[torch.Tensor]] = (None, None),
        device: str = None,
    ):
        super().__init__()

        if not isinstance(base_layer, nn.Linear):
            assert isinstance(base_layer, Linear8bitLt) or isinstance(base_layer, Linear4bit), \
                f"Unsupported base layer type '{type(base_layer)}'."

        if isinstance(base_layer, Linear4bit) or isinstance(base_layer, Linear8bitLt):
            out_dim, in_dim = base_layer.out_features, base_layer.in_features
        else:
            out_dim, in_dim = base_layer.weight.shape

        # Use object.__setattr__ to avoid registering base_layer as a submodule,
        # which would create circular references and double-count params.
        object.__setattr__(self, "base_layer_", base_layer)

        # Fix: use next(parameters()) which works for both regular and quantised layers
        if device is not None:
            self.device_ = torch.device(device)
        else:
            try:
                self.device_ = next(base_layer.parameters()).device
            except StopIteration:
                self.device_ = torch.device("cpu")

        self.dtype_ = config.dtype_
        self.initializer_ = config.lora_init_
        self.r_ = config.lora_r_
        self.alpha_ = config.lora_alpha_
        self.scaling_ = self.alpha_ / self.r_

        self.in_features_ = in_dim
        self.out_features_ = out_dim

        assert config.lora_dropout_ >= 0.0
        self.dropout_ = nn.Dropout(p=config.lora_dropout_)

        self.lora_A = nn.Linear(
            self.in_features_,
            self.r_,
            bias=False,
            dtype=self.dtype_,
            device=self.device_,
        )
        self.lora_B = nn.Linear(
            self.r_,
            self.out_features_,
            bias=False,
            dtype=self.dtype_,
            device=self.device_,
        )

        self.reset_parameters(weight)

    def reset_parameters(
        self, weight: Tuple[Optional[torch.Tensor], Optional[torch.Tensor]] = (None, None)
    ) -> None:
        assert isinstance(weight, tuple) and len(weight) == 2
        assert (weight[0] is None and weight[1] is None) or (
            isinstance(weight[0], torch.Tensor) and isinstance(weight[1], torch.Tensor)
        )

        # Fix: use 'is None' instead of == (None, None) to avoid tensor comparison error
        if weight[0] is None:
            if self.initializer_ == "original":
                nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            elif self.initializer_ == "gaussian":
                nn.init.normal_(self.lora_A.weight, std=1 / self.r_)
            else:
                raise ValueError(f"Unknown initialization {self.initializer_}")
            nn.init.zeros_(self.lora_B.weight)
        else:
            with torch.no_grad():
                self.lora_A.weight.copy_(weight[0])
                self.lora_B.weight.copy_(weight[1])

    def lora_forward(self, residual: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        result_lora = self.lora_B(self.lora_A(self.dropout_(hidden_states.to(self.dtype_)))) * self.scaling_
        return residual + result_lora.to(residual.dtype)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = self.base_layer_(hidden_states)
        return self.lora_forward(residual, hidden_states)
