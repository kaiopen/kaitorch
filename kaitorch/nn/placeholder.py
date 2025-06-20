from typing import Any

from torch import nn


Identity = nn.Identity


class ReturnFirst(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x: Any, *args, **kwargs) -> Any:
        return x
