from mamba_ssm import Mamba2 as _Mamba2


class Mamba2(_Mamba2):
    def __init__(
        self,
        in_channels: int,
        state_channels: int = 64,
        expansion: int = 2,
        kernel_size: int = 4,
        *args, **kwargs
    ) -> None:
        super().__init__(
            d_model=in_channels,
            d_state=state_channels,
            d_conv=kernel_size,
            expand=expansion
        )
