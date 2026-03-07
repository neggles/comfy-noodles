from comfy_api.latest import io

from noodles.utils import prune_dict

from .common import BootstrapMode, MaskParams, MaskStrategy


@io.comfytype(io_type="COMBO")
class BootstrapModeIO(io.Combo):
    Type = str

    class Input(io.Combo.Input):
        def __init__(
            self,
            id: str,
            display_name: str = "Bootstrap Mode",
            optional: bool = False,
            tooltip: str = "Mode for bootstrapping the latent video segment.",
            default: BootstrapMode | str = None,
            **kwargs,
        ):
            if isinstance(default, str):
                default = BootstrapMode(default)

            super().__init__(
                id,
                options=BootstrapMode,
                display_name=display_name,
                optional=optional,
                tooltip=tooltip,
                default=default,
                **kwargs,
            )
            self.default: BootstrapMode

    class Output(io.Combo.Output):
        pass


@io.comfytype(io_type="COMBO")
class MaskStrategyIO(io.ComfyTypeIO):
    Type = MaskStrategy

    class Input(io.Combo.Input):
        def __init__(
            self,
            id: str,
            display_name: str = "Mask Strategy",
            optional: bool = False,
            tooltip: str = "Strategy for generating the mask weights for overlapping latents.",
            default: MaskStrategy | str | None = None,
            **kwargs,
        ):
            if isinstance(default, str):
                default = MaskStrategy(default)

            super().__init__(
                id,
                options=MaskStrategy,
                display_name=display_name,
                optional=optional,
                tooltip=tooltip,
                default=default,
                **kwargs,
            )
            self.default: MaskStrategy

    class Output(io.Combo.Output):
        pass


@io.comfytype(io_type="MASK_PARAMS")
class MaskParamsIO(io.ComfyTypeIO):
    type = MaskParams

    class Input(io.Input):
        def __init__(
            self,
            id: str,
            display_name: str = None,
            optional=False,
            tooltip: str = None,
            lazy: bool = None,
            default: MaskParams | dict | None = None,
            extra_dict=None,
            raw_link: bool = None,
            advanced: bool = None,
        ):
            super().__init__(
                id,
                display_name=display_name,
                optional=optional,
                tooltip=tooltip,
                lazy=lazy,
                extra_dict=extra_dict,
                raw_link=raw_link,
                advanced=advanced,
            )
            self.default = MaskParams.model_validate_any(default) if default is not None else None

        def as_dict(self):
            return super().as_dict() | prune_dict({"default": self.default})

    class Output(io.Output):
        pass
