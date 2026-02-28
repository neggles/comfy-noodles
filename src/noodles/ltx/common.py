from enum import StrEnum
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import torch
from comfy.sd import VAE
from comfy_api.latest import io, ui
from pydantic import BaseModel, Field
from ulid import ULID

from noodles.utils import parse_ulid


class BootstrapMode(StrEnum):
    SegmentZero = "segment_zero"
    DummyLatent = "dummy_latent"
    VAERoundtrip = "vae_roundtrip"
    RawLatent = "raw_latent"

    class Input(io.DynamicCombo.Input):
        def __init__(
            id: str,
            display_name: str = "Bootstrap Mode",
            optional: bool = False,
            tooltip: str = "Strategy for generating the initial latent frame on segments after the first.",
            lazy: bool = None,
            extra_dict=None,
        ):
            super().__init__(
                id,
                display_name=display_name,
                options=[
                    io.DynamicCombo.Option(
                        BootstrapMode.SegmentZero,
                        [
                            io.Vae.Input("vae", display_name="VAE"),
                            io.Image.Input("init_image", display_name="Initial Image"),
                        ],
                    ),
                    io.DynamicCombo.Option(
                        BootstrapMode.DummyLatent,
                        [
                            io.Float.Input(
                                "noise_sigma",
                                display_name="Noise Sigma",
                                default=0.5,
                                min=0.0,
                                max=1.0,
                                step=0.01,
                            )
                        ],
                    ),
                    io.DynamicCombo.Option(
                        BootstrapMode.VAERoundtrip,
                        [io.Vae.Input("vae", display_name="VAE")],
                    ),
                    io.DynamicCombo.Option(
                        BootstrapMode.RawLatent,
                        [],
                    ),
                ],
                optional=optional,
                tooltip=tooltip,
                lazy=lazy,
                extra_dict=extra_dict,
            )

    def unpack_input(
        self, values: dict
    ) -> tuple["BootstrapMode", dict[str, torch.Tensor, VAE | float | None]]:
        strat = values["strat_name"]
        match strat:
            case BootstrapMode.SegmentZero:
                return BootstrapMode(strat), {"init_image": values["init_image"]}
            case BootstrapMode.DummyLatent:
                return BootstrapMode(strat), {"noise_sigma": values["noise_sigma"]}
            case BootstrapMode.VAERoundtrip:
                return BootstrapMode(strat), {"vae": values["vae"]}
            case BootstrapMode.RawLatent:
                return BootstrapMode(strat), {}
            case _:
                raise ValueError(f"Unknown bootstrap strategy: {strat}")

    class Output(io.Combo.Output):
        def __init__(
            self,
            id: str = None,
            display_name: str = None,
            tooltip: str = "Strategy for generating the initial latent frame on segments after the first.",
            **kwargs,
        ):
            super().__init__(
                id,
                display_name=display_name,
                options=BootstrapMode,
                tooltip=tooltip,
                **kwargs,
            )


class MaskParams(BaseModel):
    hard_mask_k: int | None = Field(2, ge=0, le=256)
    w_max: float | None = Field(1.0, ge=0.0, le=1.0)
    w_min: float | None = Field(0.1, ge=0.0, le=1.0)
    decay_sigma: float | None = Field(0.4, ge=0.0, le=1.0)

    HARD_MASK_K_IN: ClassVar[io.Int.Input] = io.Int.Input(
        "hard_mask_k",
        default=2,
        min=0,
        max=256,
        tooltip="Number of overlapped latents to hard-mask at strength 1.0 before decay.",
    )
    W_MAX_IN: ClassVar[io.Float.Input] = io.Float.Input(
        "w_max",
        default=1.0,
        min=0.0,
        max=1.0,
        step=0.01,
        tooltip="Maximum strength for the final latent in the overlap window.",
    )
    W_MIN_IN: ClassVar[io.Float.Input] = io.Float.Input(
        "w_min",
        default=0.1,
        min=0.0,
        max=1.0,
        step=0.01,
        tooltip="Minimum strength for the final latent in the overlap window.",
    )
    SIGMA_IN: ClassVar[io.Float.Input] = io.Float.Input(
        "decay_sigma",
        default=0.4,
        min=0.0,
        max=1.0,
        step=0.01,
        tooltip="Sigma for half-Gaussian decay curve.",
    )


@io.comfytype(io_type="NOODLES_LTX_MASK_PARAMS")
class MaskParamsIO(io.ComfyTypeIO):
    Type = MaskParams

    class Input(io.WidgetInput):
        """Mask strategy parameters input."""

        def __init__(
            self,
            id: str,
            display_name: str = None,
            optional: bool = False,
            tooltip: str = None,
            lazy: bool = None,
            default: MaskParams | dict | str | None = None,
            force_input: bool = None,
            extra_dict: dict | None = None,
            raw_link: bool = None,
            advanced: bool = None,
        ):
            if isinstance(default, dict):
                default = MaskParams.model_validate(default, strict=False)
            if isinstance(default, str):
                default = MaskParams.model_validate_json(default, strict=False)

            super().__init__(
                id,
                display_name,
                optional,
                tooltip,
                lazy,
                default,
                False,
                None,
                force_input,
                extra_dict,
                raw_link,
                advanced,
            )
            self.default: MaskParams

        def as_dict(self):
            return super().as_dict()


class MaskStrategy(StrEnum):
    SolidMask = "no_strategy"
    LinearDecay = "linear_decay"
    CosineDecayV1 = "cosine_decay_v1"
    Smoothstep = "smoothstep"
    Smootherstep = "smootherstep"
    HalfGaussian = "half_gaussian"

    class Input(io.DynamicCombo.Input):
        def __init__(
            self,
            id="mask_strat",
            display_name: str = "Mask Strategy",
            optional: bool = False,
            tooltip: str = None,
            lazy: bool = None,
            extra_dict: dict | None = None,
        ):

            super().__init__(
                id,
                display_name,
                options=[
                    io.DynamicCombo.Option(
                        "Load Params",
                        [
                            io.Combo.Input("strat_name", options=MaskStrategy, display_name="Strategy Type"),
                            MaskParamsIO.Input(
                                "mask_params", display_name="Mask Params", tooltip="Mask parameters"
                            ),
                        ],
                    ),
                    io.DynamicCombo.Option(
                        MaskStrategy.SolidMask,
                        [MaskParams.HARD_MASK_K_IN, MaskParams.W_MAX_IN],
                    ),
                    io.DynamicCombo.Option(
                        MaskStrategy.LinearDecay,
                        [MaskParams.HARD_MASK_K_IN, MaskParams.W_MAX_IN, MaskParams.W_MIN_IN],
                    ),
                    io.DynamicCombo.Option(
                        MaskStrategy.CosineDecayV1,
                        [MaskParams.HARD_MASK_K_IN, MaskParams.W_MAX_IN, MaskParams.W_MIN_IN],
                    ),
                    io.DynamicCombo.Option(
                        MaskStrategy.Smoothstep,
                        [MaskParams.HARD_MASK_K_IN, MaskParams.W_MAX_IN, MaskParams.W_MIN_IN],
                    ),
                    io.DynamicCombo.Option(
                        MaskStrategy.Smootherstep,
                        [MaskParams.HARD_MASK_K_IN, MaskParams.W_MAX_IN, MaskParams.W_MIN_IN],
                    ),
                    io.DynamicCombo.Option(
                        MaskStrategy.HalfGaussian,
                        [
                            MaskParams.HARD_MASK_K_IN,
                            MaskParams.W_MAX_IN,
                            MaskParams.W_MIN_IN,
                            MaskParams.SIGMA_IN,
                        ],
                    ),
                ],
                optional=optional,
                tooltip=tooltip,
                lazy=lazy,
                extra_dict=extra_dict,
            )

    class Output(io.DynamicCombo.Output):
        def __init__(
            self,
            id="mask_strat",
            display_name: str = "Mask Strategy",
            tooltip: str = None,
            extra_dict: dict | None = None,
        ):
            super().__init__(
                id,
                display_name,
                options=MaskStrategy.Input.options,
                tooltip=tooltip,
                extra_dict=extra_dict,
            )


def get_mask_decay_curve(
    mask_strat: MaskStrategy,
    total_k: int = 1,
    hard_mask_k: int = 0,
    w_max: float = 1.0,
    w_min: float = 0.1,
    decay_sigma: float = 0.4,
) -> list[float]:
    if total_k <= 1:
        return np.ones(total_k, dtype=np.float32).tolist()

    if hard_mask_k > (total_k - 1):
        raise ValueError("hard_mask_k must be less than total_k.")

    t = np.linspace(0, 1, total_k - hard_mask_k, dtype=np.float32)
    w = np.ones(total_k, dtype=np.float32)

    # pick a window function, here's some i prepared earlier
    match mask_strat:
        case MaskStrategy.SolidMask:
            base = np.full_like(t, w_max)
        case MaskStrategy.LinearDecay:
            base = 1.0 - t
        case MaskStrategy.CosineDecayV1:
            base = np.cos(t * np.pi / 2.0) ** 2
        case MaskStrategy.Smoothstep:
            base = 1.0 - (3.0 * t**2 - 2.0 * t**3)
        case MaskStrategy.Smootherstep:
            base = 1.0 - (6.0 * t**5 - 15.0 * t**4 + 10.0 * t**3)
        case MaskStrategy.HalfGaussian:
            # normalized so base(0)=1, base(1)=0
            raw = np.exp(-(t**2) / (2.0 * decay_sigma**2))
            raw_end = np.exp(-(1.0**2) / (2.0 * decay_sigma**2))
            base = (raw - raw_end) / (1.0 - raw_end)
        case _:
            raise ValueError(f"Unknown window function: {mask_strat}")

    # apply scaling to w_min and w_max
    if mask_strat == MaskStrategy.SolidMask:
        w[hard_mask_k:] = base
    else:
        w[hard_mask_k:] = w_min + (w_max - w_min) * base

    # if hard_mask_k=0 then w[0] may not be exactly 1.0, so force it to 1.0 to ensure model coherency
    w[0] = 1.0

    # clamp weights to 0.0 - 1.0 in case of rounding errors etc.
    w = np.clip(w, 0.0, 1.0)
    # return as a list (simpler and the performance overhead is irrelevant)
    return w.tolist()


@io.comfytype(io_type="ULID")
class LTXULID(io.ComfyTypeIO):
    Type = ULID

    class Input(io.WidgetInput):
        """ULID input."""

        def __init__(
            self,
            id: str,
            display_name: str = None,
            optional: bool = True,
            tooltip: str = None,
            lazy: bool = None,
            default: str | ULID = None,
            force_input: bool = None,
            extra_dict: dict | None = None,
            raw_link: bool = None,
            advanced: bool = None,
        ):
            if isinstance(default, str):
                default = parse_ulid(default, optional=True)

            super().__init__(
                id,
                display_name,
                optional,
                tooltip,
                lazy,
                default,
                False,
                None,
                force_input,
                extra_dict,
                raw_link,
                advanced,
            )
            self.default: ULID

        def as_dict(self):
            return super().as_dict()


class LTXULIDPreviewNood(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LTXULIDPreviewNood",
            display_name="LTX ULID Preview",
            category="noodles/ltx",
            inputs=[
                io.MultiType.Input(
                    io.String.Input("ulid", display_name="ULID", force_input=True),
                    types=[io.String, LTXULID, io.Int],
                    optional=True,
                    tooltip="A ULID string or ULID object to preview.",
                )
            ],
            outputs=[
                LTXULID.Output(display_name="ULID"),
            ],
            is_output_node=True,
            enable_expand=True,
        )

    @classmethod
    def execute(cls, ulid: str | int | ULID):
        if not isinstance(ulid, ULID):
            ulid = parse_ulid(ulid, optional=True)
        if not ulid:
            ulid = ULID()

        ulid_str = str(ulid)

        return io.NodeOutput(
            ulid,
            ui=ui.PreviewText(ulid_str),
        )


class LTXULIDFromStrNood(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LTXULIDFromStrNood",
            display_name="LTX ULID From String",
            category="noodles/ltx",
            inputs=[
                io.String.Input(
                    "ulid",
                    display_name="ULID",
                    optional=True,
                    tooltip="Optional string to parse into a ULID. If empty or not provided, a new one will be generated.",
                )
            ],
            outputs=[
                LTXULID.Output(display_name="ULID"),
            ],
        )

    @classmethod
    def execute(cls, ulid: str | None):
        if ulid:
            ulid = parse_ulid(ulid, optional=False)
        else:
            ulid = ULID()

        return io.NodeOutput(
            ulid,
            ui=ui.PreviewText(f"`{str(ulid)}`"),
        )
