from enum import StrEnum

import numpy as np
from comfy_api.latest import io, ui
from pydantic import BaseModel, ConfigDict, Field
from ulid import ULID

from noodles.utils import ValidateAnyMixin, parse_ulid

SEGMENT_METADATA_KEY = "ltx_l2v_segment"
# this will be generated once per extension load
_DEF_ULID_STR = str(ULID())


class BootstrapMode(StrEnum):
    SegmentZero = "segment_zero"
    DummyLatent = "dummy_latent"
    VAERoundtrip = "vae_roundtrip"
    RawLatent = "raw_latent"


class MaskStrategy(StrEnum):
    NoStrategy = "no_strategy"
    SolidMask = "solid_mask"
    CosineDecayV1 = "cosine_decay_v1"
    Smoothstep = "smoothstep"
    Smootherstep = "smootherstep"
    HalfGaussian = "half_gaussian"
    LinearDecay = "linear_decay"


class MaskParams(BaseModel, ValidateAnyMixin):
    strategy: MaskStrategy = Field(MaskStrategy.NoStrategy, description="The masking strategy to use")
    hard_mask_k: int = Field(2, ge=1, le=256, description="Number of latents to hard mask")
    w_max: float = Field(1.0, ge=0.0, le=1.0, description="Maximum weight for the mask")
    w_min: float = Field(0.25, ge=0.0, le=1.0, description="Minimum weight for the mask")
    decay_sigma: float = Field(0.8, ge=0.0, le=2.0, description="Sigma value for the decay function")

    model_config: ConfigDict = ConfigDict(
        extra="ignore",
    )


def get_mask_decay_curve(
    mask_strat: MaskStrategy,
    total_k: int = 1,
    hard_mask_k: int = 1,
    w_max: float = 1.0,
    w_min: float = 0.25,
    decay_sigma: float = 0.8,
) -> list[float]:
    if total_k <= 1:
        return np.ones(total_k, dtype=np.float32).tolist()

    if hard_mask_k > (total_k - 1):
        raise ValueError("hard_mask_k must be less than total_k.")

    t = np.linspace(0, 1, total_k - hard_mask_k + 1, dtype=np.float32)
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
        w[hard_mask_k - 1 :] = base
    else:
        w[hard_mask_k - 1 :] = w_min + (w_max - w_min) * base

    # if hard_mask_k=0 then w[0] may not be exactly 1.0, so force it to 1.0 to ensure model coherency
    w[0] = 1.0

    # clamp weights to 0.0 - 1.0 in case of rounding errors etc.
    w = np.clip(w, 0.0, 1.0)
    # return as a list (simpler and the performance overhead is irrelevant)
    return w.tolist()


### Miscellaneous extra nodes that are not specific to a single operating mode


@io.comfytype(io_type="ULID")
class ComfyULID(io.ComfyTypeIO):
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
            force_input: bool = True,
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

    # TODO: Work out how to make this a single-line read-only textbox. May take some custom JS,
    # since the single-row widget types are canvas-rendered rather than DOM elements.
    class Output(io.Output):
        pass


class ULIDPreviewNood(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ULIDPreviewNood",
            display_name="ULID Preview",
            category="noodles/misc",
            inputs=[
                io.MultiType.Input(
                    io.String.Input("ulid", display_name="ULID", force_input=True),
                    types=[io.String, ComfyULID, io.Int],
                    optional=True,
                    tooltip="A ULID string or ULID object to preview.",
                )
            ],
            outputs=[
                ComfyULID.Output(display_name="ULID"),
            ],
            is_output_node=True,
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


class ULIDFromStrNood(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ULIDFromStrNood",
            display_name="ULID From String",
            category="noodles/misc",
            inputs=[
                io.String.Input(
                    "ulid",
                    display_name="ULID",
                    optional=True,
                    default=_DEF_ULID_STR,
                    tooltip="Optional string to parse into a ComfyULID. If empty or not provided, a new one will be generated.",
                )
            ],
            outputs=[
                ComfyULID.Output(display_name="ULID"),
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
