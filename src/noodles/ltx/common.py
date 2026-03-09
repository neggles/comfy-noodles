from enum import StrEnum
from math import ceil

import numpy as np
import torch
from comfy import model_management
from comfy_api.latest import io, ui
from pydantic import BaseModel, ConfigDict, Field
from ulid import ULID

from ..misc import AspectRatioOption
from ..utils import RoundingMode, ValidateAnyMixin, parse_ulid, round_to_multiple

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


class LTX2StageParamsNood(io.ComfyNode):
    """
    Convenience node to output basic parameters for LTX 2-stage latent-upscale video generation.
    Width, height, number of frames, frames per second as both float and int
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LTX2StageParamsNood",
            display_name="LTX 2-Stage Params",
            category="noodles/ltx",
            inputs=[
                io.DynamicCombo.Input(
                    id="res_mode",
                    options=[
                        io.DynamicCombo.Option(
                            "Aspect Ratio",
                            [
                                io.Combo.Input(
                                    "aspect_ratio",
                                    display_name="Aspect Ratio",
                                    options=AspectRatioOption,
                                    default=AspectRatioOption.Widescreen,
                                ),
                                io.Int.Input(
                                    "side_length",
                                    display_name="Side Length",
                                    default=1920,
                                    min=1280,
                                    max=5760,
                                    step=64,
                                    display_mode=io.NumberDisplay.slider,
                                ),
                            ],
                        ),
                        io.DynamicCombo.Option(
                            "Custom",
                            [
                                io.Int.Input("width", display_name="Width", default=1920, min=576, max=5760, step=64),
                                io.Int.Input("height", display_name="Height", default=1080, min=576, max=5760, step=64),
                            ],
                        ),
                    ],
                    display_name="Resolution Mode",
                ),
                io.Int.Input(
                    "n_frames",
                    display_name="Frames",
                    default=161,
                    min=1,
                    max=1025,
                    step=8,
                ),
                io.Float.Input(
                    "framerate",
                    display_name="FPS",
                    default=24.0,
                    min=1.0,
                    max=240.0,
                    step=1.0,
                ),
            ],
            outputs=[
                io.Latent.Output(display_name="latent", tooltip="Empty latent for first stage (half the size of the final output)"),
                io.Int.Output(display_name="s1_width", tooltip="Final width in pixels"),
                io.Int.Output(display_name="s1_height", tooltip="Final height in pixels"),
                io.Int.Output(display_name="s2_width", tooltip="Final width in pixels"),
                io.Int.Output(display_name="s2_height", tooltip="Final height in pixels"),
                io.Int.Output(display_name="n_frames", tooltip="Number of frames to generate."),
                io.Float.Output(display_name="fps", tooltip="Framerate as a float, in case you want 23.976 or similar. You monster."),
                io.Int.Output(display_name="fps_int", tooltip="Framerate as an integer, rounded up from the 'fps' output."),
                io.Float.Output(display_name="aspect", tooltip="Actual aspect ratio of the output video, as a float (width / height)."),
            ],
        )

    @classmethod
    def execute(
        cls,
        res_mode: dict[str, str | int],
        n_frames: int,
        framerate: float,
    ) -> io.NodeOutput:
        # round framerate to 3 decimal places on principle
        framerate = round(framerate, 3)

        # GET
        match res_mode["res_mode"]:
            case "Aspect Ratio":
                aspect_ratio = AspectRatioOption(res_mode.get("aspect_ratio"))
                side_length = int(res_mode.get("side_length", -1))
                side_length = round_to_multiple(side_length, 64, mode=RoundingMode.Ceil)
                # get width and height from aspect ratio
                width_px, height_px = aspect_ratio.get_width_height(side_length)
            case "Custom":
                width_px = int(res_mode.get("width", -1))
                height_px = int(res_mode.get("height", -1))
            case _:
                raise ValueError(f"Invalid resolution mode: {res_mode['res_mode']}")

        if width_px < 64 or height_px < 64:
            raise ValueError(f"Width and height must be at least 64 pixels. Got {width_px}x{height_px}.")

        # ensure width and height are divisible by 64, rounding up as needed
        width_px = round_to_multiple(width_px, 64, mode=RoundingMode.Ceil)
        height_px = round_to_multiple(height_px, 64, mode=RoundingMode.Ceil)

        # get stage2 latent size
        stage1_lat_w, stage1_lat_h = width_px // 64, height_px // 64

        # create stage1 empty latent
        stage1_latent = {
            "samples": torch.zeros(
                [1, 128, ((n_frames - 1) // 8) + 1, stage1_lat_h, stage1_lat_w],
                device=model_management.intermediate_device(),
            )
        }

        return io.NodeOutput(
            stage1_latent,
            width_px // 2,
            height_px // 2,
            width_px,
            height_px,
            n_frames,
            framerate,
            ceil(framerate),
            width_px / height_px,
        )
