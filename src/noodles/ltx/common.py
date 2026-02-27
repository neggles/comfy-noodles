from enum import StrEnum

import numpy as np
from comfy_api.latest import io, ui
from ulid import ULID

from noodles.utils import parse_ulid


class MaskStrategy(StrEnum):
    NoStrategy = "none"
    CosineDecayV1 = "cosine_decay_v1"


class BootstrapStrategy(StrEnum):
    NoBootstrap = "none"
    DummyLatent = "dummy_latent"
    VAERoundtrip = "vae_roundtrip"


class WindowFunc(StrEnum):
    NoWindow = "none"
    CosineOut = "cosine_out"
    Smoothstep = "smoothstep"
    Smootherstep = "smootherstep"
    HalfGaussian = "half_gaussian"


def get_decay_curve(
    w_func: WindowFunc,
    steps: int = 1,
    start: int = 0,
    w_min: float = 0.3,
    sigma: float = 0.4,
) -> list[float]:
    if steps <= 1:
        return np.ones(steps, dtype=np.float32).tolist()

    if start > (steps - 1):
        raise ValueError("start must be less than steps.")

    t = np.linspace(0, 1, steps - start, dtype=np.float32)
    w = np.ones(steps, dtype=np.float32)
    # pick a window function, here's some i prepared earlier
    match w_func:
        case WindowFunc.NoWindow:
            w[start:] = t
        case WindowFunc.CosineOut:
            w[start:] = w_min + (1 - w_min) * np.cos(t * np.pi / 2) ** 2
        case WindowFunc.Smoothstep:
            w[start:] = 1 - (3 * t**2 - 2 * t**3)
        case WindowFunc.Smootherstep:
            w[start:] = 1 - (6 * t**5 - 15 * t**4 + 10 * t**3)
        case WindowFunc.HalfGaussian:
            w[start:] = np.exp(-(t**2) / (2 * sigma**2))
        case _:
            raise ValueError(f"Unknown window function: {w_func}")
    # force first frame to always be 1.0 to ensure the first frame is always fully weighted
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
