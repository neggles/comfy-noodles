import json
from datetime import UTC
from pathlib import Path

import torch
from comfy.comfy_types import FileLocator
from comfy.sd import VAE
from comfy.utils import load_torch_file, save_torch_file
from comfy_api.latest import LatentInput, io
from folder_paths import get_output_directory, get_save_image_path
from pydantic import BaseModel, ConfigDict, Field
from pydantic.types import AwareDatetime
from ulid import ULID

from noodles.utils import parse_ulid, prune_dict

from .common import LTXULID, BootstrapStrategy, MaskStrategy, WindowFunc, get_decay_curve

try:
    from safetensors import safe_open
except ImportError:
    safe_open = None

SEGMENT_METADATA_KEY = "ltx_i2v_segment"


def get_output_dir_path() -> Path:
    return Path(get_output_directory())


class LTXLat2VidSegmentData(BaseModel):
    video_id: ULID = Field(default_factory=ULID)
    segment_id: ULID = Field(default_factory=ULID)
    parent_id: ULID | None = None

    segment_idx: int = Field(default=0, min=0)
    iteration: int = Field(default=0, min=0)

    latent_shape: tuple[int, ...] = (0, 0, 0, 0, 0)
    num_latents: int = Field(..., min=1)

    start_frame: int = Field(
        min=0,
        description="Global frame index for the start of this segment. Used for video assembly and overlap calculations.",
    )
    end_frame: int = Field(
        min=0,
        description="Global frame index for the end of this segment. Used for video assembly and overlap calculations.",
    )
    num_frames: int = Field(
        ...,
        min=1,
        description="The number of frames generated for this run (including overlap and bootstrap).",
    )

    n_overlapped_latents: int = Field(1, min=1)
    discard_terminal_latent: bool = True

    mask_strat: MaskStrategy = MaskStrategy.NoStrategy
    bootstrap_strat: BootstrapStrategy = BootstrapStrategy.NoBootstrap

    video_name: str | None = None
    folder_name: str | None = None
    prompt: dict[str, str] = Field(default_factory=dict)
    extra_pnginfo: dict[str, str] = Field(default_factory=dict)

    created_at: AwareDatetime = Field(default_factory=lambda: AwareDatetime.now(tz=UTC))

    model_config: ConfigDict = ConfigDict(
        extra="allow",
    )


@io.comfytype(io_type="LTX_I2V_SEGMENT")
class LTXI2VSegmentInput(io.ComfyTypeIO):
    Type = LTXLat2VidSegmentData

    class Input(io.Input): ...

    class Output(io.Output): ...


def _resolve_segment_path(segment_path: str) -> Path:
    if not segment_path.strip():
        raise ValueError("segment_path is required")
    path = Path(segment_path).expanduser()
    if not path.is_absolute():
        path = get_output_dir_path() / path
    return path.resolve()


class LTXLat2VidSegmentSaveNood(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LTXLat2VidSegmentSaveNood",
            display_name="LTX Lat2Vid Segment Save",
            category="noodles/ltx",
            inputs=[
                io.Image.Input(
                    "images",
                    display_name="frames",
                    optional=True,
                    tooltip="Frames for this segment, including overlap. Overlap will be trimmed before passing to output.",
                ),
                io.String.Input(
                    "folder_name",
                    display_name="Folder Name",
                    default="ltx/{video_name}_{video_id}",
                    tooltip="Folder prefix for the saved segment data. Can include placeholders for video_name, "
                    + " and video_id. Segment index, segment ID, and iteration will be appended automatically.",
                ),
                io.Latent.Input("latent", tooltip="The latents for this segment"),
                LTXULID.Input(
                    "video_id",
                    optional=True,
                    default=None,
                    tooltip="Optional video ID. If not provided, a new one will be generated.",
                ),
                LTXULID.Input(
                    "parent_id",
                    optional=True,
                    default=None,
                    tooltip="Optional parent segment ID for hierarchical video structuring",
                ),
                io.String.Input(
                    "video_name",
                    display_name="Video Name",
                    default="untitled",
                    tooltip="Optional video name for organizational purposes. Not used for processing.",
                ),
                io.Int.Input(
                    "segment_idx",
                    display_name="Segment Index",
                    default=0,
                    min=0,
                    tooltip="Index of the segment within the video. Used for ordering segments during video assembly.",
                ),
                io.Int.Input(
                    "start_frame",
                    display_name="Start Frame",
                    default=0,
                    min=0,
                    tooltip="The starting frame index for this segment. Used for video assembly and overlap calculations.",
                ),
                io.Int.Input(
                    "num_frames",
                    display_name="Frame Count",
                    default=1,
                    min=1,
                    tooltip="The number of frames generated for this run (including overlap and bootstrap).",
                ),
                io.Int.Input(
                    "n_overlapped_latents",
                    display_name="Overlapped Latents",
                    default=1,
                    min=1,
                    tooltip="Overlap window in latents, will save this many plus one additional latent for overlap purposes.",
                ),
                io.Boolean.Input(
                    "discard_terminal_latent",
                    display_name="Discard Terminal Latent",
                    default=True,
                    tooltip="Whether to discard the final latent in the segment for overlap purposes. If true, the final latent will not be saved.",
                ),
                io.Combo.Input(
                    "mask_strat",
                    display_name="Mask Strategy",
                    options=MaskStrategy,
                    default=MaskStrategy.CosineDecayV1,
                ),
                io.Combo.Input(
                    "bootstrap_strat",
                    display_name="Bootstrap Strategy",
                    options=BootstrapStrategy,
                    default=BootstrapStrategy.DummyLatent,
                ),
            ],
            outputs=[
                io.String.Output(display_name="video_id"),
                io.String.Output(display_name="segment_id"),
                io.String.Output(display_name="segment_path"),
                io.String.Output(display_name="metadata_json"),
                io.String.Output(
                    display_name="video_prefix",
                    tooltip="File prefix for the saved segment. Feed to video save node if saving decoded video.",
                ),
                io.String.Output(
                    display_name="frames_prefix",
                    tooltip="Folder prefix for the saved segment. Feed to video save node if saving frames.",
                ),
                io.Image.Output(
                    display_name="frames",
                    tooltip="Frames for this segment, excluding overlap, for visualization purposes. Not used for processing.",
                ),
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(
        cls,
        images: torch.Tensor | None,  # [N, H, W, C]
        folder_name: str,
        latent: LatentInput,
        video_id: LTXULID | None,
        parent_id: LTXULID | None,
        video_name: str,
        segment_idx: int,
        start_frame: int,
        num_frames: int,
        n_overlapped_latents: int,
        discard_terminal_latent: bool,
        mask_strat: MaskStrategy,
        bootstrap_strat: BootstrapStrategy,
    ) -> io.NodeOutput:
        # Generate IDs where required.
        video_ulid = parse_ulid(video_id, "video_id", optional=True) or ULID()
        parent_ulid = parse_ulid(parent_id, "parent_id", optional=True)
        segment_id = ULID()

        # Construct folder prefix from template placeholders.
        try:
            folder_prefix = folder_name.format(
                video_name=video_name,
                video_id=str(video_ulid),
                segment_idx=segment_idx,
                segment_id=str(segment_id),
            )
        except KeyError as exc:
            raise ValueError(f"Unknown placeholder in folder_name template: {exc}") from exc
        filename_prefix = (
            f"{folder_prefix}/{video_name}.v{video_ulid}.s{segment_idx:03d}-{str(segment_id)[:12]}"
        )

        # Get output target path and create a unique safetensors filename.
        output_root = get_output_directory()
        out_folder, filename, counter, subfolder, _ = get_save_image_path(filename_prefix, output_root)
        # build filename and update filename_prefix
        filename = f"{filename}_{counter:05}"
        filename_prefix = f"{subfolder}/{filename}"
        # build final output path
        filename = f"{filename}.safetensors"
        output_path = Path(out_folder) / filename

        # get samples from latent (dont care about noise mask)
        samples: torch.Tensor = latent["samples"]  # shape [B, C, T, H, W]
        num_latents = samples.shape[2]

        # Work out which frames in the batch to extract, accounting for overlap and bootstrap latents
        img_start = 0 if bootstrap_strat == BootstrapStrategy.NoBootstrap else 1
        img_end = (num_latents - n_overlapped_latents) * 8
        if discard_terminal_latent:
            img_end -= 8

        # calculate global end frame
        end_frame = start_frame + (img_end - img_start)

        if images:
            save_images = images[img_start:img_end]
        else:
            save_images = torch.tensor([])

        # get "hidden" parameter values
        prompt_info = prune_dict(cls.hidden.prompt or {})
        extra_pnginfo = prune_dict(cls.hidden.extra_pnginfo or {})

        # build our metadata object
        metadata = LTXLat2VidSegmentData(
            video_id=video_ulid,
            parent_id=parent_ulid,
            segment_id=segment_id,
            segment_idx=segment_idx,
            latent_shape=tuple(samples.shape),
            num_latents=num_latents,
            start_frame=start_frame,
            end_frame=end_frame,
            num_frames=num_frames,
            n_overlapped_latents=n_overlapped_latents,
            discard_terminal_latent=discard_terminal_latent,
            mask_strat=mask_strat,
            bootstrap_strat=bootstrap_strat,
            video_name=video_name,
            prompt=prompt_info,
            extra_pnginfo=extra_pnginfo,
        )
        metadata_json = metadata.model_dump_json()

        output = {}
        output["latent_tensor"] = samples.cpu().contiguous()
        output["image_tensor"] = save_images.cpu().contiguous()
        output["ltx_l2v_format_v0"] = torch.tensor([])
        save_torch_file(
            output,
            output_path,
            metadata={SEGMENT_METADATA_KEY: metadata_json},
        )

        results: list[FileLocator] = []
        results.append({"filename": filename, "subfolder": subfolder, "type": "output"})

        return io.NodeOutput(
            str(video_ulid),
            str(segment_id),
            str(output_path),
            metadata_json,
            filename_prefix,
            filename_prefix + "/frame_",
            save_images,
            ui={"latents": results},
        )


class LTXLat2VidSegmentLoadNood(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LTXLat2VidSegmentLoadNood",
            display_name="LTX Lat2Vid Segment Load",
            category="noodles/ltx",
            inputs=[
                io.String.Input(
                    "folder_name",
                    default="ltx/{video_name}_{video_id}",
                    tooltip="Folder prefix to load segment data from",
                ),
            ],
            outputs=[
                io.Latent.Output(display_name="latent"),
                LTXULID.Output(display_name="video_id"),
                LTXULID.Output(display_name="segment_id"),
                io.String.Output(display_name="metadata_json"),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        return super().fingerprint_inputs(**kwargs)

    @classmethod
    def execute(
        cls,
        segment_path: str,
    ) -> io.NodeOutput:
        resolved_path = _resolve_segment_path(segment_path)
        if not resolved_path.exists():
            raise FileNotFoundError(f"Segment file does not exist: {resolved_path}")

        try:
            loaded = load_torch_file(str(resolved_path), return_metadata=True)
        except TypeError:
            # Older Comfy builds may not support return_metadata.
            raw_metadata: dict[str, str] | None = None
            if safe_open and resolved_path.suffix.lower() in {".safetensors", ".sft"}:
                with safe_open(str(resolved_path), framework="pt", device="cpu") as f:
                    raw_metadata = f.metadata()
            loaded = (load_torch_file(str(resolved_path)), raw_metadata)
        state_dict: dict[str, torch.Tensor]
        raw_metadata: dict[str, str] | None
        if isinstance(loaded, tuple):
            state_dict, raw_metadata = loaded
        else:
            state_dict, raw_metadata = loaded, None

        metadata_json = ""
        if raw_metadata:
            metadata_json = raw_metadata.get(SEGMENT_METADATA_KEY, "")
        if not metadata_json:
            raise ValueError(f"No '{SEGMENT_METADATA_KEY}' metadata found in {resolved_path}")

        segment_data = LTXLat2VidSegmentData.model_validate_json(metadata_json)

        if "samples" in state_dict:
            samples = state_dict["samples"]
        elif "latents" in state_dict:
            samples = state_dict["latents"]
        else:
            raise ValueError("Segment file is missing both 'samples' and 'latents' tensors")

        latent_output: dict[str, torch.Tensor] = {"samples": samples}
        if "noise_mask" in state_dict:
            latent_output["noise_mask"] = state_dict["noise_mask"]

        return io.NodeOutput(
            latent=latent_output,
            video_id=segment_data.video_id,
            segment_id=segment_data.segment_id,
            metadata_json=metadata_json,
        )


class LTXLat2VidInplaceNood(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LTXLat2VidInplaceNood",
            category="noodles/ltx",
            inputs=[
                io.Latent.Input("latent"),
                io.Latent.Input("last_segment"),
                io.Int.Input(
                    "num_latents",
                    default=1,
                    min=1,
                    max=1025,
                    step=8,
                    tooltip="Number of latents to encode and replace in the latent. Must be a multiple of the VAE time scale factor (e.g. 8).",
                ),
                io.Boolean.Input(
                    "discard_terminal_latent",
                    default=True,
                    tooltip="Whether to discard the final latent in the segment for overlap purposes. If true, the final latent will not be saved.",
                ),
                io.Combo.Input(
                    "mask_strat",
                    display_name="Mask Strategy",
                    options=MaskStrategy,
                    default=MaskStrategy.CosineDecayV1,
                ),
                io.Float.Input(
                    "strength_min",
                    default=0.3,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="Target strength for the final latent in the window when using a decay mask strategy. The first latent will always be at full strength, and the rest will decay towards this target strength based on the selected decay curve.",
                ),
                io.Combo.Input(
                    "bootstrap_strat",
                    display_name="Bootstrap Strategy",
                    options=BootstrapStrategy,
                    default=BootstrapStrategy.DummyLatent,
                ),
                io.Combo.Input(
                    "decay_curve",
                    options=WindowFunc,
                    default=WindowFunc.NoWindow,
                    tooltip="The curve/window function used to calculate strength decay across the overlap window",
                ),
                io.Int.Input(
                    "decay_start",
                    default=0,
                    min=0,
                    max=1024,
                    tooltip="Number of frames to keep at full strength before decay starts",
                ),
            ],
        )
