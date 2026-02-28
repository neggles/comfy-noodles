import json
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

import torch
from comfy.comfy_types import FileLocator
from comfy.utils import load_torch_file, save_torch_file
from comfy_api.latest import LatentInput, io
from pydantic import BaseModel, ConfigDict, Field, SerializeAsAny, computed_field
from pydantic.types import AwareDatetime
from ulid import ULID

from noodles.utils import get_next_file_idx, get_output_dir_path, parse_ulid, prune_dict

from .common import (
    LTXULID,
    BootstrapMode,
    MaskParams,
    MaskParamsIO,
    MaskStrategy,
    get_mask_decay_curve,
)

try:
    from safetensors import safe_open
except ImportError:
    safe_open = None

SEGMENT_METADATA_KEY = "ltx_l2v_segment"


class LTXLat2VidSegmentData(BaseModel):
    video_id: ULID = Field(default_factory=ULID)
    segment_id: ULID = Field(default_factory=ULID)
    parent_id: ULID | None = None

    segment_idx: int = Field(default=0, min=0, description="Index of the segment within the video.")
    iteration: int = Field(default=0, min=0, description="Iteration number for this segment.")

    n_latents_gen: int = Field(
        ...,
        min=2,
        description="Number of latents stored in the latent tensor, including bootstrap.",
    )

    start_frame: int = Field(
        ...,
        min=0,
        description="Global frame index for the start of this segment. Used for video assembly and overlap calculations.",
    )
    num_frames: int = Field(
        ...,
        min=1,
        description="Number of good frames in the frame tensor for this segment. Includes overlap at the start, "
        + "and the bootstrap frame for segment 0. start_frame + num_frames - overlap_frames = next segment's start_frame.",
    )
    keep_bootstrap: bool = Field(
        False,
        description="if True, the first (bootstrap) frame in the frames tensor will be kept during video assembly. "
        + "Generally you only want to keep the bootstrap frame from segment 0.",
    )

    overlap_k: int = Field(
        6,
        min=2,
        description="Number of latents at the end of the segment that will be overlapped with the next segment.",
    )
    mask_strat: MaskStrategy = MaskStrategy.SolidMask
    mask_params: MaskParams = Field(default_factory=MaskParams)
    bootstrap_mode: BootstrapMode = BootstrapMode.DummyLatent

    video_name: str = Field(default="untitled", description="Video name for organization purposes.")
    subfolder: str = Field(..., description="Subfolder that this segment was saved in originally.")
    width_px: int = Field(..., ge=256, le=4096, multiple_of=32, description="Width of video in pixels")
    height_px: int = Field(..., ge=256, le=4096, multiple_of=32, description="Height of video in pixels")

    prompt: SerializeAsAny[dict[str, Any]] = Field(default_factory=dict)
    extra_pnginfo: SerializeAsAny[dict[str, Any]] = Field(default_factory=dict)

    created_at: AwareDatetime = Field(default_factory=lambda: datetime.now(tz=UTC))

    model_config: ConfigDict = ConfigDict(
        extra="allow",
    )

    @computed_field
    @property
    def overlap_frames(self) -> int:
        """How many frames to subtract from num_frames to find the next segment's start frame, based on the number of overlapped latents."""
        return (self.overlap_k - 1) * 8


@io.comfytype(io_type="LTX_LAT")
class LTXLat2VidSegmentIO(io.ComfyTypeIO):
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


def _load_segment_file(segment_path: str) -> tuple[dict[str, torch.Tensor], str, Path]:
    resolved_path = _resolve_segment_path(segment_path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"Segment file does not exist: {resolved_path}")

    state_dict, metadata = load_torch_file(str(resolved_path), return_metadata=True)

    return state_dict, metadata, resolved_path


def _compute_overlap_strengths(
    total_k: int,
    bootstrap_mode: BootstrapMode,
    mask_strat: MaskStrategy,
    mask_params: MaskParams,
) -> list[float]:
    """
    Compute the strength values for each latent in the overlap region.
    """
    match bootstrap_mode:
        case BootstrapMode.SegmentZero:
            raise ValueError("Segment zero doesn't have strengths since it's a normal I2V segment.")
        case BootstrapMode.RawLatent:
            strength = [1.0] * total_k
        case BootstrapMode.DummyLatent | BootstrapMode.VAERoundtrip:
            strength = get_mask_decay_curve(
                mask_strat,
                total_k=total_k,
                **mask_params.model_dump(include=["hard_mask_k", "w_max", "w_min", "decay_sigma"]),
            )
        case _:
            raise ValueError(f"Unknown bootstrap mode: {bootstrap_mode}")


def _deterministic_seed(
    prev_segment_data: LTXLat2VidSegmentData,
    seed: int,
    overlap_count: int,
) -> int:
    seed_key = (
        f"{prev_segment_data.video_id}|{prev_segment_data.segment_id}|"
        f"{prev_segment_data.segment_idx}|{seed}|{overlap_count}"
    )
    digest = sha256(seed_key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def unpack_mask_strategy_input(mask_strat: dict) -> tuple[MaskStrategy, MaskParams]:
    """Unpack the MaskStrategy dynamiccombo input into the selected strategy
    and a MaskParams dataclass containing the relevant parameters for that strategy.
    """
    selected_strat: MaskStrategy = mask_strat["mask_strat"]
    if selected_strat not in MaskStrategy:
        raise ValueError(f"Unknown mask strategy: {selected_strat}")

    mask_params = MaskParams(**mask_strat)
    return selected_strat, mask_params


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
                    "subfolder",
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
                    tooltip="The number of frames generated for this run (including overlap, excluding bootstrap)",
                ),
                io.Int.Input(
                    "overlap_k",
                    display_name="Overlapped Latents",
                    default=6,
                    min=2,
                    tooltip="Number of latents at the end of the segment that will be overlapped with the next segment.",
                ),
                io.Boolean.Input(
                    "drop_last_latent",
                    display_name="Drop Last Latent",
                    default=True,
                    tooltip="Whether to drop the last latent before saving. The last latent is usually low-quality and best discarded.",
                ),
                MaskStrategy.Input("mask_strat", display_name="Mask Strategy"),
                BootstrapMode.Input("bootstrap_mode", display_name="Bootstrap Mode"),
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
        subfolder: str = ...,
        latent: LatentInput = ...,
        video_id: ULID | str | None = None,
        parent_id: ULID | str | None = None,
        video_name: str = ...,
        segment_idx: int = ...,
        start_frame: int = ...,
        num_frames: int = ...,
        overlap_k: int = ...,
        drop_last_latent: bool = ...,
        mask_strat: dict = ...,
        bootstrap_mode: BootstrapMode = ...,
    ) -> io.NodeOutput:

        # Generate IDs where required.
        video_ulid = parse_ulid(video_id, "video_id", optional=True) or ULID()
        parent_ulid = parse_ulid(parent_id, "parent_id", optional=True)
        segment_id = ULID()

        # Construct folder prefix from template placeholders.
        try:
            subfolder = subfolder.format(
                video_name=video_name,
                video_id=str(video_ulid),
                segment_idx=segment_idx,
                segment_id=str(segment_id),
            )
        except KeyError as exc:
            raise ValueError(f"Unknown placeholder in subfolder template: {exc}") from exc

        filename_prefix = f"{video_name}.v{str(video_ulid)[:10]}.s{segment_idx:03d}"

        # Get output target path and create a unique safetensors filename.
        output_root = get_output_dir_path()
        counter = get_next_file_idx(output_root / subfolder / f"{filename_prefix}.safetensors")

        # add counter and segment ID suffix
        segment_suffix = f"i{counter:03d}"

        output_dir = output_root.joinpath(subfolder)
        filename = f"{filename_prefix}_{segment_suffix}.safetensors"
        output_path = output_dir / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)

        frames_prefix = f"{subfolder}/{output_path.stem}/frame_"
        video_prefix = f"{subfolder}/{output_path.stem}/video_"

        # get samples from latent (dont care about noise mask)
        samples: torch.Tensor = latent["samples"]  # shape [B, C, T, H, W]
        num_latents = samples.shape[2]

        # Extract mask strategy and its various options
        mask_strat, mask_params = unpack_mask_strategy_input(mask_strat)

        # Work out which frames in the batch to extract, accounting for overlap and bootstrap latents
        match bootstrap_mode:
            case BootstrapMode.RawLatent:
                img_start = 0
            case BootstrapMode.DummyLatent | BootstrapMode.VAERoundtrip:
                img_start = 1
            case _:
                raise ValueError(f"Unknown bootstrap strategy: {bootstrap_mode}")

        # always pull every frame from segment 0
        if segment_idx == 0:
            img_start = 0
            img_end = (num_latents * 8) + 1

        img_end = (num_latents - overlap_k) * 8
        if drop_last_latent:
            img_end -= 8

        # calculate global end frame
        end_frame = start_frame + (img_end - img_start)
        # get the frames we're actually going to end up using
        save_images = images[img_start:img_end]
        # and the dimensions in pixels for metadata purposes
        width_px, height_px = save_images.shape[2], save_images.shape[1]

        # get "hidden" parameter values
        prompt_info = cls.hidden.prompt or {}
        extra_pnginfo = prune_dict(cls.hidden.extra_pnginfo or {})

        # build our metadata object
        metadata = LTXLat2VidSegmentData(
            video_id=video_ulid,
            parent_id=parent_ulid,
            segment_id=segment_id,
            segment_idx=segment_idx,
            iteration=counter,
            num_latents=num_latents,
            width_px=width_px,
            height_px=height_px,
            start_frame=start_frame,
            start_idx=img_start,
            num_frames=num_frames,
            overlap_k=overlap_k,
            mask_strat=mask_strat,
            mask_params=mask_params,
            bootstrap_mode=bootstrap_mode,
            video_name=video_name,
            subfolder=subfolder,
            prompt=prompt_info,
            extra_pnginfo=extra_pnginfo,
        )

        output = {}
        output["latent_tensor"] = samples.cpu().contiguous()
        output["image_tensor"] = save_images.cpu().contiguous()
        output["ltx_l2v_format_v0"] = torch.tensor([])
        save_torch_file(
            output,
            output_path,
            metadata={
                SEGMENT_METADATA_KEY: metadata.model_dump_json(exclude={"prompt", "extra_pnginfo"}),
                "prompt": json.dumps(prompt_info),
                "extra_pnginfo": json.dumps(extra_pnginfo),
            },
        )

        results: list[FileLocator] = []
        results.append({"filename": filename, "subfolder": subfolder, "type": "output"})

        return io.NodeOutput(
            str(video_ulid),
            str(segment_id),
            str(output_path),
            metadata.model_dump_json(),
            video_prefix,
            frames_prefix,
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
                io.Combo.Input(
                    "video_folder",
                    upload=io.UploadType.image,
                    image_folder=io.FolderType.output,
                    remote=io.RemoteOptions(
                        route="/internal/files/output",
                        refresh_button=True,
                        control_after_refresh="last",
                    ),
                )
            ],
            outputs=[
                io.Latent.Output(display_name="latent"),
                LTXLat2VidSegmentIO.Output(display_name="segment_data"),
                io.String.Output(display_name="metadata_json"),
                LTXULID.Input(
                    "video_id",
                    optional=True,
                    default=None,
                    tooltip="Video ID loaded from the segment file metadata.",
                ),
                LTXULID.Input(
                    "segment_id",
                    optional=True,
                    default=None,
                    tooltip="Segment ID loaded from the segment file metadata, for use as parent_id for next segment.",
                ),
                io.Int.Output(display_name="segment_idx"),
                io.Int.Output(display_name="next_segment_idx"),
                io.Int.Output(display_name="start_frame"),
                io.Int.Output(display_name="end_frame"),
                io.Int.Output(display_name="overlap_k"),
                io.Boolean.Output(display_name="drop_last_latent"),
                io.Combo.Output(display_name="bootstrap_mode", options=BootstrapMode),
                io.Combo.Output(display_name="mask_strat", options=MaskStrategy),
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
        state_dict, metadata_json, _ = _load_segment_file(segment_path)
        if "latent_tensor" not in state_dict:
            raise ValueError("Segment file is missing required 'latent_tensor' tensor")

        segment_data = LTXLat2VidSegmentData.model_validate_json(metadata_json)

        samples = state_dict["latent_tensor"]
        latent_output: dict[str, torch.Tensor] = {"samples": samples}
        if "noise_mask" in state_dict:
            latent_output["noise_mask"] = state_dict["noise_mask"]

        parent_id = str(segment_data.parent_id) if segment_data.parent_id else ""
        segment_idx = segment_data.segment_idx
        next_segment_idx = segment_idx + 1

        return io.NodeOutput(
            latent_output,
            segment_data,
            metadata_json,
            str(segment_data.video_id),
            str(segment_data.segment_id),
            parent_id,
            segment_idx,
            next_segment_idx,
            int(segment_data.start_frame),
            int(segment_data.end_frame),
            int(segment_data.overlap_k),
            bool(segment_data.drop_last_latent),
            str(segment_data.bootstrap_mode),
            str(segment_data.mask_strat),
        )


class LTXLat2VidPrepareSegmentNood(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LTXLat2VidPrepareSegmentNood",
            display_name="LTX Prepare Next Segment Latents",
            category="noodles/ltx",
            inputs=[
                io.Int.Input(
                    "noise_seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF, control_after_generate=True
                ),
                io.Latent.Input(
                    "latent", display_name="Latent", tooltip="Empty latent tensor for this segment."
                ),
                io.Latent.Input("prev_latent"),
                LTXLat2VidSegmentIO.Input("prev_segment_data"),
                io.Int.Input(
                    "overlap_k",
                    default=6,
                    min=1,
                    max=512,
                    tooltip="Number of overlapped latents to carry into the next segment.",
                ),
                io.Boolean.Input(
                    "drop_last_latent",
                    default=True,
                    tooltip="Drop the final latent from the previous segment before selecting overlap latents.",
                ),
                BootstrapMode.Input(
                    "bootstrap_mode",
                    display_name="Bootstrap Strategy",
                    default=BootstrapMode.DummyLatent,
                    tooltip="Strategy for generating the initial latent frame on segments after the first.",
                ),
                MaskStrategy.Input(
                    "mask_strat",
                    display_name="Mask Strategy",
                    options=MaskStrategy,
                ),
                MaskParamsIO.Input("mask_params", display_name="Mask Params"),
                io.Int.Input(
                    "iteration",
                    default=0,
                    min=0,
                    tooltip="Iteration index for next save; use 0 for a fresh segment branch.",
                ),
            ],
            outputs=[
                io.Latent.Output(display_name="next_latent"),
                io.String.Output(display_name="mask_strengths_json"),
                LTXLat2VidSegmentIO.Output(display_name="next_segment_data"),
                io.String.Output(display_name="next_segment_data_json"),
            ],
        )

    @classmethod
    def execute(
        cls,
        prev_latent: LatentInput,
        prev_segment_data: LTXLat2VidSegmentData,
        overlap_k: int,
        drop_last_latent: bool,
        bootstrap_mode: BootstrapMode,
        mask_strat: MaskStrategy,
        windomask_strat: MaskStrategy,
        mask_params: MaskParams,
        iteration: int,
    ) -> io.NodeOutput:
        prev_samples = prev_latent["samples"]
        if prev_samples.ndim != 5:
            raise ValueError(
                f"Expected prev_latent['samples'] to have shape [B,C,T,H,W], got {prev_samples.shape!r}"
            )

        usable_samples = prev_samples
        if drop_last_latent and prev_samples.shape[2] > 1:
            usable_samples = prev_samples[:, :, :-1, :, :]

        usable_latents = int(usable_samples.shape[2])
        if usable_latents < 1:
            raise ValueError("No usable latents left after applying drop_last_latent.")

        overlap_count = min(int(overlap_k), usable_latents)
        overlap = usable_samples[:, :, usable_latents - overlap_count :, :, :].clone()

        if bootstrap_mode == BootstrapMode.NoBootstrap:
            next_samples = overlap
        else:
            seed = _deterministic_seed(prev_segment_data, int(iteration), overlap_count)
            rng = torch.Generator(device=overlap.device.type)
            rng.manual_seed(seed)
            if overlap.shape[2] >= 1:
                latent1 = overlap[:, :, 0:1, :, :]
                if bootstrap_mode == BootstrapMode.VAERoundtrip:
                    bootstrap_latent = latent1.clone()
                else:
                    noise = torch.randn(
                        latent1.shape, dtype=latent1.dtype, device=latent1.device, generator=rng
                    )
                    bootstrap_latent = latent1 + float(noise_sigma) * noise
            else:
                bootstrap_latent = torch.randn(
                    usable_samples[:, :, 0:1, :, :].shape,
                    dtype=usable_samples.dtype,
                    device=usable_samples.device,
                    generator=rng,
                )
            next_samples = torch.cat([bootstrap_latent, overlap], dim=2)

        total_latents = int(next_samples.shape[2])
        strengths = _compute_overlap_strengths(
            total_latents=total_latents,
            overlap_k=overlap_count,
            hard_mask_k=mask_params.hard_mask_k,
            noise_sigma=noise_sigma,
            bootstrap_mode=bootstrap_mode,
            mask_strat=mask_strat,
            windomask_strat=windomask_strat,
        )

        strength_tensor = torch.tensor(strengths, dtype=next_samples.dtype, device=next_samples.device)
        strength_tensor = strength_tensor.view(1, 1, total_latents, 1, 1).repeat(
            next_samples.shape[0], 1, 1, 1, 1
        )
        noise_mask = 1.0 - strength_tensor

        next_num_frames = 1 + max(0, (total_latents - 1) * 8)
        next_start_frame = int(prev_segment_data.end_frame)
        next_end_frame = next_start_frame + next_num_frames
        next_segment_data = LTXLat2VidSegmentData(
            video_id=prev_segment_data.video_id,
            parent_id=prev_segment_data.segment_id,
            segment_id=ULID(),
            segment_idx=int(prev_segment_data.segment_idx) + 1,
            iteration=max(0, int(iteration)),
            latent_shape=tuple(next_samples.shape),
            num_latents=total_latents,
            start_frame=next_start_frame,
            end_frame=next_end_frame,
            num_frames=next_num_frames,
            overlap_k=overlap_count,
            drop_last_latent=drop_last_latent,
            mask_strat=mask_strat,
            bootstrap_mode=bootstrap_mode,
            video_name=prev_segment_data.video_name,
            subfolder=prev_segment_data.subfolder,
            prompt=dict(prev_segment_data.prompt),
            extra_pnginfo=dict(prev_segment_data.extra_pnginfo),
        )

        next_latent: dict[str, torch.Tensor] = {
            "samples": next_samples,
            "noise_mask": noise_mask,
        }
        return io.NodeOutput(
            next_latent,
            json.dumps(strengths),
            next_segment_data,
            next_segment_data.model_dump_json(),
        )
