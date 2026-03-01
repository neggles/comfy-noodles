import json
import re
from datetime import UTC, datetime
from hashlib import sha256
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from comfy.comfy_types import FileLocator
from comfy.sd import VAE
from comfy.utils import ProgressBar, load_torch_file, save_torch_file
from comfy_api.latest import LatentInput, io, ui
from comfy_extras.nodes_lt import get_noise_mask
from pydantic import BaseModel, ConfigDict, Field, SerializeAsAny
from pydantic.types import AwareDatetime
from ulid import ULID

from noodles.utils import (
    compress_image_tensor_webp,
    decompress_image_tensor_webp,
    get_folders_in_outdir,
    get_output_dir_path,
    parse_ulid,
    prune_dict,
)

from .common import (
    BootstrapMode,
    ComfyULID,
    MaskParams,
    MaskStrategy,
    get_mask_decay_curve,
    get_next_segment_iteration,
    parse_segment_name,
)
from .io import (
    BootstrapModeIO,
    MaskParamsIO,
    MaskStrategyIO,
)

SEGMENT_METADATA_KEY = "ltx_l2v_segment"


class LTXLat2VidSegmentData(BaseModel):
    video_id: ULID = Field(default_factory=ULID)
    parent_id: ULID | None = None
    segment_id: ULID = Field(default_factory=ULID)

    segment_idx: int = Field(default=0, min=0, description="Index of the segment within the video.")
    iteration: int = Field(default=0, min=0, description="Iteration number for this segment.")

    start_frame: int = Field(
        ...,
        min=0,
        description="Global frame index for the start of this segment. Used for video assembly and overlap calculations.",
    )
    n_frames: int = Field(
        ...,
        min=1,
        description="Number of good frames in the frame tensor for this segment. Includes overlap at the start, "
        + "and the bootstrap frame for segment 0. start_frame + n_frames - overlap_frames = next segment's start_frame.",
    )
    n_frames_batch: int = Field(
        ...,
        min=1,
        description="Number of frames generated for this segment, including overlap and bootstrap frame. Used to make future segments the same length.",
    )

    n_latents: int = Field(
        ...,
        min=2,
        description="Number of latents stored in the latent tensor, inc. bootstrap, excl. discarded terminal (if enabled)",
    )
    overlap_k: int = Field(
        6,
        ge=2,
        le=512,
        description="Number of latents at the end of the segment that will be overlapped with the next segment.",
    )
    keep_bootstrap: bool = Field(
        False,
        description="if True, the first (bootstrap) frame in the frames tensor will be kept during video assembly. "
        + "Generally you only want to keep the bootstrap frame from segment 0.",
    )
    drop_last_latent: bool = Field(
        True,
        description="Whether to drop the last latent before saving. The last latent is usually low-quality and best discarded.",
    )

    bootstrap_mode: BootstrapMode = BootstrapMode.DummyLatent
    mask_strat: MaskStrategy = MaskStrategy.SolidMask
    mask_params: MaskParams = Field(default_factory=MaskParams)

    video_name: str = Field(default="untitled", description="Video name for organization purposes.")
    subfolder: str = Field(..., description="Subfolder that this segment was saved in originally.")
    width_px: int = Field(..., ge=256, le=4096, multiple_of=32, description="Width of video in pixels")
    height_px: int = Field(..., ge=256, le=4096, multiple_of=32, description="Height of video in pixels")

    prompt: SerializeAsAny[dict[str, Any]] = Field(default_factory=dict)
    extra_pnginfo: SerializeAsAny[dict[str, Any]] = Field(default_factory=dict)

    created_at: AwareDatetime = Field(default_factory=lambda: datetime.now(tz=UTC))

    model_config: ConfigDict = ConfigDict(
        extra="ignore",
    )

    @classmethod
    def model_validate_any(
        cls, value: Any, strict: bool = False, extra: bool = True, **kwargs
    ) -> "LTXLat2VidSegmentData":
        match value:
            case cls():
                return value
            case str() if value.strip():
                return cls.model_validate_json(value, strict=strict, extra=extra, **kwargs)
            case dict():
                return cls.model_validate(value, strict=strict, extra=extra, **kwargs)
            case _:
                raise TypeError(f"Unsupported segment metadata type: {type(value)!r}")


@io.comfytype(io_type="LAT2VID_SEGMENT")
class LTXLat2VidSegmentIO(io.ComfyTypeIO):
    if TYPE_CHECKING:
        Type = LTXLat2VidSegmentData

    class Input(io.Input):
        pass

    class Output(io.Output):
        pass


def _segment_data_from_headers(raw_metadata: dict[str, str]) -> tuple[LTXLat2VidSegmentData, str]:
    metadata_json = raw_metadata.get(SEGMENT_METADATA_KEY)
    if not metadata_json:
        raise ValueError(f"No '{SEGMENT_METADATA_KEY}' metadata found in checkpoint")

    segment_data = LTXLat2VidSegmentData.model_validate_json(metadata_json)

    update = {}
    for key in ["prompt", "extra_pnginfo"]:
        if key in raw_metadata and raw_metadata[key]:
            update[key] = raw_metadata[key]
    if update:
        segment_data = segment_data.model_copy(update=update)

    return segment_data, segment_data.model_dump_json()


def load_segment_file(
    segment_path: PathLike[str],
) -> tuple[dict[str, torch.Tensor], LTXLat2VidSegmentData, str, Path]:
    segment_path = Path(segment_path).expanduser()
    if not segment_path.is_absolute():
        segment_path = get_output_dir_path() / segment_path
    resolved_path = segment_path.resolve()

    if not resolved_path.exists():
        raise FileNotFoundError(f"Segment file does not exist: {resolved_path}")

    state_dict, raw_metadata = load_torch_file(str(resolved_path), return_metadata=True)
    segment_data, metadata_json = _segment_data_from_headers(raw_metadata)
    return state_dict, segment_data, metadata_json, resolved_path


def _find_segment_file(video_folder: str, segment_idx: int, iteration: int) -> Path:
    output_root = get_output_dir_path()
    folder_path = output_root / video_folder
    if not folder_path.exists() or not folder_path.is_dir():
        raise FileNotFoundError(f"Subfolder does not exist: {folder_path}")

    candidates: list[tuple[int, Path]] = []
    for path in folder_path.glob("*.safetensors"):
        ids = parse_segment_name(path)
        if not ids:
            continue
        seg_idx, seg_iter = ids
        if seg_idx == segment_idx:
            candidates.append((seg_iter, path))

    if not candidates:
        raise FileNotFoundError(f"No segment files found for segment_idx={segment_idx} in {folder_path}")

    candidates.sort(key=lambda item: item[0])

    if iteration <= 0:
        return candidates[-1][1]

    for seg_iter, path in candidates:
        if seg_iter == iteration:
            return path

    available = ", ".join(str(seg_iter) for seg_iter, _ in candidates)
    raise FileNotFoundError(
        f"No segment file found for segment_idx={segment_idx}, iteration={iteration}. Available iterations: {available}"
    )


def _compute_overlap_strengths(
    total_k: int,
    bootstrap_mode: BootstrapMode,
    bootstrap_strength: float | None = None,
    mask_strat: MaskStrategy = ...,
    mask_params: MaskParams = ...,
) -> list[float]:
    if total_k < 2:
        raise ValueError("total_k must be at least 2")

    match bootstrap_mode:
        case BootstrapMode.SegmentZero:
            raise ValueError("SegmentZero bootstrap mode is invalid for continuation segments")
        case BootstrapMode.RawLatent:
            bootstrap_strength = 1.0
        case BootstrapMode.DummyLatent:
            # For Dummy Latent continuation, bootstrap latent should be freely diffused (strength 0.0)
            bootstrap_strength = 0.0
        case BootstrapMode.VAERoundtrip:
            # For VAE Roundtrip, the bootstrap latent is a VAE roundtrip of the last frame and may be somewhat noisy/artifacted,
            # so we give it a moderate strength to allow some correction while still being anchored to the source.
            bootstrap_strength = bootstrap_strength if bootstrap_strength is not None else 0.5

    # and overlap latents should be masked according to selected strategy.
    overlap_strengths = get_mask_decay_curve(
        mask_strat,
        total_k=total_k - 1,
        **mask_params.model_dump(include={"hard_mask_k", "w_max", "w_min", "decay_sigma"}),
    )
    return [bootstrap_strength, *overlap_strengths]


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
                    optional=False,
                    tooltip="Frames for this segment, including overlap. Overlap will be trimmed before saving.",
                ),
                io.String.Input(
                    "subfolder",
                    display_name="Folder Name",
                    multiline=False,
                    default="ltx/{video_name}_{video_id}",
                    tooltip="Folder prefix for the saved segment data. Can include placeholders for video_name, "
                    + " and video_id. Segment index, segment ID, and iteration will be appended automatically.",
                ),
                io.Latent.Input("latent", tooltip="The latents for this segment"),
                ComfyULID.Input(
                    "video_id",
                    optional=True,
                    default=None,
                    force_input=True,
                    tooltip="Optional video ID. If not provided, a new one will be generated.",
                ),
                ComfyULID.Input(
                    "parent_id",
                    optional=True,
                    default=None,
                    force_input=True,
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
                    "n_frames_batch",
                    display_name="Batch Frames",
                    default=1,
                    min=1,
                    tooltip="The number of frames generated for this run (including overlap and bootstrap)",
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
                    display_name="Last Latent",
                    default=True,
                    label_on="Drop",
                    label_off="Keep",
                    tooltip="Whether to drop the last latent before saving. The last latent is usually low-quality and best discarded.",
                ),
                BootstrapModeIO.Input("bootstrap_mode", default=BootstrapMode.DummyLatent),
                MaskStrategyIO.Input("mask_strat", default=MaskStrategy.CosineDecayV1),
                MaskParamsIO.Input("mask_params"),
            ],
            outputs=[
                ComfyULID.Output(display_name="video_id"),
                ComfyULID.Output(display_name="segment_id"),
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
        images: torch.Tensor,
        subfolder: str,
        latent: LatentInput,
        video_id: ULID | str | None,
        parent_id: ULID | str | None,
        video_name: str,
        segment_idx: int,
        start_frame: int,
        n_frames_batch: int,
        overlap_k: int,
        drop_last_latent: bool,
        bootstrap_mode: BootstrapMode,
        mask_strat: MaskStrategy,
        mask_params: MaskParams,
    ) -> io.NodeOutput:
        if not isinstance(mask_params, MaskParams):
            mask_params = MaskParams.model_validate(mask_params, extra="ignore", strict=False)
        if not isinstance(bootstrap_mode, BootstrapMode):
            bootstrap_mode = BootstrapMode(bootstrap_mode)
        if not isinstance(mask_strat, MaskStrategy):
            mask_strat = MaskStrategy(mask_strat)

        video_ulid = parse_ulid(video_id, "video_id", optional=True) or ULID()
        parent_ulid = parse_ulid(parent_id, "parent_id", optional=True)
        segment_id = ULID()

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
        output_root = get_output_dir_path()
        counter = get_next_segment_iteration(output_root / subfolder / f"{filename_prefix}.safetensors")

        filename = f"{filename_prefix}_i{counter:03d}.safetensors"
        output_dir = output_root / subfolder
        output_path = output_dir / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)

        frames_prefix = f"{subfolder}/{output_path.stem}/frame_"
        video_prefix = f"{subfolder}/{output_path.stem}/video_"

        samples: torch.Tensor = latent["samples"]
        if samples.ndim != 5:
            raise ValueError(f"Expected latent['samples'] with shape [B,C,T,H,W], got {samples.shape!r}")

        if drop_last_latent:
            if samples.shape[2] <= 1:
                raise ValueError("Cannot drop terminal latent when only one latent is present")
            samples = samples[:, :, :-1, :, :]
            images = images[:-8]

        if segment_idx == 0:
            bootstrap_frame = images[:1]
            save_images = images
        else:
            bootstrap_frame = images[:1]
            save_images = images[1:]

        width_px = int(save_images.shape[2])
        height_px = int(save_images.shape[1])

        prompt_info = prune_dict(cls.hidden.prompt or {})
        extra_pnginfo = prune_dict(cls.hidden.extra_pnginfo or {})

        n_latents = int(samples.shape[2])
        n_frames = int(save_images.shape[0])

        metadata = LTXLat2VidSegmentData(
            video_id=video_ulid,
            parent_id=parent_ulid,
            segment_id=segment_id,
            segment_idx=segment_idx,
            iteration=counter,
            start_frame=start_frame,
            n_frames=n_frames,
            n_frames_batch=n_frames_batch,
            n_latents=n_latents,
            overlap_k=overlap_k,
            keep_bootstrap=segment_idx == 0,
            drop_last_latent=drop_last_latent,
            bootstrap_mode=bootstrap_mode,
            mask_strat=mask_strat,
            mask_params=mask_params,
            video_name=video_name,
            subfolder=subfolder,
            width_px=width_px,
            height_px=height_px,
            prompt=prompt_info,
            extra_pnginfo=extra_pnginfo,
        )

        compressed_frames = compress_image_tensor_webp(save_images)
        compressed_bootstrap = compress_image_tensor_webp(bootstrap_frame)

        output = {
            "latent": samples.cpu().contiguous(),
            "bootstrap_latent": samples[:, :, :1, :, :].cpu().contiguous(),
            "compressed_frames": compressed_frames.cpu().contiguous(),
            "compressed_bootstrap": compressed_bootstrap.cpu().contiguous(),
            "ltx_l2v_format_v0": torch.tensor([]),
        }
        save_torch_file(
            output,
            output_path,
            metadata={
                SEGMENT_METADATA_KEY: metadata.model_dump_json(exclude={"prompt", "extra_pnginfo"}),
                "prompt": json.dumps(prompt_info),
                "extra_pnginfo": json.dumps(extra_pnginfo),
            },
        )

        results: list[FileLocator] = [{"filename": filename, "subfolder": subfolder, "type": "output"}]

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
                    options=get_folders_in_outdir(depth=2),
                    tooltip="Subfolder to load segments from. Can be connected to save node's 'subfolder' output.",
                ),
                io.Int.Input(
                    "segment_idx",
                    default=0,
                    min=0,
                    tooltip="Segment index to load.",
                ),
                io.Int.Input(
                    "iteration",
                    default=0,
                    min=0,
                    tooltip="Iteration index to load. 0 loads the latest iteration for the segment.",
                ),
            ],
            outputs=[
                io.Latent.Output(display_name="latent"),
                io.Image.Output(display_name="frames"),
                LTXLat2VidSegmentIO.Output(display_name="metadata"),
            ],
        )

    @classmethod
    def execute(
        cls,
        video_folder: str,
        segment_idx: int,
        iteration: int,
    ) -> io.NodeOutput:
        segment_path = _find_segment_file(video_folder, segment_idx, iteration)
        state_dict, segment_data, _, _ = load_segment_file(segment_path)

        if "latent" not in state_dict:
            raise ValueError("Segment file is missing required 'latent' tensor")
        if "frames" not in state_dict and "compressed_frames" not in state_dict:
            raise ValueError("Segment file does not contain a frame tensor")

        samples = state_dict["latent"].cpu().contiguous()
        if "frames" in state_dict:
            frames = state_dict["frames"].cpu().contiguous()
        elif "compressed_frames" in state_dict:
            frames = state_dict["compressed_frames"].cpu().contiguous()
            pbar = ProgressBar(total=frames.shape[0])
            frames = decompress_image_tensor_webp(
                frames, (segment_data.width_px, segment_data.height_px), as_float=True, pbar=pbar
            )

        latent: dict[str, torch.Tensor] = {"samples": samples}

        return io.NodeOutput(
            latent,
            frames,
            segment_data,
        )


class LTXLat2VidInplaceNood(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LTXLat2VidInplaceNood",
            display_name="LTX Lat2Vid Inplace",
            category="noodles/ltx",
            inputs=[
                io.Vae.Input("vae", optional=True),
                io.Int.Input(
                    "noise_seed",
                    display_name="Noise Seed",
                    default=0,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    control_after_generate=True,
                    tooltip="Seed for the deterministic seed generator.",
                ),
                io.Latent.Input(
                    "latent",
                    tooltip="Empty latent tensor for this segment.",
                ),
                io.Latent.Input(
                    "prev_latent",
                    tooltip="Latent tensor from previous segment.",
                ),
                LTXLat2VidSegmentIO.Input(
                    "prev_metadata",
                    display_name="metadata",
                    tooltip="Metadata from the previous segment.",
                ),
                io.Int.Input(
                    "overlap_k",
                    default=6,
                    min=2,
                    max=512,
                    tooltip="Number of overlapped latents to carry into the next segment.",
                ),
                io.Combo.Input("bootstrap_mode", options=BootstrapMode, default=BootstrapMode.DummyLatent),
                io.Float.Input(
                    "bootstrap_strength",
                    default=1.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    display_name="strength",
                    tooltip="Strength of the bootstrap frame when doing inplace I2V. Should be 1.0 most of the time.",
                ),
                io.Combo.Input("mask_strat", options=MaskStrategy, default=MaskStrategy.CosineDecayV1),
                MaskParamsIO.Input("mask_params", optional=True),
            ],
            outputs=[
                io.Latent.Output(display_name="latent"),
                io.Int.Output(display_name="noise_seed"),
                LTXLat2VidSegmentIO.Output(display_name="metadata"),
                BootstrapModeIO.Output(display_name="bootstrap_mode"),
            ],
        )

    @classmethod
    def execute(
        cls,
        vae: VAE | None,
        noise_seed: int,
        latent: LatentInput,
        prev_latent: LatentInput,
        prev_metadata: LTXLat2VidSegmentData | str,
        overlap_k: int,
        bootstrap_mode: BootstrapMode,
        bootstrap_strength: float,
        mask_strat: MaskStrategy,
        mask_params: MaskParams | dict | None,
    ) -> io.NodeOutput:
        prev_segment_data = LTXLat2VidSegmentData.model_validate_any(prev_metadata)

        prev_samples: torch.Tensor = prev_latent["samples"]
        next_samples: torch.Tensor = latent["samples"]

        if prev_samples.ndim != 5 or next_samples.ndim != 5:
            raise ValueError(
                f"Expected both prev and next latent samples to have shape [B,C,T,H,W], got {prev_samples.shape!r} and {next_samples.shape!r}"
            )

        if overlap_k < 2:
            raise ValueError("overlap_k must be at least 2")
        if next_samples.shape[2] < overlap_k:
            raise ValueError(f"next_latent has {next_samples.shape[2]} latents but overlap_k is {overlap_k}")

        if prev_samples.shape[2] < overlap_k:
            raise ValueError(
                f"prev_latent has {prev_samples.shape[2]} usable latents but overlap_k is {overlap_k}"
            )

        overlap_chunk = prev_samples[:, :, -overlap_k:, :, :].clone()
        source_latent = overlap_chunk[:, :, 0:1, :, :]
        carried_latents = overlap_chunk[:, :, 1:, :, :]

        if bootstrap_mode == BootstrapMode.SegmentZero:
            raise ValueError(
                "BootstrapMode.SegmentZero is only valid for the first segment and cannot be used for continuation"
            )

        new_seed = _deterministic_seed(prev_segment_data, seed=noise_seed, overlap_count=overlap_k)

        if bootstrap_mode == BootstrapMode.RawLatent:
            bootstrap_latent = source_latent
        elif bootstrap_mode == BootstrapMode.DummyLatent:
            bootstrap_latent = next_samples[:, :, :1, :, :].clone()
        elif bootstrap_mode == BootstrapMode.VAERoundtrip:
            if vae is None:
                raise ValueError("VAE is required for BootstrapMode.VAERoundtrip")
            decoded = vae.decode(source_latent)
            if decoded.ndim == 4:
                # Expected image decode shape [N, H, W, C]
                last_frame = decoded[-1:, :, :, :3]
            else:
                raise ValueError(f"Unexpected decoded latent shape from VAE: {decoded.shape!r}")

            encoded = vae.encode(last_frame)
            if encoded.ndim == 4:
                encoded = encoded.unsqueeze(2)
            if encoded.ndim != 5:
                raise ValueError(f"Unexpected encoded latent shape from VAE: {encoded.shape!r}")

            bootstrap_latent = encoded[:, :, :1, :, :]
        else:
            raise ValueError(f"Unknown bootstrap mode: {bootstrap_mode}")

        next_samples[:, :, 0:1, :, :] = bootstrap_latent
        next_samples[:, :, 1:overlap_k, :, :] = carried_latents[:, :, : overlap_k - 1, :, :]

        strengths = _compute_overlap_strengths(
            total_k=overlap_k,
            bootstrap_mode=bootstrap_mode,
            bootstrap_strength=bootstrap_strength,
            mask_strat=mask_strat,
            mask_params=mask_params,
        )

        noise_mask = torch.ones(
            (next_samples.shape[0], 1, next_samples.shape[2], 1, 1),
            dtype=next_samples.dtype,
            device=next_samples.device,
        )
        strengths_tensor = torch.tensor(strengths, dtype=next_samples.dtype, device=next_samples.device)

        noise_mask[:, :, :overlap_k, :, :] = 1.0 - strengths_tensor.view(1, 1, overlap_k, 1, 1)

        next_segment_idx = int(prev_segment_data.segment_idx) + 1
        next_start_frame = (
            int(prev_segment_data.start_frame) + int(prev_segment_data.n_frames) - (8 * (overlap_k - 1))
        )

        next_metadata = prev_segment_data.model_copy(
            update={
                "parent_id": prev_segment_data.segment_id,
                "segment_id": ULID(),
                "segment_idx": next_segment_idx,
                "start_frame": next_start_frame,
                "overlap_k": overlap_k,
                "bootstrap_mode": bootstrap_mode,
                "mask_strat": mask_strat,
                "mask_params": mask_params,
            }
        )

        out_latent: dict[str, torch.Tensor] = {
            "samples": next_samples,
            "noise_mask": noise_mask,
        }
        return io.NodeOutput(
            out_latent,
            new_seed,
            next_metadata,
            bootstrap_mode,
        )


class LTXLat2VidGetNextSegmentSaveDataNood(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LTXLat2VidGetNextSegmentSaveDataNood",
            display_name="LTX Lat2Vid Next Segment Data",
            category="noodles/ltx",
            inputs=[
                LTXLat2VidSegmentIO.Input(
                    id="metadata", display_name="metadata", tooltip="Previous segment metadata to unpack"
                ),
            ],
            outputs=[
                MaskParamsIO.Output(display_name="mask_params"),
                io.Int.Output(display_name="overlap_k"),
                BootstrapModeIO.Output(display_name="bootstrap_mode"),
                MaskStrategyIO.Output(display_name="mask_strat"),
                io.Int.Output(display_name="width_px"),
                io.Int.Output(display_name="height_px"),
                io.Int.Output(display_name="n_frames_batch"),
                io.Int.Output(display_name="next_start_frame"),
            ],
        )

    @classmethod
    def execute(cls, metadata: LTXLat2VidSegmentData) -> io.NodeOutput:
        # work out the last frame index of the current segment so we can calculate the next segment's start frame
        last_end_frame = metadata.start_frame + metadata.n_frames
        # calculate how many frames of overlap, accounting for the first overlapped latent only actually counting for one frame of overlap
        n_overlap_frames = (metadata.overlap_k * 8) - 7
        # the next segment should start right after the non-overlapped frames of the previous segment
        next_start_frame = last_end_frame - n_overlap_frames

        return io.NodeOutput(
            metadata.mask_params,
            metadata.overlap_k,
            metadata.bootstrap_mode,
            metadata.mask_strat,
            metadata.width_px,
            metadata.height_px,
            metadata.n_frames_batch,
            next_start_frame,
        )


class LTXLat2VidGetNextSegmentDataNood(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LTXLat2VidGetNextSegmentDataNood",
            display_name="LTX Lat2Vid Next Segment Save Data",
            category="noodles/ltx",
            inputs=[
                LTXLat2VidSegmentIO.Input(
                    id="metadata", display_name="metadata", tooltip="Previous segment metadata to unpack"
                ),
            ],
            outputs=[
                MaskParamsIO.Output(display_name="mask_params"),
                ComfyULID.Output(display_name="video_id"),
                ComfyULID.Output(display_name="parent_id"),
                io.String.Output(display_name="subfolder"),
                io.String.Output(display_name="video_name"),
                io.Int.Output(display_name="next_segment_idx"),
                io.Int.Output(display_name="next_start_frame"),
                io.Int.Output(display_name="n_frames_batch"),
                io.Int.Output(display_name="overlap_k"),
                BootstrapModeIO.Output(display_name="bootstrap_mode"),
                MaskStrategyIO.Output(display_name="mask_strat"),
            ],
        )

    @classmethod
    def execute(cls, metadata: LTXLat2VidSegmentData) -> io.NodeOutput:
        # work out the last frame index of the current segment so we can calculate the next segment's start frame
        last_end_frame = metadata.start_frame + metadata.n_frames
        # calculate how many frames of overlap, accounting for the first overlapped latent only actually counting for one frame of overlap
        n_overlap_frames = (metadata.overlap_k * 8) - 7
        # the next segment should start right after the non-overlapped frames of the previous segment
        next_start_frame = last_end_frame - n_overlap_frames

        # increment segment idx for the next segment
        next_segment_idx = metadata.segment_idx + 1

        return io.NodeOutput(
            metadata.mask_params,
            metadata.video_id,
            metadata.segment_id,
            metadata.subfolder,
            metadata.video_name,
            next_segment_idx,
            next_start_frame,
            metadata.n_frames_batch,
            metadata.overlap_k,
            metadata.bootstrap_mode,
            metadata.mask_strat,
        )


class LTXMaskParamsNood(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LTXMaskParamsNood",
            display_name="LTX Mask Params",
            category="noodles/ltx",
            inputs=[
                MaskStrategyIO.Input("mask_strat", default=MaskStrategy.CosineDecayV1),
                io.Int.Input(
                    "hard_mask_k",
                    default=2,
                    min=1,
                    max=32,
                    tooltip="Number of overlapped latents to hard-mask at strength 1.0 before decay.",
                ),
                io.Float.Input(
                    "w_max",
                    default=1.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    display_name="w_max",
                    tooltip="Maximum strength for the final latent in the overlap window.",
                ),
                io.Float.Input(
                    "w_min",
                    default=0.2,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    display_name="w_min",
                    tooltip="Minimum strength for the final latent in the overlap window.",
                ),
                io.Float.Input(
                    "decay_sigma",
                    default=0.4,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="Sigma for half-Gaussian decay curve.",
                ),
            ],
            outputs=[
                MaskParamsIO.Output(display_name="mask_params"),
                MaskStrategyIO.Output(display_name="mask_strat"),
            ],
        )

    @classmethod
    def execute(
        cls,
        mask_strat: MaskStrategy,
        hard_mask_k: int,
        w_max: float,
        w_min: float,
        decay_sigma: float,
    ) -> dict:
        mask_params = MaskParams(
            hard_mask_k=hard_mask_k,
            w_max=w_max,
            w_min=w_min,
            decay_sigma=decay_sigma,
        )

        return io.NodeOutput(mask_params, mask_strat)
